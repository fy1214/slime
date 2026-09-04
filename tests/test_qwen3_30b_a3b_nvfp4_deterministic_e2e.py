"""Deterministic train/rollout alignment gate for Qwen3-30B-A3B NVFP4 MoE.

Same shape as ``test_qwen3_30b_a3b_deterministic_e2e.py``, but Megatron MoE uses the
post-DeepEP cutlass NVFP4 pertoken path and SGLang rollout uses
``modelopt_fp4`` + ``cutlass`` + ``SGLANG_NVFP4_PERTOKEN_SCALE=1``.

GPU sizing (EP/DP/actor):
  - ``NUM_GPUS`` env overrides count (default 8 from parent test)
  - ``CUDA_VISIBLE_DEVICES`` selects physical cards; if ``NUM_GPUS`` is unset,
    the visible device count becomes the job size
  - when both are set, they must agree

Requires SGLang checkout with ``cutlass_moe_fp4_pertoken`` / ``fp4_utils`` and
``miniTransformer`` ``fp4_gemm`` installed in the runtime environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from test_qwen3_30b_a3b_deterministic_e2e import (
    NUM_GPUS as DEFAULT_NUM_GPUS,
    REPO_ROOT,
    _iface_ipv4,
    _pythonpath,
    _resolve_prompt_data,
    _run,
)

DEFAULT_MAX_TRAIN_ROLLOUT_DIFF = "9.999e-7"
DEFAULT_HF_MODEL = "/home/admin/mingfa/model/hf/Qwen3-30B-A3B-NVFP4"
DEFAULT_PROMPT_DATA = "/home/admin/mingfa/data/dapo-math-17k/dapo-math-17k.jsonl"
_FALLBACK_PROMPT_DATA = "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl"
DEFAULT_SGLANG_ROOT = "/sgl-workspace/sglang"
DEFAULT_MEGATRON_ROOT = "/root/Megatron-LM"


def _cuda_visible_device_ids() -> list[str] | None:
    """Parse CUDA_VISIBLE_DEVICES into physical ids, or None if unset."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _count_visible_gpus() -> int:
    """Count GPUs this process can use (respects CUDA_VISIBLE_DEVICES)."""
    cvd = _cuda_visible_device_ids()
    if cvd is not None:
        return len(cvd)
    visible = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return len([line for line in visible.stdout.splitlines() if line.strip()])


def _resolve_num_gpus() -> int:
    """Resolve EP/DP/actor GPU count.

    Priority:
      1) ``NUM_GPUS`` env (explicit override)
      2) ``len(CUDA_VISIBLE_DEVICES)`` when set
      3) parent default (8)

    When both ``NUM_GPUS`` and ``CUDA_VISIBLE_DEVICES`` are set, ``NUM_GPUS``
    must equal the visible count so Ray/EP size matches the device list.
    """
    cvd = _cuda_visible_device_ids()
    cvd_n = len(cvd) if cvd is not None else None
    raw = os.environ.get("NUM_GPUS")
    if raw is not None and raw.strip():
        n = int(raw)
        if cvd_n is not None and n != cvd_n:
            raise ValueError(
                f"NUM_GPUS={n} conflicts with CUDA_VISIBLE_DEVICES "
                f"({cvd_n} devices: {os.environ['CUDA_VISIBLE_DEVICES']})"
            )
        if n < 1:
            raise ValueError(f"NUM_GPUS must be >= 1, got {n}")
        return n
    if cvd_n is not None:
        return cvd_n
    return int(DEFAULT_NUM_GPUS)

_PREREQ_PROBE = """
import fp4_gemm
from sglang.srt.layers.moe.cutlass_moe import (
    cutlass_moe_fp4_pertoken,
    cutlass_moe_fp4_pertoken_deepep_ll,
)
from sglang.srt.layers.quantization.fp4_utils import nvfp4_grouped_gemm, nvfp4_quantize_pertoken
from sglang.srt.server_args import ServerArgs

assert hasattr(fp4_gemm, "grouped_cutlass_gemm_v2")
assert callable(cutlass_moe_fp4_pertoken)
assert callable(cutlass_moe_fp4_pertoken_deepep_ll)
assert callable(nvfp4_grouped_gemm)
assert "enable_fp32_moe_router" in ServerArgs.__dataclass_fields__
"""


def _deterministic_env(
    sglang_root: str,
    megatron_root: str,
    kv_cache_dtype: str,
) -> dict:
    from slime.backends.megatron_utils.alignment.env import alignment_env

    ifname = os.environ.get("MLP_SOCKET_IFNAME")
    env = alignment_env(kv_fp8_qat=kv_cache_dtype == "fp8_e4m3")
    env.update(
        {
            "PYTHONPATH": _pythonpath(sglang_root, megatron_root),
            "PYTHONUNBUFFERED": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
            "SGLANG_NVFP4_PERTOKEN_SCALE": os.environ.get("SGLANG_NVFP4_PERTOKEN_SCALE", "1"),
            "QWEN3_MOE_ALIGNED_SPEC": os.environ.get("QWEN3_MOE_ALIGNED_SPEC", "1"),
            "QWEN3_ALIGNED_USE_FUSED_QK_ROPE": os.environ.get(
                "QWEN3_ALIGNED_USE_FUSED_QK_ROPE", "1"
            ),
            "SGLANG_MASKED_GEMM_FAST_ACT": os.environ.get("SGLANG_MASKED_GEMM_FAST_ACT", "0"),
            "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": os.environ.get(
                "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK", "1"
            ),
            # Force off; parent DISABLE vars must not be inherited (SGLang
            # deprecation shim would copy them into ENABLE=true).
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
            "LD_LIBRARY_PATH": os.pathsep.join(
                part
                for part in (
                    os.environ.get("LD_LIBRARY_PATH", ""),
                    "/usr/local/lib/python3.12/dist-packages/torch/lib",
                )
                if part
            ),
        }
    )
    if ifname:
        env.update(
            {
                "GLOO_SOCKET_IFNAME": ifname,
                "TP_SOCKET_IFNAME": ifname,
                "NCCL_SOCKET_IFNAME": ifname,
                "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": ifname,
            }
        )
    if os.environ.get("NVSHMEM_IBGDA_NIC_HANDLER"):
        env["NVSHMEM_IBGDA_NIC_HANDLER"] = os.environ["NVSHMEM_IBGDA_NIC_HANDLER"]
    return env


def _skip_reason(sglang_root: str, megatron_root: str, num_gpus: int) -> str | None:
    if not __import__("shutil").which("nvidia-smi"):
        return "nvidia-smi not found (no GPUs)"
    try:
        n_visible = _count_visible_gpus()
    except Exception as exc:  # noqa: BLE001
        return f"failed to count GPUs: {exc}"
    if n_visible < num_gpus:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        hint = f" (CUDA_VISIBLE_DEVICES={cvd})" if cvd else ""
        return f"needs {num_gpus} GPUs, found {n_visible}{hint}"

    probe_env = {**os.environ, "PYTHONPATH": _pythonpath(sglang_root, megatron_root)}
    probe = subprocess.run(
        [sys.executable, "-c", _PREREQ_PROBE],
        capture_output=True,
        text=True,
        env=probe_env,
    )
    if probe.returncode != 0:
        return f"NVFP4 pertoken stack unavailable: {probe.stdout.strip() or probe.stderr.strip()}"

    layer_src = Path(f"{megatron_root}/megatron/core/transformer/transformer_layer.py")
    if not (layer_src.exists() and "_use_sglang_fused_residual_rmsnorm" in layer_src.read_text()):
        return f"Megatron root missing megatron-sglang-aligned.patch: {megatron_root}"

    return None


def _train_args(
    hf_model: str,
    prompt_data: str,
    threshold: str,
    rollout_dump: str,
    kv_cache_dtype: str,
    *,
    num_gpus: int,
    rollout_max_response_len: int = 512,
    num_layers_override: int | None = None,
    rollout_batch_size: int | None = None,
) -> str:
    num_layers = num_layers_override if num_layers_override is not None else 48
    if rollout_batch_size is None:
        rollout_batch_size = int(os.environ.get("SLIME_E2E_ROLLOUT_BATCH_SIZE", "8"))
    context_parallel_size = int(os.environ.get("SLIME_E2E_CONTEXT_PARALLEL_SIZE", "1"))
    layer_list = " ".join(str(i) for i in range(num_layers))

    groups = [
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {num_gpus} "
        f"--rollout-num-gpus {num_gpus} --colocate "
        "--update-weight-mode full --update-weight-transport nccl "
        "--update-weight-buffer-size 2147483648 --no-check-for-nan-in-loss-and-grad",
        f"--num-layers {num_layers} --hidden-size 2048 --ffn-hidden-size 6144 "
        "--num-attention-heads 32 --num-query-groups 4 --kv-channels 128 "
        f"--moe-layer-freq [1]*{num_layers} "
        "--num-experts 128 --moe-router-topk 8 --moe-grouped-gemm "
        "--moe-ffn-hidden-size 768 "
        "--moe-router-score-function softmax "
        "--moe-router-topk-scaling-factor 1.0 "
        "--moe-aux-loss-coeff 0 --moe-router-dtype fp32 "
        "--vocab-size 151936 "
        "--make-vocab-size-divisible-by 16 "
        "--group-query-attention --qk-layernorm --disable-bias-linear "
        "--swiglu --untie-embeddings-and-output-weights "
        "--position-embedding-type rope --no-position-embedding "
        "--normalization RMSNorm "
        "--rotary-base 1000000 --norm-epsilon 1e-6",
        f"--hf-checkpoint {hf_model} --load {hf_model} --ref-load {hf_model}",
        f"--prompt-data {prompt_data} --input-key prompt --label-key label --apply-chat-template "
        f"--rollout-shuffle --rm-type deepscaler --rollout-batch-size {rollout_batch_size} --n-samples-per-prompt 1 "
        f"--global-batch-size {rollout_batch_size} --num-rollout 1 --rollout-max-context-len 4096 "
        f"--rollout-max-response-len {rollout_max_response_len} "
        "--rollout-temperature 1.0 --rollout-top-p 1.0 "
        f"--save-debug-rollout-data {rollout_dump}",
        "--optimizer adam --lr 2e-6 --lr-warmup-iters 0 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 --no-load-optim --no-save-optim --use-stateless-adam",
        "--advantage-estimator grpo --kl-loss-coef 0 --kl-loss-type low_var_kl --kl-coef 0 --entropy-coef 0 "
        "--eps-clip 0.2 --eps-clip-high 0.28 --disable-grpo-std-normalization --reset-optimizer-states",
        f"--tensor-model-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size {context_parallel_size} --expert-model-parallel-size {num_gpus} --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 --data-pad-size-multiplier 512 "
        "--log-probs-chunk-size 1024",
        f"--rollout-num-gpus-per-engine {num_gpus} --sglang-server-concurrency 128 "
        "--sglang-mem-fraction-static 0.50 --sglang-enable-dp-attention --sglang-enable-dp-lm-head "
        f"--sglang-ep-size {num_gpus} --sglang-dp-size {num_gpus} --sglang-moe-dp-size 1 "
        f"--sglang-moe-dense-tp-size 1 --sglang-moe-a2a-backend deepep --sglang-deepep-mode {os.environ.get('SLIME_E2E_DEEPEP_MODE', 'low_latency')} "
        "--sglang-quantization modelopt_fp4 --sglang-moe-runner-backend cutlass "
        f"--sglang-page-size 64 --sglang-kv-cache-dtype {kv_cache_dtype} "
        "--sglang-attention-backend fa4 "
        "--sglang-enable-fused-qk-norm-rope "
        "--sglang-chunked-prefill-size 4096 --sglang-context-length 8192 "
        "--sglang-max-prefill-tokens 4096 --sglang-enable-fp32-moe-router "
        "--sglang-enable-deterministic-inference --sglang-disable-prefill-cuda-graph "
        "--sglang-disable-flashinfer-autotune "
        "--sglang-cuda-graph-max-bs-decode 64 --sglang-watchdog-timeout 7200 --sglang-dist-timeout 1800 "
        "--sglang-trust-remote-code",
        "--attention-dropout 0 --hidden-dropout 0 --attention-softmax-in-fp32 "
        "--accumulate-allreduce-grads-in-fp32 --attention-backend flash "
        "--moe-token-dispatcher-type flex --moe-enable-deepep "
        '--train-env-vars {"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","CUDA_LAUNCH_BLOCKING":"1"} '
        "--custom-megatron-before-log-prob-hook-path "
        "slime.backends.megatron_utils.alignment.nvfp4_alignment.enable_nvfp4_alignment_all_forward "
        "--custom-megatron-before-train-step-hook-path "
        "slime.backends.megatron_utils.alignment.nvfp4_alignment.enable_nvfp4_alignment_all_forward_before_train_step "
        f"--megatron-cutlass-nvfp4-moe-forward-layers {layer_list} "
        "--deterministic-mode --skip-eval-before-train "
        f"--ci-test --ci-disable-kl-checker --ci-train-rollout-logprob-abs-diff-threshold {threshold}",
    ]

    if os.environ.get("QWEN3_MOE_ALIGNED_SPEC", "1") == "1":
        groups.append(
            "--spec slime_plugins.models.qwen3_moe_aligned get_qwen3_moe_aligned_spec"
        )

    if os.environ.get("SLIME_E2E_DISABLE_DECODE_CG", "0") == "1":
        groups.append("--sglang-disable-decode-cuda-graph")

    return " ".join(groups)


def run_gate(*, rollout_batch_size: int | None = None) -> None:
    sglang_root = os.environ.get("SGLANG_ROOT", DEFAULT_SGLANG_ROOT)
    megatron_root = os.environ.get("MEGATRON_ROOT", DEFAULT_MEGATRON_ROOT)
    hf_model = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
    prompt_data = _resolve_prompt_data(os.environ.get("PROMPT_DATA", DEFAULT_PROMPT_DATA))
    threshold = os.environ.get("MAX_TRAIN_ROLLOUT_DIFF", DEFAULT_MAX_TRAIN_ROLLOUT_DIFF)
    kv_cache_dtype = os.environ.get("SGLANG_KV_CACHE_DTYPE", "bfloat16")
    num_gpus = _resolve_num_gpus()

    reason = _skip_reason(sglang_root, megatron_root, num_gpus)
    if reason is not None:
        msg = f"Qwen3-30B-A3B NVFP4 deterministic gate skipped: {reason}"
        print(msg, flush=True)
        pytest.skip(msg)

    if not Path(hf_model, "config.json").exists():
        raise FileNotFoundError(f"HF checkpoint missing: {hf_model}/config.json")
    if not Path(prompt_data).exists():
        raise FileNotFoundError(f"Prompt data missing: {prompt_data}")

    master_addr = "127.0.0.1"
    ifname = os.environ.get("MLP_SOCKET_IFNAME")
    if ifname and (ip := _iface_ipv4(ifname)):
        master_addr = ip
    master_port = os.environ.get("MASTER_PORT", "29500")

    env = {**os.environ, **_deterministic_env(sglang_root, megatron_root, kv_cache_dtype)}
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(proxy_var, None)
    # SGLang deprecation shim: SGL*_DISABLE_* copies into ENABLE and re-arms the check.
    for bad in (
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK",
        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK",
    ):
        env.pop(bad, None)
    env["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "0"
    env["MASTER_ADDR"] = master_addr
    env["RAY_ADDRESS"] = f"{master_addr}:{master_port}"
    # Keep CUDA_VISIBLE_DEVICES for Ray/train children when the caller set it.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    env["NUM_GPUS"] = str(num_gpus)

    if rollout_batch_size is None:
        rollout_batch_size = int(os.environ.get("SLIME_E2E_ROLLOUT_BATCH_SIZE", "8"))

    print(
        f"Running Qwen3-30B-A3B NVFP4 deterministic train/rollout gate "
        f"(gpus={num_gpus}, CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<all>')}, "
        f"samples={rollout_batch_size}, logprob limit={float(threshold):g}) ...",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="qwen3_30b_nvfp4_gate_") as tmp:
        rollout_dump = os.path.join(tmp, "rollout_data", "{rollout_id}.pt")
        argv = _train_args(
            hf_model,
            prompt_data,
            threshold,
            rollout_dump,
            kv_cache_dtype,
            num_gpus=num_gpus,
            rollout_max_response_len=32,
            rollout_batch_size=rollout_batch_size,
        ).split()

        _run(["pkill", "-9", "sglang"], check=False)
        _run(["ray", "stop", "--force"], check=False, env=env)
        try:
            _run(
                ["ray", "start", "--head", "--node-ip-address", master_addr, "--num-gpus", str(num_gpus),
                 "--port", master_port, "--disable-usage-stats", "--include-dashboard=false"],
                env=env,
            )
            code, _out = _run(
                [sys.executable, "-u", "train.py", *argv],
                env=env,
                cwd=str(REPO_ROOT),
                stream=True,
            )
            assert code == 0, (
                f"train.py exited {code} (train/rollout bound {float(threshold):g} likely breached)"
            )
        finally:
            _run(["ray", "stop", "--force"], check=False, env=env)

    print(
        f"Qwen3-30B-A3B NVFP4 deterministic train/rollout gate PASSED "
        f"(gpus={num_gpus}, logprob limit={float(threshold):g})",
        flush=True,
    )


def test_qwen3_30b_a3b_nvfp4_deterministic_train_rollout_alignment(request):
    run_gate(rollout_batch_size=request.config.getoption("rollout_batch_size"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-rs"]))
