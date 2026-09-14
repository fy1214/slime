#!/bin/bash
# Sourced in EVERY container of the deterministic run (ray head, ray workers,
# driver) BEFORE `ray start` / train.py.
#
# Why before ray start: the alignment hooks read these from inside the Megatron
# ACTOR process (e.g. enable_sglang_global_batch_invariant_ops() reads
# SGLANG_DEEPGEMM_BATCH_INVARIANT).  Ray actors inherit the raylet's environment,
# so exporting only in the driver would silently disable half the alignment.
# The customer's script gets this via `ray job submit --runtime-env-json`;
# we get it by sourcing here on every node.
#
# Content = slime/backends/megatron_utils/alignment/env.py::alignment_env()
# plus the Qwen3 overrides from scripts/run-qwen3-30B-A3B-deterministic.sh.

W=/lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/investigation/nvfp4_router_long_20260912_v1
export PYTHONPATH=/root/Megatron-LM:/root/slime:/root/miniTransformer:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export NO_PROXY="${NO_PROXY:-*}" no_proxy="${no_proxy:-*}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

# GB200 4-GPU node = ~890GB host shared by 4 actors; ray's default 0.95 kill
# threshold trips before the true physical limit (carried over from the QAT8 runs).
export RAY_memory_usage_threshold=0.99

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

# ---- shared by both arms ----------------------------------------------------
#  * colocate: Megatron grabs memory before SGLang, so the balance guard is moot
#  * dispatch-token cap: a capacity knob, not a numerics knob (dp=4 needs the bump)
#  * NVLink topology: describes the box, not the recipe
export NCCL_P2P_LEVEL=NVL
export NCCL_NVLS_ENABLE="$HAS_NVLINK"
# low_latency deepep routes PREFILL through low_latency_dispatch too; without staging a
# whole chunked-prefill chunk goes out in one call and trips DeepEP's
# `x.size(0) <= num_max_dispatch_tokens_per_rank` assert (default cap 128).  This is a
# capacity mechanism, not an alignment knob, so BOTH arms need it at dp=4.
export SGLANG_DEEPEP_LL_PREFILL_STAGING=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-bfloat16}"
export DSA_KV_FP8_QAT=$([ "$SGLANG_KV_CACHE_DTYPE" = "fp8_e4m3" ] && echo 1 || echo 0)
export DSA_KV_FP8_QAT_BLOCK_SIZE=128
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

# =============================================================================
# EVERYTHING BELOW IS THE ALIGNMENT STACK -- MODE=det ONLY.
# Setting these in the bf16 control is not merely wasteful, it is WRONG:
# MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS activates the megatron-sglang-aligned
# patch branch, which reads self.input_layernorm.weight -- but without
# --spec qwen3_moe_aligned that module is an IdentityOp (the norm is fused into
# linear_qkv), so the bf16 arm dies with
#   AttributeError: 'IdentityOp' object has no attribute 'weight'.
# The customer's own run-qwen3-30B-A3B-bf16.sh sets none of these.
# =============================================================================
if true; then   # NVFP4 always needs the alignment env
  # ---- deterministic collectives / matmul -------------------------------------
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export NCCL_ALGO="^NVLS"
  export CUBLAS_WORKSPACE_CONFIG=":4096:8"
  export TORCH_COMPILE_DISABLE=1
  export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
  export TE_DISABLE_FA3=TRUE
  export NVSHMEM_DISABLE_NCCL=1
  # ---- DeepGEMM batch-invariant FP8 forward -----------------------------------
  export SGLANG_DEEPGEMM_BATCH_INVARIANT=1
  export SGLANG_DEEPGEMM_PAD_EXPERT_M=1
  export SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
  export SGLANG_JIT_KERNEL_EXTRA_PATH=/root/slime/slime/backends/sglang_utils/jit_kernels
  # Qwen3 moe_ffn=768 -> 6 scale groups; the v2 masked-quant kernel refuses G%16/G%4.
  export SGLANG_MASKED_GEMM_FAST_ACT="${SGLANG_MASKED_GEMM_FAST_ACT:-0}"
  # ---- DeepEP low-latency dispatch --------------------------------------------
  # DELTA: customer runs dp_size=8; at dp_size=4 each rank carries ~2x the tokens,
  # so the low-latency dispatch buffer must be sized up or dispatch overflows.
  # ---- DSA indexer (GLM-5 path; inert for Qwen3 GQA but kept identical) --------
  export SGLANG_DSA_FUSE_TOPK=0
  export SGLANG_DISABLE_DSA_INDEXER_FUSION=1
  export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0
  export INDEXER_ROPE_NEOX_STYLE=0
  # ---- Megatron borrows SGLang's aligned kernels -------------------------------
  export MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS=1
  export MEGATRON_USE_SGLANG_FP8_INDEXER=1
  export MEGATRON_USE_SGLANG_ROUTER_GEMM=1
  export MEGATRON_USE_SGLANG_ROPE=1
  export MEGATRON_USE_SGLANG_SPARSE_MLA=1
  # ---- KV cache ----------------------------------------------------------------
  # ---- colocate ----------------------------------------------------------------
  # ---- Qwen3 aligned attention plugin -----------------------------------------
  export QWEN3_MOE_ALIGNED_SPEC="${QWEN3_MOE_ALIGNED_SPEC:-1}"
  export QWEN3_ALIGNED_USE_FUSED_QK_ROPE="${QWEN3_ALIGNED_USE_FUSED_QK_ROPE:-1}"
  export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK="${SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK:-1}"
fi

# wandb: one run per EXP, merged across chained jobs (creds from mounted ~/.netrc).

# ---------------------------------------------------------------------------
# Megatron-LM #1861: segfault in the checkpoint serializer
#   crc32_16bytes -> mz_crc32 -> mz_zip_writer_add_mem_ex_v2 -> PyTorchStreamWriter::writeRecord
# The async save forks a child while CUDA + threads are live, so the child inherits a
# broken memory state.  The issue (closed 2025-12-15) lists two independently confirmed
# fixes; we apply BOTH, in every Ray container because the actors serialize remotely.
#   1. filesystem_async.py:266  mp.get_context("fork") -> "spawn"
#   2. preload_tensors(non_blocking=True) -> False   (no async D2H staging)
# Reproduced here on GB200/aarch64: FP8 model-only saved fine, but FP8+optimizer and
# bf16 model-only both segfault at the same frame.
_FSA=/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py
if [ -f "$_FSA" ]; then
  # fork->spawn was tried and REVERTED: spawn must pickle the payload and Megatron's
  # write-bucket object is not picklable -> "ForkingPickler.dump ... IndexError: tuple index
  # out of range" (jobs 2749702/2749703).  non_blocking=False alone is enough -- it forces a
  # synchronous D2H copy so the data is fully materialised in host memory BEFORE the fork,
  # which is what made the child's inherited state inconsistent in the first place.
  # Measured on GB200/aarch64 (all with optimizer in the ckpt):
  #   fork  + non_blocking=True  -> segfault   (2749609/2749610)
  #   fork  + non_blocking=False -> segfault   (2749750/2749751)  <- the issue's non_blocking
  #                                                                  workaround does NOT hold here
  #   spawn + non_blocking=False -> no segfault, but unpicklable payload (2749702/2749703)
  # So the fork itself is the problem.  Remove the subprocess entirely.
  python3 "$W/jobs/patch_ckpt_inproc.py" "$_FSA"
else
  echo "CKPT1861_FIX: $_FSA not found" >&2
fi

# TEGroupedLinear cannot SAVE a dist checkpoint when the FP8 recipe carries no global
# FP8 meta (--fp8-recipe blockwise / MXFP8): _extra_state is an empty tensor, decoded to
# None, and _split_extra_state indexes it -> "TypeError: 'NoneType' object is not
# subscriptable" at the first save (MODE=fp8plain, job 2765230, 10 steps in).  The load
# path in the same file already guards for None; only the save path is missing it.
# Deterministic arms never pass --fp8-format, so they are unaffected either way.
_TEEXT=/root/Megatron-LM/megatron/core/extensions/transformer_engine.py
if [ -f "$_TEEXT" ]; then
  python3 "$W/jobs/patch_te_grouped_extra_state.py" "$_TEEXT"
else
  echo "TE_EXTRA_STATE_FIX: $_TEEXT not found" >&2
fi

export WANDB_MODE="${WANDB_MODE:-online}"
[ -n "${EXP:-}" ] && export WANDB_RUN_ID="${WANDB_ID:-$EXP}" WANDB_RESUME="${WANDB_RESUME_POLICY:-allow}"

# Diagnostic: synchronous dump of the route-preserving metadata self-check.
# Read inside the Megatron ACTOR, so it must be exported before ray start.
[ -n "${SLIME_ROUTE_DEBUG:-}" ] && export SLIME_ROUTE_DEBUG

ulimit -n 524288 2>/dev/null || true
echo "DET_PREP_OK nvlink=$HAS_NVLINK fast_act=$SGLANG_MASKED_GEMM_FAST_ACT dispatch_tokens=$SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK kv=$SGLANG_KV_CACHE_DTYPE"

# ---- NVFP4-only ----------------------------------------------------------
# Real NVFP4 rollout: per-token activation scale (their patch reads this).
export SGLANG_NVFP4_PERTOKEN_SCALE="${SGLANG_NVFP4_PERTOKEN_SCALE:-1}"
# NVFP4's cutlass DeepEP-LL does a FULL padded GEMM over [E, max_m] (FP8 does not),
# so max_m must stay small.  Their script defaults to 64; our FP8 path uses 128
# because dp=4 doubles tokens per rank -- that trade does NOT apply here.
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${NVFP4_MAX_M:-64}"   # unconditional: the copied FP8 block already set 128
echo "NVFP4_PREP_OK pertoken=$SGLANG_NVFP4_PERTOKEN_SCALE max_m=$SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"
