#!/usr/bin/env bash
# Patch SGLang 0.5.15.post1 with NVFP4 pertoken cutlass support (fp4_utils + helpers).
# Usage (inside container):
#   bash scripts/setup-sglang-nvfp4-pertoken.sh /path/to/nvfp4-pertoken-worktree
set -euo pipefail

WORKTREE="${1:-/root/nvfp4-pertoken-worktree}"
SGLANG_ROOT="${SGLANG_ROOT:-/sgl-workspace/sglang}"
PY="${SGLANG_ROOT}/python/sglang/srt"

SRC="${WORKTREE}/python/sglang/srt"
if [[ ! -f "${SRC}/layers/quantization/fp4_utils.py" ]]; then
  echo "Missing ${SRC}/layers/quantization/fp4_utils.py" >&2
  exit 1
fi

echo "Merging NVFP4 pertoken helpers into fp4_utils.py ..."
python3 - <<'PY' "${PY}/layers/quantization/fp4_utils.py" "${SRC}/layers/quantization/fp4_utils.py"
import sys
from pathlib import Path

dst_path = Path(sys.argv[1])
src_path = Path(sys.argv[2])
dst = dst_path.read_text()
if "def nvfp4_quantize_pertoken" in dst:
    print("nvfp4_quantize_pertoken already present")
    raise SystemExit(0)

src = src_path.read_text()
marker = "# ---------------------------------------------------------------------------\n# NVFP4 per-token quantize"
start = src.index(marker)
pertoken_block = src[start:]
if not dst.endswith("\n"):
    dst += "\n"
merged = dst + "\n" + pertoken_block
if "NamedTuple" not in merged.split("NvFp4QuantResult")[0]:
    merged = merged.replace(
        "from typing import TYPE_CHECKING, Optional",
        "from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Tuple",
        1,
    )
old_import = (
    "from sglang.srt.utils.common import (\n"
    "    get_device_capability,\n"
    "    is_cuda,\n"
    "    is_sm100_supported,\n"
    ")"
)
new_import = (
    "from sglang.srt.utils.common import (\n"
    "    get_device_capability,\n"
    "    is_cuda,\n"
    "    is_sm100_supported,\n"
    "    is_sm120_supported,\n"
    ")"
)
if old_import in merged and "is_sm120_supported" not in merged.split("NvFp4QuantResult")[0]:
    merged = merged.replace(old_import, new_import, 1)
if "import torch.nn.functional as F" not in merged.split("NvFp4QuantResult")[0]:
    merged = merged.replace(
        "import torch\n",
        "import torch\nimport torch.nn.functional as F\n",
        1,
    )
dst_path.write_text(merged)
print("Merged pertoken helpers into fp4_utils.py")
PY

echo "Syncing cutlass_moe_fp4_pertoken helpers from worktree ..."
python3 - <<'PY' "${PY}/layers/moe/cutlass_moe.py" "${SRC}/layers/moe/cutlass_moe.py"
import sys
from pathlib import Path

dst_path = Path(sys.argv[1])
src_path = Path(sys.argv[2])
dst = dst_path.read_text()
src = src_path.read_text()


def extract(text: str, name: str, next_names: list[str]) -> str:
    marker = f"def {name}("
    if marker not in text:
        raise SystemExit(f"missing {marker} in {src_path}")
    start = text.index(marker)
    end = len(text)
    for nxt in next_names:
        i = text.find(f"\ndef {nxt}(", start + 1)
        if i != -1:
            end = min(end, i)
    return text[start:end].rstrip() + "\n\n"


pertoken = extract(src, "cutlass_moe_fp4_pertoken", ["cutlass_moe_fp4_pertoken_deepep_ll"])
ll = extract(src, "cutlass_moe_fp4_pertoken_deepep_ll", [])
for marker in (
    "def cutlass_moe_fp4_pertoken_deepep_ll(",
    "def cutlass_moe_fp4_pertoken(",
):
    if marker in dst:
        dst = dst[: dst.index(marker)].rstrip() + "\n\n"
dst_path.write_text(dst + pertoken + ll)
print("Wrote cutlass_moe_fp4_pertoken + cutlass_moe_fp4_pertoken_deepep_ll")
PY

echo "Patching modelopt_quant.py for NVFP4 pertoken ..."
python3 - <<'PY' "${PY}/layers/quantization/modelopt_quant.py"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

if "NVFP4_PERTOKEN_SCALE" not in text:
    needle = "MOE_NVFP4_DISPATCH = envs.SGLANG_MOE_NVFP4_DISPATCH.get()"
    insert = needle + "\nNVFP4_PERTOKEN_SCALE = envs.SGLANG_NVFP4_PERTOKEN_SCALE.get()"
    if needle not in text:
        raise SystemExit("MOE_NVFP4_DISPATCH anchor missing")
    text = text.replace(needle, insert, 1)

if "layer.w1_pertoken_wgt_scale" not in text:
    anchor = "                )  # k\n"
    # Use copy_or_rebind so decode CUDA graph keeps stable addresses after
    # colocate weight sync + process_weights_after_loading (see
    # scripts/patch_nvfp4_cg_decode_rebind.py).
    block = '''                )  # k

            if NVFP4_PERTOKEN_SCALE:
                w13_bs = layer.w13_blockscale_swizzled
                w2_bs = layer.w2_blockscale_swizzled
                num_local = w13_bs.shape[0]

                w1_sf_stride = w13_bs.shape[1] * w13_bs.shape[2]
                copy_or_rebind_param(
                    layer,
                    "w1_blockscale_flat",
                    w13_bs.reshape(-1).contiguous(),
                )
                copy_or_rebind_param(
                    layer,
                    "w1_row_offsets",
                    (
                        torch.arange(num_local + 1, dtype=torch.int32, device=device)
                        * layer.w13_weight.shape[1]
                    ),
                )
                copy_or_rebind_param(
                    layer,
                    "w1_scale_offsets",
                    (
                        torch.arange(num_local + 1, dtype=torch.int64, device=device)
                        * w1_sf_stride
                    ),
                )

                w2_sf_stride = w2_bs.shape[1] * w2_bs.shape[2]
                copy_or_rebind_param(
                    layer,
                    "w2_blockscale_flat",
                    w2_bs.reshape(-1).contiguous(),
                )
                copy_or_rebind_param(
                    layer,
                    "w2_row_offsets",
                    (
                        torch.arange(num_local + 1, dtype=torch.int32, device=device)
                        * layer.w2_weight.shape[1]
                    ),
                )
                copy_or_rebind_param(
                    layer,
                    "w2_scale_offsets",
                    (
                        torch.arange(num_local + 1, dtype=torch.int64, device=device)
                        * w2_sf_stride
                    ),
                )

                if layer.w13_weight_scale_2.dim() == 2:
                    w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]
                else:
                    w13_weight_scale_2 = layer.w13_weight_scale_2
                copy_or_rebind_param(
                    layer,
                    "w1_pertoken_wgt_scale",
                    w13_weight_scale_2.to(torch.float32),
                )
                copy_or_rebind_param(
                    layer,
                    "w2_pertoken_wgt_scale",
                    layer.w2_weight_scale_2.to(torch.float32),
                )
'''
    if anchor not in text:
        raise SystemExit("cutlass_moe_params anchor missing")
    text = text.replace(anchor, block, 1)

ep_slice_anchor = """        if self.quant_config.use_per_token_activation:
            # FlashInfer computes activation scales dynamically per token, so
            # the static checkpoint activation scale is intentionally neutral.
            w13_input_scale = torch.ones_like(w13_input_scale, dtype=torch.float32)
            w2_input_scale = torch.ones_like(w2_input_scale, dtype=torch.float32)

        # Create shared parameters. g1_alphas / g1_alphas_up are the gate (w1)
"""
ep_slice_block = """        if self.quant_config.use_per_token_activation:
            # FlashInfer computes activation scales dynamically per token, so
            # the static checkpoint activation scale is intentionally neutral.
            w13_input_scale = torch.ones_like(w13_input_scale, dtype=torch.float32)
            w2_input_scale = torch.ones_like(w2_input_scale, dtype=torch.float32)

        if layer.moe_ep_size > 1 and w13_input_scale.shape[0] == layer.num_experts:
            def _slice_scale(w):
                assert w.shape == (layer.num_experts,)
                assert layer.moe_ep_size * layer.num_local_experts == layer.num_experts
                return w[
                    layer.moe_ep_rank
                    * layer.num_local_experts : (layer.moe_ep_rank + 1)
                    * layer.num_local_experts
                ]

            w13_input_scale = _slice_scale(w13_input_scale)
            if w2_input_scale.shape == (layer.num_experts,):
                w2_input_scale = _slice_scale(w2_input_scale)

        # Create shared parameters. g1_alphas / g1_alphas_up are the gate (w1)
"""
if ep_slice_anchor in text and "if layer.moe_ep_size > 1 and w13_input_scale.shape[0] == layer.num_experts:" not in text:
    text = text.replace(ep_slice_anchor, ep_slice_block, 1)

if "cutlass_moe_fp4_pertoken_deepep_ll" not in text:
    old_apply = '''        if NVFP4_PERTOKEN_SCALE and moe_runner_backend.is_cutlass():
            from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4_pertoken

            topk_output = dispatch_output.topk_output
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
            x = dispatch_output.hidden_states
            output = cutlass_moe_fp4_pertoken(
'''
    new_apply = '''        if NVFP4_PERTOKEN_SCALE and moe_runner_backend.is_cutlass():
            from sglang.srt.layers.moe.cutlass_moe import (
                cutlass_moe_fp4_pertoken,
                cutlass_moe_fp4_pertoken_deepep_ll,
            )
            from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
            from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

            if DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
                output = cutlass_moe_fp4_pertoken_deepep_ll(
                    a=dispatch_output.hidden_states,
                    masked_m=dispatch_output.masked_m,
                    w1_fp4=layer.w13_weight,
                    w1_blockscale_flat=layer.w1_blockscale_flat,
                    w1_weight_scale_2=layer.w1_pertoken_wgt_scale,
                    w2_fp4=layer.w2_weight,
                    w2_blockscale_flat=layer.w2_blockscale_flat,
                    w2_weight_scale_2=layer.w2_pertoken_wgt_scale,
                    w1_row_offsets=layer.w1_row_offsets,
                    w1_scale_offsets=layer.w1_scale_offsets,
                    w2_row_offsets=layer.w2_row_offsets,
                    w2_scale_offsets=layer.w2_scale_offsets,
                ).to(dispatch_output.hidden_states.dtype)
                return DeepEPLLCombineInput(
                    hidden_states=output,
                    topk_ids=dispatch_output.topk_ids,
                    topk_weights=dispatch_output.topk_weights,
                )

            topk_output = dispatch_output.topk_output
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
            x = dispatch_output.hidden_states
            output = cutlass_moe_fp4_pertoken(
'''
    if old_apply in text:
        text = text.replace(old_apply, new_apply, 1)

if "cutlass_moe_fp4_pertoken" not in text:
    anchor = "        if self.enable_flashinfer_cutlass_moe:"
    block = '''        if NVFP4_PERTOKEN_SCALE and moe_runner_backend.is_cutlass():
            from sglang.srt.layers.moe.cutlass_moe import (
                cutlass_moe_fp4_pertoken,
                cutlass_moe_fp4_pertoken_deepep_ll,
            )
            from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
            from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

            if DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
                output = cutlass_moe_fp4_pertoken_deepep_ll(
                    a=dispatch_output.hidden_states,
                    masked_m=dispatch_output.masked_m,
                    w1_fp4=layer.w13_weight,
                    w1_blockscale_flat=layer.w1_blockscale_flat,
                    w1_weight_scale_2=layer.w1_pertoken_wgt_scale,
                    w2_fp4=layer.w2_weight,
                    w2_blockscale_flat=layer.w2_blockscale_flat,
                    w2_weight_scale_2=layer.w2_pertoken_wgt_scale,
                    w1_row_offsets=layer.w1_row_offsets,
                    w1_scale_offsets=layer.w1_scale_offsets,
                    w2_row_offsets=layer.w2_row_offsets,
                    w2_scale_offsets=layer.w2_scale_offsets,
                ).to(dispatch_output.hidden_states.dtype)
                return DeepEPLLCombineInput(
                    hidden_states=output,
                    topk_ids=dispatch_output.topk_ids,
                    topk_weights=dispatch_output.topk_weights,
                )

            topk_output = dispatch_output.topk_output
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
            x = dispatch_output.hidden_states
            output = cutlass_moe_fp4_pertoken(
                a=x,
                w1_fp4=layer.w13_weight,
                w1_blockscale_flat=layer.w1_blockscale_flat,
                w1_weight_scale_2=layer.w1_pertoken_wgt_scale,
                w2_fp4=layer.w2_weight,
                w2_blockscale_flat=layer.w2_blockscale_flat,
                w2_weight_scale_2=layer.w2_pertoken_wgt_scale,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                params=layer.cutlass_moe_params,
                w1_row_offsets=layer.w1_row_offsets,
                w1_scale_offsets=layer.w1_scale_offsets,
                w2_row_offsets=layer.w2_row_offsets,
                w2_scale_offsets=layer.w2_scale_offsets,
                apply_router_weight_on_input=moe_runner_config.apply_router_weight_on_input,
            ).to(x.dtype)
            return StandardCombineInput(hidden_states=output)

'''
    if anchor not in text:
        raise SystemExit("flashinfer_cutlass anchor missing")
    text = text.replace(anchor, block + anchor, 1)

old_deepep_call = """                output = cutlass_moe_fp4_pertoken_deepep_ll(
                    a=dispatch_output.hidden_states,
                    w1_fp4=layer.w13_weight,"""
new_deepep_call = """                output = cutlass_moe_fp4_pertoken_deepep_ll(
                    a=dispatch_output.hidden_states,
                    masked_m=dispatch_output.masked_m,
                    w1_fp4=layer.w13_weight,"""
if old_deepep_call in text:
    text = text.replace(old_deepep_call, new_deepep_call)

old_deepep_call_compact = """            output = cutlass_moe_fp4_pertoken_deepep_ll(
                a=dispatch_output.hidden_states,
                w1_fp4=layer.w13_weight,"""
new_deepep_call_compact = """            output = cutlass_moe_fp4_pertoken_deepep_ll(
                a=dispatch_output.hidden_states,
                masked_m=dispatch_output.masked_m,
                w1_fp4=layer.w13_weight,"""
if old_deepep_call_compact in text:
    text = text.replace(old_deepep_call_compact, new_deepep_call_compact)

path.write_text(text)
print("modelopt_quant.py patched")
PY

echo "Patching environ.py for SGLANG_NVFP4_PERTOKEN_SCALE ..."
python3 - <<'PY' "${PY}/environ.py"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
if "SGLANG_NVFP4_PERTOKEN_SCALE" in text:
    print("SGLANG_NVFP4_PERTOKEN_SCALE already present")
    raise SystemExit(0)
needle = "    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)"
insert = needle + "\n    SGLANG_NVFP4_PERTOKEN_SCALE = EnvBool(False)"
if needle not in text:
    raise SystemExit("SGLANG_MOE_NVFP4_DISPATCH anchor missing in environ.py")
path.write_text(text.replace(needle, insert, 1))
print("environ.py patched")
PY

echo "Patching http_server.py so /post_process_weights accepts a JSON body ..."
python3 - <<'PY' "${PY}/entrypoints/http_server.py"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
if "Annotated[PostProcessWeightsReqInput, Body()]" in text:
    print("post_process_weights Body() already present")
    raise SystemExit(0)
old = """async def post_process_weights(req: PostProcessWeightsReqInput, request: Request):
    \"\"\"Optional post-processing for updated weights, e.g. quantization packing.\"\"\"
    success, message = await _global_state.tokenizer_manager.post_process_weights(
        req, request
    )"""
new = """async def post_process_weights(
    obj: Annotated[PostProcessWeightsReqInput, Body()], request: Request
):
    \"\"\"Optional post-processing for updated weights, e.g. quantization packing.\"\"\"
    success, message = await _global_state.tokenizer_manager.post_process_weights(
        obj, request
    )"""
if old not in text:
    # Container SGLang 0.5.15.post1 uses this docstring; newer trees may differ.
    old_alt = """async def post_process_weights(req: PostProcessWeightsReqInput, request: Request):
    \"\"\"Optional post-processing after weight sync (e.g. w4afp8 interleave).\"\"\"
    success, message = await _global_state.tokenizer_manager.post_process_weights(
        req, request
    )"""
    if old_alt not in text:
        print("http_server.py post_process_weights anchor missing; skip")
        raise SystemExit(0)
    old = old_alt
path.write_text(text.replace(old, new, 1))
print("http_server.py post_process_weights now uses Body()")
PY

echo "Applying decode-CG-safe pertoken rebind / layout cache ..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/patch_nvfp4_cg_decode_rebind.py" "${SGLANG_ROOT}"

echo "Checking imports ..."
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
python3 -c "import fp4_gemm; from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4_pertoken; print('nvfp4 pertoken stack OK')"
