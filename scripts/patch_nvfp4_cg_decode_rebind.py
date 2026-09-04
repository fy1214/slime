#!/usr/bin/env python3
"""Make NVFP4 pertoken MoE decode-CUDA-graph safe after weight sync.

Root cause (colocate weight sync + decode CUDA graph):
  1. SGLang captures decode CG after HF load (addresses baked into graph).
  2. Megatron→SGLang sync runs process_weights_after_loading again.
  3. Plain ``layer.w1_blockscale_flat = ...`` rebinds to a NEW tensor.
  4. CG replay still reads the OLD address → garbage decode logprobs.
  Prefill (eager) uses the new address → decode/prefill logprob mismatch.

Fix (from b300-minite-patched):
  - create_weights: preallocate pertoken derived Parameters
  - process_weights: copy_or_rebind_param (in-place when shape matches)
  - get_prob_host: grow cached max_m (compact prefill must not undersize LL)
  - deepep_ll: cache static layout tensors; keep masked_m compare dynamic
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path


def patch_modelopt(path: Path) -> None:
    text = path.read_text()

    create_anchor = """        w2_input_scale._sglang_require_global_experts = True
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        \"\"\"Process FP4 MoE weights after loading from serialized checkpoint.
"""

    create_insert = """        w2_input_scale._sglang_require_global_experts = True
        layer.register_parameter("w2_input_scale", w2_input_scale)

        # Preallocate pertoken derived buffers so decode CUDA graph capture
        # keeps stable Parameter addresses across post_process_weights /
        # colocate weight sync (plain assign would rebind and leave the
        # captured graph reading stale blockscales).
        if NVFP4_PERTOKEN_SCALE:
            device = w13_weight.device
            E = layer.num_local_experts
            w1_sf_stride = w13_weight_scale.shape[1] * w13_weight_scale.shape[2]
            w2_sf_stride = w2_weight_scale.shape[1] * w2_weight_scale.shape[2]
            layer.w1_blockscale_flat = Parameter(
                torch.zeros(E * w1_sf_stride, dtype=torch.float8_e4m3fn, device=device),
                requires_grad=False,
            )
            layer.w1_row_offsets = Parameter(
                torch.zeros(E + 1, dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w1_scale_offsets = Parameter(
                torch.zeros(E + 1, dtype=torch.int64, device=device),
                requires_grad=False,
            )
            layer.w2_blockscale_flat = Parameter(
                torch.zeros(E * w2_sf_stride, dtype=torch.float8_e4m3fn, device=device),
                requires_grad=False,
            )
            layer.w2_row_offsets = Parameter(
                torch.zeros(E + 1, dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_scale_offsets = Parameter(
                torch.zeros(E + 1, dtype=torch.int64, device=device),
                requires_grad=False,
            )
            layer.w1_pertoken_wgt_scale = Parameter(
                torch.zeros(E, dtype=torch.float32, device=device),
                requires_grad=False,
            )
            layer.w2_pertoken_wgt_scale = Parameter(
                torch.zeros(E, dtype=torch.float32, device=device),
                requires_grad=False,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        \"\"\"Process FP4 MoE weights after loading from serialized checkpoint.
"""

    if 'layer.w1_blockscale_flat = Parameter(' not in text:
        if create_anchor not in text:
            raise SystemExit(f"create_weights anchor missing in {path}")
        text = text.replace(create_anchor, create_insert, 1)
        print(f"[ok] create_weights prealloc: {path}")
    else:
        print(f"[skip] create_weights prealloc already present: {path}")

    old_pp = """            if NVFP4_PERTOKEN_SCALE:
                w13_bs = layer.w13_blockscale_swizzled
                w2_bs = layer.w2_blockscale_swizzled
                num_local = w13_bs.shape[0]

                w1_sf_stride = w13_bs.shape[1] * w13_bs.shape[2]
                layer.w1_blockscale_flat = w13_bs.reshape(-1).contiguous()
                layer.w1_row_offsets = (
                    torch.arange(num_local + 1, dtype=torch.int32, device=device)
                    * layer.w13_weight.shape[1]
                )
                layer.w1_scale_offsets = (
                    torch.arange(num_local + 1, dtype=torch.int64, device=device) * w1_sf_stride
                )

                w2_sf_stride = w2_bs.shape[1] * w2_bs.shape[2]
                layer.w2_blockscale_flat = w2_bs.reshape(-1).contiguous()
                layer.w2_row_offsets = (
                    torch.arange(num_local + 1, dtype=torch.int32, device=device)
                    * layer.w2_weight.shape[1]
                )
                layer.w2_scale_offsets = (
                    torch.arange(num_local + 1, dtype=torch.int64, device=device) * w2_sf_stride
                )

                if layer.w13_weight_scale_2.dim() == 2:
                    w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]
                else:
                    w13_weight_scale_2 = layer.w13_weight_scale_2
                layer.w1_pertoken_wgt_scale = w13_weight_scale_2.to(torch.float32)
                layer.w2_pertoken_wgt_scale = layer.w2_weight_scale_2.to(torch.float32)
"""

    new_pp = """            if NVFP4_PERTOKEN_SCALE:
                w13_bs = layer.w13_blockscale_swizzled
                w2_bs = layer.w2_blockscale_swizzled
                num_local = w13_bs.shape[0]

                w1_sf_stride = w13_bs.shape[1] * w13_bs.shape[2]
                # In-place copy/rebind keeps CUDA-graph-captured addresses stable
                # across weight sync + process_weights_after_loading.
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
"""

    if 'copy_or_rebind_param(\n                    layer,\n                    "w1_blockscale_flat"' not in text:
        if old_pp not in text:
            raise SystemExit(f"process_weights pertoken block missing in {path}")
        text = text.replace(old_pp, new_pp, 1)
        print(f"[ok] process_weights copy_or_rebind: {path}")
    else:
        print(f"[skip] process_weights already copy_or_rebind: {path}")

    path.write_text(text)
    py_compile.compile(str(path), doraise=True)


def patch_fp4_utils(path: Path) -> None:
    text = path.read_text()
    old = '''def get_prob_host(num_groups: int, max_m: int, n: int, k: int) -> torch.Tensor:
    """Return a cached CPU prob_host tensor for CUTLASS initialize().

    Only the first call to grouped_cutlass_gemm_v2 per (num_groups, device)
    actually reads prob_host (for workspace planning); subsequent calls use
    the cached CUTLASS Gemm object.  We use conservative max_m so the
    workspace is always large enough.
    """
    key = (num_groups, n, k)
    if key not in _prob_host_cache:
        _prob_host_cache[key] = torch.tensor(
            [[max_m, n, k]] * num_groups, dtype=torch.int32
        ).reshape(-1)
    return _prob_host_cache[key]
'''
    new = '''def get_prob_host(num_groups: int, max_m: int, n: int, k: int) -> torch.Tensor:
    """Return a cached CPU prob_host tensor for CUTLASS initialize().

    Only the first call to grouped_cutlass_gemm_v2 per (num_groups, device)
    actually reads prob_host (for workspace planning); subsequent calls use
    the cached CUTLASS Gemm object.  We keep the *largest* max_m seen for a
    given (num_groups, n, k) so a compact-prefill init cannot undersize the
    later DeepEP-LL decode path (M=num_max_dispatch*world).
    """
    key = (num_groups, n, k)
    cached = _prob_host_cache.get(key)
    if cached is None or int(cached[0].item()) < int(max_m):
        _prob_host_cache[key] = torch.tensor(
            [[max_m, n, k]] * num_groups, dtype=torch.int32
        ).reshape(-1)
    return _prob_host_cache[key]
'''
    if "int(cached[0].item()) < int(max_m)" in text:
        print(f"[skip] get_prob_host already grows max_m: {path}")
        return
    if old not in text:
        raise SystemExit(f"get_prob_host anchor missing in {path}")
    path.write_text(text.replace(old, new, 1))
    py_compile.compile(str(path), doraise=True)
    print(f"[ok] get_prob_host grow max_m: {path}")


def patch_cutlass_moe(path: Path) -> None:
    text = path.read_text()
    if "_deepep_ll_layout_cache" in text:
        print(f"[skip] deepep_ll layout cache already present: {path}")
        return

    marker = "def cutlass_moe_fp4_pertoken_deepep_ll("
    helper = '''# Static DeepEP-LL layout tensors (offsets / arange) keyed by shape.
# Values are constant for a given (E, max_m); caching keeps CUDA-graph
# parameter addresses stable across replays even if the graph pool is
# not used for these host-driven allocations.
_deepep_ll_layout_cache = {}


def _get_deepep_ll_layout(num_experts: int, max_m: int, device, dtype):
    m_padded = ((max_m + 127) // 128) * 128
    key = (num_experts, max_m, m_padded, str(device), str(dtype))
    cached = _deepep_ll_layout_cache.get(key)
    if cached is not None:
        return cached
    positions = torch.arange(max_m, device=device, dtype=dtype).unsqueeze(0)
    expert_offsets = (
        torch.arange(num_experts + 1, device=device, dtype=torch.int32) * max_m
    )
    padded_offsets = (
        torch.arange(num_experts + 1, device=device, dtype=torch.int32) * m_padded
    )
    total_padded = num_experts * m_padded
    row_indices_padded = torch.arange(
        total_padded, device=device, dtype=padded_offsets.dtype
    )
    expert_ids = torch.searchsorted(
        padded_offsets[1:], row_indices_padded, right=True
    ).clamp(max=num_experts - 1)
    cached = {
        "m_padded": m_padded,
        "total_padded": total_padded,
        "positions": positions,
        "expert_offsets": expert_offsets,
        "padded_offsets": padded_offsets,
        "expert_ids": expert_ids,
    }
    _deepep_ll_layout_cache[key] = cached
    return cached


'''
    if marker not in text:
        raise SystemExit(f"deepep_ll marker missing in {path}")
    text = text.replace(marker, helper + marker, 1)

    old_body = '''    capturing = torch.cuda.is_current_stream_capturing()
    positions = torch.arange(max_m, device=device, dtype=counts.dtype).unsqueeze(0)
    valid_mask = positions < counts.unsqueeze(1)
    x = a.reshape(num_experts * max_m, hidden) * valid_mask.reshape(-1, 1).to(
        dtype=a.dtype
    )

    # Host-side sizes from the static 3D layout; never .item() on masked_m.
    m_padded = ((max_m + 127) // 128) * 128
    total_padded = num_experts * m_padded
    expert_offsets = (
        torch.arange(num_experts + 1, device=device, dtype=torch.int32) * max_m
    )
    padded_offsets = (
        torch.arange(num_experts + 1, device=device, dtype=torch.int32) * m_padded
    )
    unpadded_offsets = padded_offsets if m_padded == max_m else expert_offsets

    n2_w1 = w1_fp4.shape[1]
    n_w2 = w2_fp4.shape[1]

    act_scale_offsets_1 = compute_act_scale_offsets(padded_offsets, hidden)
    row_indices_padded = torch.arange(
        total_padded, device=device, dtype=padded_offsets.dtype
    )
    expert_ids = torch.searchsorted(
        padded_offsets[1:], row_indices_padded, right=True
    ).clamp(max=num_experts - 1)
'''
    new_body = '''    capturing = torch.cuda.is_current_stream_capturing()
    layout = _get_deepep_ll_layout(num_experts, max_m, device, counts.dtype)
    positions = layout["positions"]
    m_padded = layout["m_padded"]
    total_padded = layout["total_padded"]
    expert_offsets = layout["expert_offsets"]
    padded_offsets = layout["padded_offsets"]
    expert_ids = layout["expert_ids"]
    # masked_m is dynamic: keep the comparison in-graph so replay sees
    # the current DeepEP packed_recv_count buffer contents.
    valid_mask = positions < counts.unsqueeze(1)
    x = a.reshape(num_experts * max_m, hidden) * valid_mask.reshape(-1, 1).to(
        dtype=a.dtype
    )

    # Host-side sizes from the static 3D layout; never .item() on masked_m.
    unpadded_offsets = padded_offsets if m_padded == max_m else expert_offsets

    n2_w1 = w1_fp4.shape[1]
    n_w2 = w2_fp4.shape[1]

    act_scale_offsets_1 = compute_act_scale_offsets(padded_offsets, hidden)
'''
    if old_body not in text:
        raise SystemExit(f"deepep_ll body anchor missing in {path}")
    text = text.replace(old_body, new_body, 1)
    path.write_text(text)
    py_compile.compile(str(path), doraise=True)
    print(f"[ok] deepep_ll layout cache: {path}")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
    py = root / "python" / "sglang" / "srt"
    patch_modelopt(py / "layers" / "quantization" / "modelopt_quant.py")
    patch_fp4_utils(py / "layers" / "quantization" / "fp4_utils.py")
    patch_cutlass_moe(py / "layers" / "moe" / "cutlass_moe.py")
    print("all patches applied")


if __name__ == "__main__":
    main()
