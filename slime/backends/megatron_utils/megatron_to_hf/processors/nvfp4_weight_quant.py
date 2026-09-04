"""Shared NVFP4 weight quantization for Megatron->HF sync and MoE alignment."""

from __future__ import annotations

import torch

from slime.backends.megatron_utils.hf_to_megatron.nvfp4_dequant import (
    modelopt_nvfp4_decode_scale_from_amax,
)


def quantize_matrix_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one BF16 weight matrix to ModelOpt NVFP4 checkpoint layout."""
    from transformer_engine.pytorch import NVFP4Quantizer

    if weight.dtype != torch.bfloat16:
        weight = weight.to(torch.bfloat16)

    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
    )
    qweight = quantizer(weight)
    if hasattr(qweight, "_rowwise_data") and qweight._rowwise_data is not None:
        packed = qweight._rowwise_data.contiguous()
        if qweight._rowwise_scale_inv is None:
            raise RuntimeError("NVFP4Quantizer returned rowwise data without scales")
        blockscale = qweight._rowwise_scale_inv.view(torch.float8_e4m3fn).contiguous()
        metadata = qweight.get_metadata()
        amax = metadata.get("amax_rowwise")
        if amax is None:
            amax = torch.ones((), device=weight.device, dtype=torch.float32)
        global_scale = modelopt_nvfp4_decode_scale_from_amax(amax).contiguous()
    elif hasattr(qweight, "data") and hasattr(qweight, "scale"):
        packed = qweight.data.contiguous()
        blockscale = qweight.scale.contiguous()
        if blockscale.dtype != torch.float8_e4m3fn:
            blockscale = blockscale.to(torch.float8_e4m3fn)

        if hasattr(qweight, "scale_inv") and qweight.scale_inv is not None:
            amax = qweight.scale_inv.float().reshape(())
        elif hasattr(qweight, "_scale_inv") and qweight._scale_inv is not None:
            amax = qweight._scale_inv.float().reshape(())
        else:
            amax = torch.ones((), device=weight.device, dtype=torch.float32)
        global_scale = modelopt_nvfp4_decode_scale_from_amax(amax).contiguous()
    else:
        raise RuntimeError("NVFP4Quantizer returned an unexpected tensor type")

    if packed.dtype != torch.uint8:
        packed = packed.view(torch.uint8)
    return packed, blockscale, global_scale


def _named_nvfp4_tensors(
    name: str,
    packed: torch.Tensor,
    blockscale: torch.Tensor,
    global_scale: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    if not name.endswith(".weight"):
        raise ValueError(f"Expected a weight parameter name, got {name}")
    stem = name[: -len(".weight")]
    # Clone scale_2/input_scale so gate and up do not share storage.
    global_scale = global_scale.reshape(()).to(torch.float32).contiguous()
    return [
        (name, packed),
        (f"{stem}.weight_scale", blockscale),
        (f"{stem}.weight_scale_2", global_scale.clone()),
        (f"{stem}.input_scale", global_scale.clone()),
    ]


def quantize_named_weight_nvfp4(name: str, weight: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Return HF ModelOpt NVFP4 tensors for one ``*.weight`` parameter."""
    packed, blockscale, global_scale = quantize_matrix_nvfp4(weight)
    return _named_nvfp4_tensors(name, packed, blockscale, global_scale)


def quantize_named_gated_pair_nvfp4(
    gate_name: str,
    gate_weight: torch.Tensor,
    up_name: str,
    up_weight: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    """Quantize fused ``w13 = cat(gate, up)`` with one amax, then split back.

    ModelOpt and Megatron cutlass wrap both treat SwiGLU ``linear_fc1`` as one
    matrix. Quantizing gate/up independently gives ``up`` a smaller
    ``weight_scale_2`` and different blockscales, which is what broke
    train/rollout alignment on Qwen3-30B-A3B-NVFP4.
    """
    if gate_weight.ndim != 2 or up_weight.ndim != 2:
        raise ValueError(
            f"Gated NVFP4 pair must be 2D, got gate={tuple(gate_weight.shape)} up={tuple(up_weight.shape)}"
        )
    if gate_weight.shape[1] != up_weight.shape[1]:
        raise ValueError(
            f"Gated NVFP4 pair K mismatch: gate={tuple(gate_weight.shape)} up={tuple(up_weight.shape)}"
        )

    fused = torch.cat([gate_weight, up_weight], dim=0)
    packed, blockscale, global_scale = quantize_matrix_nvfp4(fused)
    gate_rows = gate_weight.shape[0]
    if packed.shape[0] != fused.shape[0] or blockscale.shape[0] != fused.shape[0]:
        raise RuntimeError(
            "NVFP4 gated-pair quantize returned unexpected leading dim: "
            f"fused={tuple(fused.shape)} packed={tuple(packed.shape)} scale={tuple(blockscale.shape)}"
        )
    return [
        *_named_nvfp4_tensors(gate_name, packed[:gate_rows], blockscale[:gate_rows], global_scale),
        *_named_nvfp4_tensors(up_name, packed[gate_rows:], blockscale[gate_rows:], global_scale),
    ]
