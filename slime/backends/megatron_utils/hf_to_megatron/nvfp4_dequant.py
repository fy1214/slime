"""Dequantize ModelOpt NVFP4 HuggingFace weights for Megatron BF16 load."""

from __future__ import annotations

import torch

_FP4_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
_FLOAT4_E2M1_MAX = FLOAT4_E2M1_MAX
_FLOAT8_E4M3_MAX = FLOAT8_E4M3_MAX


def modelopt_nvfp4_decode_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Convert a TE tensor amax to ModelOpt ``weight_scale_2``.

    HuggingFace ModelOpt stores the NVFP4 decode scale ``amax / (6 * 448)``.
    Transformer Engine's ``amax_rowwise`` is the tensor amax itself.
    """
    return amax.to(dtype=torch.float32).reshape(()) / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)


def _unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def _fp4_to_fp32(fp4: torch.Tensor) -> torch.Tensor:
    return _FP4_LUT.to(device=fp4.device)[fp4.to(torch.long)]


def dequantize_nvfp4_weight(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Restore a ModelOpt NVFP4 weight matrix to high precision.

    ``global_scale`` is HuggingFace ``weight_scale_2``, the decode scale
    ``amax / (6 * 448)``, not Transformer Engine's tensor amax.
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 NVFP4 payload, got {packed.dtype}")

    values = _fp4_to_fp32(_unpack_fp4(packed))
    scale = block_scale.repeat_interleave(16, dim=1).view(torch.float8_e4m3fn).to(torch.float32)
    scale = scale[: values.shape[0], : values.shape[1]]
    global_scale = global_scale.to(torch.float32).reshape(())
    restored = values * scale * global_scale
    return restored.to(dtype=dtype)
