"""Public entry hooks for Qwen3 NVFP4 cutlass MoE train/rollout alignment.

Kept separate from ``deepgemm_forward`` / ``deepgemm_moe_forward`` so the NVFP4
path does not share those modules' CLI surface or enable APIs.
"""

from __future__ import annotations

import os

from slime.backends.megatron_utils.alignment.cutlass_nvfp4_moe_forward import (
    enable_cutlass_nvfp4_moe_forward,
)
from slime.backends.megatron_utils.alignment.deepgemm_forward import (
    enable_sglang_absorbed_kv_rmsnorm,
    enable_sglang_final_rmsnorm,
    enable_sglang_global_batch_invariant_ops,
    enable_sglang_layer0_input_rmsnorm,
    enable_sglang_router_gemm,
)


def enable_nvfp4_alignment_all_forward(args, model, store_prefix: str) -> None:
    """Install Qwen3 NVFP4 cutlass MoE alignment hooks (no DeepGEMM)."""
    enable_sglang_global_batch_invariant_ops()
    enable_sglang_layer0_input_rmsnorm(args, model, store_prefix)
    enable_sglang_absorbed_kv_rmsnorm(args, model, store_prefix)
    enable_sglang_final_rmsnorm(args, model, store_prefix)

    layers = getattr(args, "megatron_cutlass_nvfp4_moe_forward_layers", None)
    if os.environ.get("MEGATRON_USE_SGLANG_ROUTER_GEMM", "0") == "1":
        enable_sglang_router_gemm(
            args,
            model,
            store_prefix,
            selected_layers=layers,
        )

    enable_cutlass_nvfp4_moe_forward(args, model, store_prefix)


def enable_nvfp4_alignment_all_forward_before_train_step(
    args,
    rollout_id: int,
    step_id: int,
    model,
    optimizer,
    opt_param_scheduler,
) -> None:
    del rollout_id, step_id, optimizer, opt_param_scheduler
    enable_nvfp4_alignment_all_forward(args, model, store_prefix="")
