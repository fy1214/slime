"""Align Megatron TEGroupedMLP with SGLang ModelOpt NVFP4 cutlass pertoken path.

After DeepEP dispatch, Megatron hands TEGroupedMLP expert-major hidden states,
``tokens_per_expert``, and permuted router probabilities.  This module replaces
the grouped BF16/FP8 expert compute with the same kernel stages SGLang uses in
``cutlass_moe_fp4_pertoken`` (``nvfp4_quantize_pertoken`` + ``nvfp4_grouped_gemm``
+ ``silu_and_mul``), while keeping DeepEP combine semantics via
``enable_sglang_deepep_moe_alignment``.

Training reuses the shared BF16 expert backward in ``moe_bf16_expert_backward``;
only the forward stays NVFP4 cutlass.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from slime.backends.megatron_utils.alignment.deepgemm_forward import (
    _format_int_ranges,
    _should_log_deepgemm_summary,
)
from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import (
    _COMBINE_WORKSPACE_ATTR,
    _MoELayout,
    _PREALLOCATED_COMBINE_BUFFER_ATTR,
    _apply_router_probability_fp32_inplace,
    _combine_workspace_bytes,
    _compact_valid_rows_inplace,
    _get_expert_weights,
    _get_global_layer_index,
    _normalize_model_chunks,
    _validate_parallelism,
    _validate_routing_inputs,
    _validate_te_grouped_mlp,
    _wrap_preallocated_combine_preprocess,
    _wrap_preallocated_dispatch_postprocess,
    _wrap_preallocated_token_combine,
    enable_sglang_deepep_moe_alignment,
)
from slime.backends.megatron_utils.alignment.moe_bf16_expert_backward import (
    moe_bf16_expert_backward,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_SUFFIXES = ("mlp.experts",)


class _CutlassNvfp4MoEWithBF16Backward(torch.autograd.Function):
    """NVFP4 cutlass MoE forward with DeepGEMM-aligned BF16 expert backward."""

    @staticmethod
    def forward(
        ctx,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        module: torch.nn.Module,
        layout: _MoELayout,
        module_name: str,
        *weights: torch.Tensor,
    ) -> torch.Tensor:
        if len(weights) != 2 * layout.num_local_experts:
            raise RuntimeError(
                f"{module_name} expected {2 * layout.num_local_experts} expert weights, got {len(weights)}"
            )
        counts = _validate_routing_inputs(
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
            layout,
        )
        state = _build_nvfp4_moe_state(
            module,
            layout,
            module_name=module_name,
        )
        output = _expert_major_cutlass_nvfp4_moe_forward(
            module,
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
            state=state,
            layout=layout,
        )
        ctx.layout = layout
        ctx.counts = counts
        ctx.module_name = module_name
        ctx.defer_router_probabilities = bool(getattr(module, "_slime_defer_router_probabilities", False))
        ctx.reuse_expert_input_for_grad = bool(getattr(module, "_slime_reuse_expert_input_for_grad", False))
        ctx.grad_workspace = getattr(module, _COMBINE_WORKSPACE_ATTR, None)
        ctx.save_for_backward(permuted_local_hidden_states, permuted_probs, *weights)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        hidden_states = saved[0]
        permuted_probs = saved[1]
        weights = saved[2:]
        layout: _MoELayout = ctx.layout
        num_experts = layout.num_local_experts
        needs = ctx.needs_input_grad
        grad_hidden, grad_probs, grad_fc1_weights, grad_fc2_weights = moe_bf16_expert_backward(
            hidden_states=hidden_states,
            permuted_probs=permuted_probs,
            grad_output=grad_output,
            fc1_weights=weights[:num_experts],
            fc2_weights=weights[num_experts:],
            counts=ctx.counts,
            layout=layout,
            module_name=ctx.module_name,
            needs_hidden=needs[0],
            needs_probs=needs[2],
            needs_fc1_weights=needs[6 : 6 + num_experts],
            needs_fc2_weights=needs[6 + num_experts :],
            defer_router_probabilities=ctx.defer_router_probabilities,
            reuse_expert_input_for_grad=ctx.reuse_expert_input_for_grad,
            grad_workspace=ctx.grad_workspace,
        )
        return (
            grad_hidden,
            None,
            grad_probs,
            None,
            None,
            None,
            *grad_fc1_weights,
            *grad_fc2_weights,
        )


@dataclass(frozen=True)
class _NvFp4MoEState:
    layout: _MoELayout
    w1_fp4: torch.Tensor
    w1_blockscale_flat: torch.Tensor
    w1_weight_scale_2: torch.Tensor
    w1_row_offsets: torch.Tensor
    w1_scale_offsets: torch.Tensor
    w2_fp4: torch.Tensor
    w2_blockscale_flat: torch.Tensor
    w2_weight_scale_2: torch.Tensor
    w2_row_offsets: torch.Tensor
    w2_scale_offsets: torch.Tensor
    cutlass_moe_params: CutlassMoEParams


from slime.backends.megatron_utils.megatron_to_hf.processors.nvfp4_weight_quant import (
    quantize_matrix_nvfp4,
)


def _stack_expert_nvfp4(
    expert_weights: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_rows = []
    blockscale_rows = []
    global_scales = []
    for weight in expert_weights:
        packed, blockscale, global_scale = quantize_matrix_nvfp4(weight)
        packed_rows.append(packed)
        blockscale_rows.append(blockscale)
        global_scales.append(global_scale.reshape(()).expand(1))
    w_fp4 = torch.stack(packed_rows, dim=0)
    w_blockscale = torch.stack(blockscale_rows, dim=0)
    w_scale_2 = torch.cat(global_scales, dim=0).to(torch.float32)
    return w_fp4, w_blockscale, w_scale_2


def _build_nvfp4_moe_state(
    experts: torch.nn.Module,
    layout: _MoELayout,
    *,
    module_name: str,
) -> _NvFp4MoEState:
    from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    fc1_weights = _get_expert_weights(
        experts.linear_fc1,
        num_local_experts=layout.num_local_experts,
        expected_shape=layout.fc1_weight_shape,
        module_name=f"{module_name}.linear_fc1",
    )
    fc2_weights = _get_expert_weights(
        experts.linear_fc2,
        num_local_experts=layout.num_local_experts,
        expected_shape=layout.fc2_weight_shape,
        module_name=f"{module_name}.linear_fc2",
    )

    w1_fp4, w1_blockscale, w1_scale_2 = _stack_expert_nvfp4(fc1_weights)
    w2_fp4, w2_blockscale, w2_scale_2 = _stack_expert_nvfp4(fc2_weights)

    w1_blockscale_swizzled = swizzle_blockscale(w1_blockscale)
    w2_blockscale_swizzled = swizzle_blockscale(w2_blockscale)
    device = w1_fp4.device
    num_experts = layout.num_local_experts
    w1_sf_stride = w1_blockscale_swizzled.shape[1] * w1_blockscale_swizzled.shape[2]
    w2_sf_stride = w2_blockscale_swizzled.shape[1] * w2_blockscale_swizzled.shape[2]

    cutlass_moe_params = CutlassMoEParams(
        CutlassMoEType.BlockscaledFP4,
        device,
        num_experts=num_experts,
        intermediate_size_per_partition=layout.ffn_hidden_size,
        hidden_size=layout.hidden_size,
    )

    return _NvFp4MoEState(
        layout=layout,
        w1_fp4=w1_fp4,
        w1_blockscale_flat=w1_blockscale_swizzled.reshape(-1).contiguous(),
        w1_weight_scale_2=w1_scale_2,
        w1_row_offsets=(torch.arange(num_experts + 1, dtype=torch.int32, device=device) * w1_fp4.shape[1]),
        w1_scale_offsets=(torch.arange(num_experts + 1, dtype=torch.int64, device=device) * w1_sf_stride),
        w2_fp4=w2_fp4,
        w2_blockscale_flat=w2_blockscale_swizzled.reshape(-1).contiguous(),
        w2_weight_scale_2=w2_scale_2,
        w2_row_offsets=(torch.arange(num_experts + 1, dtype=torch.int32, device=device) * w2_fp4.shape[1]),
        w2_scale_offsets=(torch.arange(num_experts + 1, dtype=torch.int64, device=device) * w2_sf_stride),
        cutlass_moe_params=cutlass_moe_params,
    )


def _build_expert_offsets(tokens_per_expert: torch.Tensor) -> torch.Tensor:
    counts = tokens_per_expert.to(torch.int32)
    return F.pad(torch.cumsum(counts, dim=0), (1, 0), value=0)


def _expert_major_cutlass_nvfp4_moe_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    permuted_probs: torch.Tensor,
    *,
    state: _NvFp4MoEState,
    layout: _MoELayout,
) -> torch.Tensor:
    """Run the pertoken NVFP4 grouped GEMM path on DeepEP expert-major activations."""
    from sglang.srt.layers.quantization.fp4_utils import (
        compute_act_scale_offsets,
        compute_padded_expert_offsets,
        get_prob_host,
        nvfp4_grouped_gemm,
        nvfp4_quantize_pertoken,
    )
    from sgl_kernel import silu_and_mul

    counts = _validate_routing_inputs(
        hidden_states,
        tokens_per_expert,
        permuted_probs,
        layout,
    )
    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        return hidden_states.new_empty((0, layout.hidden_size))

    device = hidden_states.device
    out_dtype = hidden_states.dtype
    num_experts = layout.num_local_experts
    hidden_size = layout.hidden_size
    n2_w1 = 2 * layout.ffn_hidden_size
    n_w2 = layout.hidden_size
    inter_k = layout.ffn_hidden_size

    expert_offsets = _build_expert_offsets(tokens_per_expert).to(device=device, dtype=torch.int32)
    padded_offsets = compute_padded_expert_offsets(expert_offsets)
    total_padded = int(padded_offsets[-1].item())
    # Build valid-row map without materializing a full padded activation buffer.
    # pad_fuse reads unpadded expert-major ``hidden_states`` directly.
    counts_i = tokens_per_expert.to(device=device, dtype=torch.int32)
    max_count = int(counts_i.max().item()) if num_tokens else 0
    if max_count:
        pos = torch.arange(max_count, device=device, dtype=counts_i.dtype).unsqueeze(0)
        valid_mask = pos < counts_i.unsqueeze(1)
        valid_rows = (padded_offsets[:-1].unsqueeze(1) + pos)[valid_mask].to(torch.long)
    else:
        valid_rows = None

    act_scale_offsets_1 = compute_act_scale_offsets(padded_offsets, hidden_size)
    row_indices = torch.arange(total_padded, device=device, dtype=padded_offsets.dtype)
    expert_ids = torch.searchsorted(padded_offsets[1:], row_indices, right=True).clamp(max=num_experts - 1)

    data1, sf1, gs1 = nvfp4_quantize_pertoken(
        hidden_states,
        padded_offsets,
        expert_offsets,
        act_scale_offsets_1,
        total_padded,
        hidden_size,
    )
    pts1 = (gs1 * state.w1_weight_scale_2[expert_ids]).contiguous()

    max_m = ((total_padded + 127) // 128) * 128
    prob_host_1 = get_prob_host(num_experts, max_m, n2_w1, hidden_size)
    c1 = torch.empty(total_padded, n2_w1, dtype=out_dtype, device=device)
    nvfp4_grouped_gemm(
        c1,
        data1,
        state.w1_fp4.reshape(-1, state.w1_fp4.shape[-1]),
        sf1,
        state.w1_blockscale_flat,
        padded_offsets,
        state.w1_row_offsets,
        act_scale_offsets_1,
        state.w1_scale_offsets,
        num_experts,
        pts1,
        prob_host_1,
    )

    intermediate = torch.empty((total_padded, inter_k), device=device, dtype=out_dtype)
    silu_and_mul(c1, intermediate)
    del c1

    act_scale_offsets_2 = compute_act_scale_offsets(padded_offsets, inter_k)
    data2, sf2, gs2 = nvfp4_quantize_pertoken(
        intermediate,
        padded_offsets,
        padded_offsets,
        act_scale_offsets_2,
        total_padded,
        inter_k,
    )
    del intermediate

    pts2 = (gs2 * state.w2_weight_scale_2[expert_ids]).contiguous()
    prob_host_2 = get_prob_host(num_experts, max_m, n_w2, inter_k)
    c2 = torch.empty(total_padded, n_w2, dtype=out_dtype, device=device)
    nvfp4_grouped_gemm(
        c2,
        data2,
        state.w2_fp4.reshape(-1, state.w2_fp4.shape[-1]),
        sf2,
        state.w2_blockscale_flat,
        padded_offsets,
        state.w2_row_offsets,
        act_scale_offsets_2,
        state.w2_scale_offsets,
        num_experts,
        pts2,
        prob_host_2,
    )

    if valid_rows is not None:
        output = _compact_valid_rows_inplace(c2, valid_rows)
    else:
        output = c2[:num_tokens]

    defer_router_probabilities = bool(getattr(module, "_slime_defer_router_probabilities", False))
    if not defer_router_probabilities:
        output = _apply_router_probability_fp32_inplace(output, permuted_probs)
    return output


def _wrap_te_grouped_mlp_cutlass_nvfp4(
    module: torch.nn.Module,
    module_name: str,
) -> bool:
    if getattr(module, "_slime_cutlass_nvfp4_moe_forward_wrapped", False):
        return False

    _validate_parallelism()
    layout = _validate_te_grouped_mlp(module, module_name)
    fc1_weights = _get_expert_weights(
        module.linear_fc1,
        num_local_experts=layout.num_local_experts,
        expected_shape=layout.fc1_weight_shape,
        module_name=f"{module_name}.linear_fc1",
    )
    fc2_weights = _get_expert_weights(
        module.linear_fc2,
        num_local_experts=layout.num_local_experts,
        expected_shape=layout.fc2_weight_shape,
        module_name=f"{module_name}.linear_fc2",
    )
    if getattr(getattr(module, "config", None), "delay_wgrad_compute", False):
        raise RuntimeError(
            "Cutlass NVFP4 MoE custom backward does not support Megatron delay_wgrad_compute; "
            "disable delay_wgrad_compute."
        )

    def cutlass_nvfp4_moe_forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        combine_buffer = getattr(
            permuted_local_hidden_states,
            _PREALLOCATED_COMBINE_BUFFER_ATTR,
            None,
        )
        if torch.is_grad_enabled():
            output = _CutlassNvfp4MoEWithBF16Backward.apply(
                permuted_local_hidden_states,
                tokens_per_expert,
                permuted_probs,
                self,
                layout,
                module_name,
                *fc1_weights,
                *fc2_weights,
            )
        else:
            state = _build_nvfp4_moe_state(
                self,
                layout,
                module_name=module_name,
            )
            output = _expert_major_cutlass_nvfp4_moe_forward(
                self,
                permuted_local_hidden_states,
                tokens_per_expert,
                permuted_probs,
                state=state,
                layout=layout,
            )
        if combine_buffer is not None:
            setattr(output, _PREALLOCATED_COMBINE_BUFFER_ATTR, combine_buffer)
        return output, getattr(self, "output_bias", None)

    module.forward = types.MethodType(cutlass_nvfp4_moe_forward, module)
    module._slime_cutlass_nvfp4_moe_forward_wrapped = True
    module._slime_cutlass_nvfp4_moe_module_name = module_name
    module._slime_cutlass_nvfp4_moe_layout = layout
    return True


def install_cutlass_nvfp4_moe_forward(
    model,
    global_layer_indices: Iterable[int],
    *,
    target_suffixes: Iterable[str] = _DEFAULT_TARGET_SUFFIXES,
) -> list[str]:
    """Wrap selected global TEGroupedMLPs with the NVFP4 cutlass pertoken path."""
    _validate_parallelism()
    selected_layers = {int(layer_index) for layer_index in global_layer_indices}
    if not selected_layers:
        raise RuntimeError("global_layer_indices must select at least one MoE layer")

    suffixes = tuple(target_suffixes)
    if not suffixes:
        raise RuntimeError("target_suffixes must select at least one module name")

    workspace_bytes = _combine_workspace_bytes()
    wrapped: list[str] = []
    workspaces_by_device: dict[torch.device, torch.Tensor] = {}
    for model_chunk in _normalize_model_chunks(model):
        for name, module in model_chunk.named_modules():
            if not any(name.endswith(suffix) for suffix in suffixes):
                continue
            if _get_global_layer_index(model_chunk, name) not in selected_layers:
                continue
            if _wrap_te_grouped_mlp_cutlass_nvfp4(module, name):
                mlp_name = name.rsplit(".", 1)[0]
                mlp = model_chunk.get_submodule(mlp_name)
                dispatcher = getattr(mlp, "token_dispatcher", None)
                if (
                    workspace_bytes is not None
                    and dispatcher is not None
                    and int(getattr(dispatcher, "num_local_experts", 1)) > 1
                ):
                    _wrap_preallocated_combine_preprocess(dispatcher)
                    parameter = next(module.parameters())
                    workspace = workspaces_by_device.get(parameter.device)
                    if workspace is None:
                        workspace = torch.empty(
                            workspace_bytes,
                            dtype=torch.uint8,
                            device=parameter.device,
                        )
                        workspaces_by_device[parameter.device] = workspace
                    setattr(dispatcher, _COMBINE_WORKSPACE_ATTR, workspace)
                    setattr(module, _COMBINE_WORKSPACE_ATTR, workspace)
                    _wrap_preallocated_dispatch_postprocess(dispatcher)
                    _wrap_preallocated_token_combine(dispatcher)
                wrapped.append(name)

    if wrapped and _should_log_deepgemm_summary():
        logger.info(
            "Enabled SGLang cutlass NVFP4 pertoken MoE forward+BF16-backward on %d "
            "TEGroupedMLPs (global layers=%s)",
            len(wrapped),
            _format_int_ranges(selected_layers),
        )
        logger.debug("Cutlass NVFP4 wrapped TEGroupedMLPs: %s", ", ".join(wrapped))
    else:
        logger.debug(
            "No TEGroupedMLP matched the requested cutlass NVFP4 MoE layers %s and suffixes %s",
            sorted(selected_layers),
            suffixes,
        )
    return wrapped


def enable_cutlass_nvfp4_moe_forward(args, model, store_prefix: str) -> None:
    """Install NVFP4 cutlass pertoken forward + DeepEP combine on selected MoE layers."""
    del store_prefix
    layers = getattr(args, "megatron_cutlass_nvfp4_moe_forward_layers", None)
    if not layers:
        raise RuntimeError(
            "args.megatron_cutlass_nvfp4_moe_forward_layers is required; "
            "pass --megatron-cutlass-nvfp4-moe-forward-layers"
        )
    suffixes = getattr(args, "megatron_cutlass_nvfp4_moe_forward_modules", None) or _DEFAULT_TARGET_SUFFIXES
    install_cutlass_nvfp4_moe_forward(model, layers, target_suffixes=suffixes)
    enable_sglang_deepep_moe_alignment(
        args,
        model,
        store_prefix="",
        selected_layers=layers,
    )
