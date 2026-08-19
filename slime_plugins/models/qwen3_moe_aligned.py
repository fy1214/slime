# Qwen3-30B-A3B SGLang-aligned self-attention plugin.
#
# Purpose: bit-align Megatron train-side attention with SGLang rollout-side
# attention for Qwen3-30B-A3B (standard GQA, 32 Q / 4 KV heads, head_dim=128,
# q_norm+k_norm on head_dim, RoPE base=1e6).
#
# Strategy mirrors slime_plugins.models.glm5.glm5:
#   * Register a spec that REPLACES layer.self_attention with a custom module
#     while leaving mlp untouched (the DeepGEMM MoE forward hook already
#     aligns the MLP path with SGLang).
#   * Custom module uses plain Megatron ColumnParallelLinear / RowParallelLinear
#     (NOT TELayerNormColumnParallelLinear) so `linear_qkv` is a pure matmul
#     and cannot double-normalize its input.
#   * Custom module reads the FP32 residual sum written by transformer_layer
#     ("_use_sglang_fused_residual_rmsnorm" path), does the RMSNorm itself in
#     FP32, then runs qkv_proj -> split -> q_norm/k_norm -> SGLang RoPE ->
#     flash_attn.cute.flash_attn_varlen_func (FA4) -> o_proj. Every step
#     mirrors SGLang qwen3_moe.py Qwen3MoeAttention.forward_prepare_native +
#     forward_core.

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from megatron.core import mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import get_num_layers_to_build

_SGLANG_ROPE_CACHE: dict = {}


# ---------------------------------------------------------------------------
# SGLang-aligned RoPE (bit-identical to sglang.srt.layers.rotary_embedding when
# used with base=rope_theta, is_neox=True; matches GLM-5 plugin's approach).
# ---------------------------------------------------------------------------

@torch.no_grad()
def _apply_sglang_rope_forward(
    value: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace

    output = torch.empty_strided(
        value.size(), value.stride(), dtype=value.dtype, device=value.device
    )
    output.copy_(value)
    dummy_k = torch.empty(
        (value.shape[0], 1, value.shape[-1]),
        dtype=value.dtype,
        device=value.device,
    )
    apply_rope_with_cos_sin_cache_inplace(
        output,
        dummy_k,
        cos_sin_cache,
        positions,
        is_neox=True,
    )
    return output


class _SGLangRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, cos_sin_cache, positions):
        ctx.save_for_backward(cos_sin_cache, positions)
        ctx.is_neox = True
        return _apply_sglang_rope_forward(value, cos_sin_cache, positions)

    @staticmethod
    def backward(ctx, grad_output):
        cos_sin_cache, positions = ctx.saved_tensors
        half = grad_output.shape[-1] // 2
        broadcast_shape = (positions.numel(),) + (1,) * (grad_output.ndim - 2) + (half,)
        cos = cos_sin_cache[positions, :half].view(broadcast_shape).to(grad_output.dtype)
        sin = cos_sin_cache[positions, half:].view(broadcast_shape).to(grad_output.dtype)
        # is_neox=True: pairs are (x[:half], x[half:]) rotated as
        #   out[:half] = x[:half] * cos - x[half:] * sin
        #   out[half:] = x[half:] * cos + x[:half] * sin
        # Gradient is the transpose:
        grad_first = grad_output[..., :half]
        grad_second = grad_output[..., half:]
        grad_input = torch.cat(
            (
                grad_first * cos + grad_second * sin,
                grad_second * cos - grad_first * sin,
            ),
            dim=-1,
        )
        return grad_input, None, None


def _get_sglang_rope_cache(
    device: torch.device,
    rotary_dim: int,
    rotary_base: float,
    needed_positions: int,
) -> torch.Tensor:
    key = (device.type, device.index, rotary_dim, float(rotary_base))
    cache = _SGLANG_ROPE_CACHE.get(key)
    if cache is not None and cache.shape[0] >= needed_positions:
        return cache
    inv_freq = 1.0 / (
        rotary_base
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    positions = torch.arange(needed_positions, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    _SGLANG_ROPE_CACHE[key] = cache
    return cache


def _apply_rope(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_dim: int,
    rotary_base: float,
) -> torch.Tensor:
    """Apply SGLang-aligned RoPE to x with shape [T, H, D]. Positions are
    reconstructed from cu_seqlens so pack layout matches SGLang."""

    token_ids = torch.arange(x.shape[0], dtype=torch.int64, device=x.device)
    seq_ids = torch.searchsorted(cu_seqlens[1:], token_ids, right=True)
    positions = token_ids - cu_seqlens[seq_ids]
    cache = _get_sglang_rope_cache(
        x.device,
        rotary_dim,
        rotary_base,
        int(positions.max().item()) + 1,
    )
    if torch.is_grad_enabled() and x.requires_grad:
        return _SGLangRoPE.apply(x, cache, positions)
    return _apply_sglang_rope_forward(x, cache, positions)



# ---------------------------------------------------------------------------
# SGLang-aligned per-token RMSNorm for input_layernorm.
#
# Rationale: on layer 0, transformer_layer.py's FP32-residual-sum branch is
# skipped (no residual sum yet), and it falls through to
# `self.input_layernorm(hidden_states)`. Baseline (TE-fused linear_qkv) routes
# through the DeepGEMM `_norm_forward` hook, which short-circuits to
# `sglang.srt.batch_invariant_ops.rms_norm_batch_invariant` when
# MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS=1. Our plain-linear_qkv plugin does
# not go through that hook; using TENorm here therefore diverges from SGLang.
# This wrapper is a drop-in replacement that calls rms_norm_batch_invariant
# directly, matching SGLang's Qwen3MoeDecoderLayer.forward on layer 0.
# ---------------------------------------------------------------------------


class _SGLangBatchInvariantRMSNorm(torch.nn.Module):
    """Weight-only RMSNorm whose forward is bit-identical to SGLang.

    Signature matches Megatron's ``TENorm(config, hidden_size, eps=1e-5)`` so
    it can be dropped into ``layer_specs.submodules.input_layernorm``.
    """

    def __init__(self, config, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        import os as _os
        if _os.environ.get("QWEN3_ALIGNED_PROBE", "0") == "1":
            print(f"[QWEN3-ALIGNED v12-PROBE] _SGLangBatchInvariantRMSNorm __init__ hidden={hidden_size} eps={eps}", flush=True)
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        # Mark weight as sequence-parallel-safe for Megatron gradient allreduce.
        setattr(self.weight, "sequence_parallel", bool(getattr(config, "sequence_parallel", False)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        try:
            from sglang.srt.batch_invariant_ops import rms_norm_batch_invariant
        except ImportError:
            rms_norm_batch_invariant = None
        if rms_norm_batch_invariant is not None:
            out_shape = hidden_states.shape
            x_flat = hidden_states.reshape(-1, self.hidden_size)
            out = rms_norm_batch_invariant(x_flat, self.weight, self.eps)
            return out.view(out_shape)
        # FP32-reduction fallback (matches _sglang_native_rmsnorm_from_fp32_sum).
        x_fp32 = hidden_states.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_fp32 * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Submodule dataclass + custom self-attention module for Qwen3-30B-A3B.
# ---------------------------------------------------------------------------

@dataclass
class Qwen3MoeAlignedSelfAttentionSubmodules:
    """Submodules for the SGLang-aligned Qwen3 MoE self-attention layer."""

    linear_qkv: ModuleSpec | type = None
    linear_proj: ModuleSpec | type = None
    q_layernorm: ModuleSpec | type = None
    k_layernorm: ModuleSpec | type = None


class Qwen3MoeAlignedSelfAttention(MegatronModule):
    """SGLang-aligned Qwen3 MoE self-attention.

    Mirrors :class:`Qwen3MoeAttention` from
    ``sglang.srt.models.qwen3_moe`` at the arithmetic level:

    1. ``qkv_proj(hidden_states)``  (plain ColumnParallelLinear)
    2. split into Q / K / V of sizes ``[q_size, kv_size, kv_size]``
    3. reshape to ``[T, H, head_dim]`` and apply per-head RMSNorm
       (``q_norm`` / ``k_norm`` with weight of shape ``[head_dim]``)
    4. apply SGLang RoPE (``is_neox=True``, base = ``rotary_base``)
    5. ``flash_attn.cute.flash_attn_varlen_func`` (FA4) — the same kernel
       SGLang uses with ``--attention-backend fa4``
    6. ``o_proj`` (plain RowParallelLinear)

    The Megatron ``transformer_layer`` still owns the residual-add and the
    input-layernorm hop. This module accepts ``hidden_states`` that has
    already been normalized. Alignment with SGLang RMSNorm is achieved by
    setting the layer's ``input_layernorm`` to the local ``FusedLayerNorm``
    (which has a plain ``.weight``); with
    ``MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS=1`` the layer replaces its
    output with the FP32-residual-sum RMSNorm value, and because our
    ``linear_qkv`` is a *plain* ColumnParallelLinear (not a TE fused
    LayerNormColumnParallelLinear) there is no double normalization.
    """

    def __init__(
        self,
        config,
        submodules: Qwen3MoeAlignedSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: Optional[str] = None,
        model_comm_pgs=None,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type

        tp_size = mpu.get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % tp_size == 0, (
            f"num_attention_heads={config.num_attention_heads} not divisible by TP={tp_size}"
        )

        total_num_heads = config.num_attention_heads
        total_num_kv_heads = getattr(config, "num_query_groups", None) or config.num_attention_heads
        assert total_num_heads % total_num_kv_heads == 0

        self.head_dim = getattr(config, "kv_channels", None) or (
            config.hidden_size // total_num_heads
        )
        # v34c: get rotary_base from TransformerConfig, but also check
        # the HF config via args if TransformerConfig has the default.
        _tc_rb = float(getattr(config, "rotary_base", 10000.0))
        if _tc_rb == 10000.0:
            # TransformerConfig may not have the correct value;
            # try to get it from the HF model config.
            import os as _os34c
            _hf_path = _os34c.environ.get("HF_CHECKPOINT", "")
            if not _hf_path:
                # fallback: try reading from megatron args
                try:
                    from megatron.training import get_args
                    _args34 = get_args()
                    _hf_path = getattr(_args34, "hf_checkpoint", "")
                    _tc_rb = float(getattr(_args34, "rotary_base", 10000.0))
                except Exception:
                    pass
        self.rotary_base = _tc_rb
        self.rms_norm_eps = float(config.layernorm_epsilon)

        if total_num_kv_heads >= tp_size:
            assert total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % total_num_kv_heads == 0

        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # QKV projection: plain ColumnParallelLinear over [q_size + 2*kv_size].
        # Bias controlled by attention_bias (Qwen3 = False).
        attn_bias = not getattr(config, "add_bias_linear", False) is False
        # Actually: --disable-bias-linear -> add_bias_linear=False. Qwen3 has no attn bias.
        # Force off:
        attn_bias = False

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            config.hidden_size,
            total_num_heads * self.head_dim + 2 * total_num_kv_heads * self.head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=attn_bias,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        )

        # Output projection: plain RowParallelLinear.
        self.linear_proj = build_module(
            submodules.linear_proj,
            total_num_heads * self.head_dim,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=attn_bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )

        # Per-head RMSNorm on head_dim. Match SGLang RMSNorm(head_dim, eps).
        # We use torch.nn.RMSNorm (available in PyTorch >= 2.4) for a portable,
        # weight-only RMSNorm that matches SGLang bit-for-bit when both sides
        # run FP32-cast internal reduction (rms_norm_batch_invariant path).
        self.q_norm = torch.nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.k_norm = torch.nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        import os as _os
        if _os.environ.get("QWEN3_ALIGNED_PROBE", "0") == "1":
            print(f"[QWEN3-ALIGNED v12-PROBE] Qwen3MoeAlignedSelfAttention __init__ layer={layer_number} q_size={self.q_size} kv_size={self.kv_size}", flush=True)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head RMSNorm over the head_dim axis.

        Follows sglang.srt.models.utils.apply_qk_norm: reshape flat
        [..., num_heads * head_dim] into [..., num_heads, head_dim],
        RMSNorm on the last axis with a shared weight, then flatten.
        """

        if os.getenv("MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS", "0") == "1":
            try:
                from sglang.srt.batch_invariant_ops import rms_norm_batch_invariant
            except ImportError:
                rms_norm_batch_invariant = None
            if rms_norm_batch_invariant is not None:
                q_flat = q.reshape(-1, self.head_dim)
                k_flat = k.reshape(-1, self.head_dim)
                q_flat = rms_norm_batch_invariant(
                    q_flat, self.q_norm.weight, self.rms_norm_eps
                )
                k_flat = rms_norm_batch_invariant(
                    k_flat, self.k_norm.weight, self.rms_norm_eps
                )
                return q_flat.view_as(q), k_flat.view_as(k)

        # Fallback: torch.nn.functional.rms_norm with FP32 reduction.
        q_out = F.rms_norm(
            q.float(),
            normalized_shape=(self.head_dim,),
            weight=self.q_norm.weight.float(),
            eps=self.rms_norm_eps,
        ).to(q.dtype)
        k_out = F.rms_norm(
            k.float(),
            normalized_shape=(self.head_dim,),
            weight=self.k_norm.weight.float(),
            eps=self.rms_norm_eps,
        ).to(k.dtype)
        return q_out, k_out


    def _apply_fused_qk_norm_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """v18: SGLang fused QK-Norm + RoPE (bit-identical to
        sglang.srt.models.qwen3_moe.Qwen3MoeAttention.apply_qk_norm_rope
        when use_fused_qk_norm_rope=True).

        query: [T, num_heads, head_dim] bf16
        key:   [T, num_kv_heads, head_dim] bf16
        value: [T, num_kv_heads, head_dim] bf16 (unused here, only for shape)
        Returns (query, key) after QK-norm + RoPE. Value untouched.
        """
        from sglang.jit_kernel.fused_qknorm_rope import fused_qk_norm_rope
        T = query.shape[0]
        # Pack to SGLang block-concat layout [Q(nq*hd), K(nkv*hd), V(nkv*hd)].
        q_flat = query.reshape(T, self.num_heads * self.head_dim)
        k_flat = key.reshape(T, self.num_kv_heads * self.head_dim)
        v_flat = value.reshape(T, self.num_kv_heads * self.head_dim)
        qkv_packed = torch.cat([q_flat, k_flat, v_flat], dim=-1).contiguous()
        # Position ids reconstructed from cu_seqlens_q (same as _apply_rope).
        cu_q = packed_seq_params.cu_seqlens_q
        token_ids = torch.arange(T, dtype=torch.int64, device=qkv_packed.device)
        seq_ids = torch.searchsorted(cu_q[1:], token_ids, right=True)
        positions = (token_ids - cu_q[seq_ids]).to(torch.int32)
        q_w = self.q_norm.weight.to(torch.bfloat16)
        k_w = self.k_norm.weight.to(torch.bfloat16)
        # v32: dump positions before fused_qk_norm_rope
        if not getattr(self, '_v32_pos_dumped', False):
            import torch as _t32b, os as _o32b
            _dd32b = '/tmp/qwen3_layer0_dumps/positions_meg'
            _o32b.makedirs(_dd32b, exist_ok=True)
            _rk32b = _t32b.distributed.get_rank() if _t32b.distributed.is_initialized() else 0
            _t32b.save(positions.detach().cpu(), _dd32b + f'/positions.rank{_rk32b}.pt')
            _t32b.save(qkv_packed.detach().cpu(), _dd32b + f'/qkv_packed_pre_rope.rank{_rk32b}.pt')
            self._v32_pos_dumped = True
        # v33b: print scalar params
        if not getattr(self, '_v33b_printed', False):
            import torch as _t33b
            _rk33b = _t33b.distributed.get_rank() if _t33b.distributed.is_initialized() else 0
            if _rk33b == 0:
                print(f'[v33b-MEG] rotary_base={self.rotary_base} eps={self.rms_norm_eps}')
                print(f'[v33b-MEG] num_heads={self.num_heads} num_kv_heads={self.num_kv_heads} head_dim={self.head_dim}')
                print(f'[v33b-MEG] q_w dtype={q_w.dtype} shape={q_w.shape} sum={q_w.sum().item():.6f}')
                print(f'[v33b-MEG] k_w dtype={k_w.dtype} shape={k_w.shape} sum={k_w.sum().item():.6f}')
            self._v33b_printed = True
        fused_qk_norm_rope(
            qkv_packed,
            self.num_heads,
            self.num_kv_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rms_norm_eps,
            q_w,
            k_w,
            self.rotary_base,
            True,  # is_neox
            positions,
            1.0,  # yarn factor
            0.0,  # yarn low
            0.0,  # yarn high
            1.0,  # yarn attention_factor
        )
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q_out, k_out, v_out = qkv_packed.split([q_size, kv_size, kv_size], dim=-1)
        query = q_out.reshape(T, self.num_heads, self.head_dim)
        key = k_out.reshape(T, self.num_kv_heads, self.head_dim)
        # v22: return value view of qkv_packed so fa4 sees identical stride to SGL.
        value_v = v_out.reshape(T, self.num_kv_heads, self.head_dim)
        return query, key, value_v

    def _fa4_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> torch.Tensor:
        """FA4 varlen forward. query/key/value are [T, H, D] (already squeezed)."""

        from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func

        cu_q = packed_seq_params.cu_seqlens_q.to(torch.int32)
        cu_kv = packed_seq_params.cu_seqlens_kv.to(torch.int32)
        max_sq = int(packed_seq_params.max_seqlen_q)
        max_sk = int(packed_seq_params.max_seqlen_kv)

        # v27: dump fa4 kwargs
        if getattr(self, 'layer_number', 0) == 1 and not getattr(self, '_v27_kw_dumped', False):
            import torch as _t27, os as _o27
            _dd = '/tmp/qwen3_layer0_dumps/fa_kwargs_plugin'
            _o27.makedirs(_dd, exist_ok=True)
            _rk = _t27.distributed.get_rank() if _t27.distributed.is_initialized() else 0
            _kw = {
                'q_shape': tuple(query.shape),
                'q_dtype': str(query.dtype),
                'q_stride': tuple(query.stride()),
                'q_is_contig': bool(query.is_contiguous()),
                'k_shape': tuple(key.shape),
                'k_dtype': str(key.dtype),
                'k_stride': tuple(key.stride()),
                'v_shape': tuple(value.shape),
                'v_dtype': str(value.dtype),
                'v_stride': tuple(value.stride()),
                'cu_q': cu_q.detach().cpu().tolist(),
                'cu_kv': cu_kv.detach().cpu().tolist(),
                'max_sq': max_sq, 'max_sk': max_sk,
                'softmax_scale': float(self.scaling),
                'causal': True,
                'softcap': 0.0,
            }
            import json as _j27
            with open(_dd + f'/rank{_rk}.json','w') as _f: _j27.dump(_kw, _f, indent=2)
            self._v27_kw_dumped = True
        # v29: unpack packed batch, per-seq fa4
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        # v31: dump fa4 input tensors
        if getattr(self, 'layer_number', 0) == 1 and not getattr(self, '_v31_dumped', False):
            import torch as _t31, os as _o31
            _dd31 = '/tmp/qwen3_layer0_dumps/fa4_inputs_meg'
            _o31.makedirs(_dd31, exist_ok=True)
            _rk31 = _t31.distributed.get_rank() if _t31.distributed.is_initialized() else 0
            _t31.save(query.detach().cpu(), _dd31 + f'/q.rank{_rk31}.pt')
            _t31.save(key.detach().cpu(), _dd31 + f'/k.rank{_rk31}.pt')
            _t31.save(value.detach().cpu(), _dd31 + f'/v.rank{_rk31}.pt')
            _t31.save(cu_q.detach().cpu(), _dd31 + f'/cu_q.rank{_rk31}.pt')
            _t31.save(cu_kv.detach().cpu(), _dd31 + f'/cu_kv.rank{_rk31}.pt')
            self._v31_dumped = True
        n_seq = cu_q.numel() - 1
        outs = []
        for _s in range(n_seq):
            _sq = int(cu_q[_s+1] - cu_q[_s])
            _sk = int(cu_kv[_s+1] - cu_kv[_s])
            if _sq == 0:
                outs.append(query.new_zeros(0, query.shape[1], query.shape[2]))
                continue
            _q1 = query[int(cu_q[_s]):int(cu_q[_s+1])].contiguous()
            _k1 = key[int(cu_kv[_s]):int(cu_kv[_s+1])].contiguous()
            _v1 = value[int(cu_kv[_s]):int(cu_kv[_s+1])].contiguous()
            _cu1 = torch.tensor([0, _sq], device=query.device, dtype=torch.int32)
            _ck1 = torch.tensor([0, _sk], device=query.device, dtype=torch.int32)
            _o1 = flash_attn_varlen_func(
                _q1, _k1, _v1,
                cu_seqlens_q=_cu1,
                cu_seqlens_k=_ck1,
                max_seqlen_q=_sq,
                max_seqlen_k=_sk,
                softmax_scale=self.scaling,
                causal=True,
                softcap=0.0,
            )
            if isinstance(_o1, tuple):
                _o1 = _o1[0]
            outs.append(_o1)
        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        return out

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb=None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        import os as _os
        import torch as _th
        _dump = _os.environ.get("QWEN3_ALIGNED_LAYER0_DUMP", "0") == "1" and self.layer_number == 1
        if _dump and not getattr(self, "_dumped", False):
            _ddir = "/tmp/qwen3_layer0_dumps/megatron"
            _os.makedirs(_ddir, exist_ok=True)
            _RANK = _th.distributed.get_rank() if _th.distributed.is_initialized() else 0
            _th.save(hidden_states.detach().float().cpu(), _ddir + f"/plugin_input_layernorm_output.rank{_RANK}.pt")
        assert packed_seq_params is not None, (
            "Qwen3MoeAlignedSelfAttention requires packed_seq_params (THD layout)"
        )
        assert inference_context is None and inference_params is None, (
            "Qwen3MoeAlignedSelfAttention is training-only (no KV-cache path)"
        )

        # hidden_states: [s, b, h]. We keep this layout throughout the module.
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        # mixed_qkv: [T, 1, num_groups * (heads_per_group + 2) * head_dim].
        # Megatron GQA layout (see hf_to_megatron/common.merge_qkv) interleaves
        # per KV group: [Q_g0, K_g0, V_g0, Q_g1, K_g1, V_g1, ...]. We MUST
        # respect that layout when splitting, otherwise Q/K/V get scrambled.
        num_groups = self.num_kv_heads
        heads_per_group = self.num_heads // num_groups
        per_group = (heads_per_group + 2) * self.head_dim
        T = mixed_qkv.shape[0]
        mixed = mixed_qkv.view(T, 1, num_groups, per_group)
        q_chunk = heads_per_group * self.head_dim
        query = mixed[..., :q_chunk]
        key   = mixed[..., q_chunk:q_chunk + self.head_dim]
        value = mixed[..., q_chunk + self.head_dim:]
        query = query.reshape(T, self.num_heads, self.head_dim)
        key   = key.reshape(T, num_groups, self.head_dim)
        value = value.reshape(T, num_groups, self.head_dim)

        # v18: optional SGLang fused QK-Norm + RoPE path.
        _use_fused_qk_rope = _os.environ.get("QWEN3_ALIGNED_USE_FUSED_QK_ROPE", "0") == "1" and query.dtype == _th.bfloat16
        if _use_fused_qk_rope:
            query, key, value = self._apply_fused_qk_norm_rope(query, key, value, packed_seq_params)
            # v32: dump positions
            if getattr(self, 'layer_number', 0) == 1 and not getattr(self, '_v32_dumped', False):
                import torch as _t32, os as _o32
                _dd32 = '/tmp/qwen3_layer0_dumps/positions_meg'
                _o32.makedirs(_dd32, exist_ok=True)
                _rk32 = _t32.distributed.get_rank() if _t32.distributed.is_initialized() else 0
                _t32.save(packed_seq_params.cu_seqlens_q.detach().cpu(), _dd32 + f'/cu_seqlens_q.rank{_rk32}.pt')
                _t32.save(packed_seq_params.cu_seqlens_kv.detach().cpu(), _dd32 + f'/cu_seqlens_kv.rank{_rk32}.pt')
                _t32.save(packed_seq_params.qkv_format.value if hasattr(packed_seq_params.qkv_format, 'value') else torch.tensor(0), _dd32 + f'/qkv_format.rank{_rk32}.txt')
                self._v32_dumped = True
        else:
            # Per-head RMSNorm on head_dim (matches SGLang apply_qk_norm).
            query, key = self._apply_qk_norm(query, key)

            # RoPE (SGLang-aligned). Positions are derived from cu_seqlens_q/kv.
            cu_q = packed_seq_params.cu_seqlens_q
            cu_kv = packed_seq_params.cu_seqlens_kv
            query = _apply_rope(query, cu_q, self.head_dim, self.rotary_base)
            key = _apply_rope(key, cu_kv, self.head_dim, self.rotary_base)

        # FA4 varlen attention. Output is [T, num_heads * head_dim].
        core_attn_out = self._fa4_attention(query, key, value, packed_seq_params)
        # core_attn_out: [T, num_heads, head_dim] -> flatten heads
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], -1)
        # Restore [T, 1, num_heads*head_dim] so RowParallelLinear sees the
        # expected THD-with-batch-1 layout (matches Megatron's SelfAttention
        # which does the same unsqueeze via `packed_seq_params` handling).
        core_attn_out = core_attn_out.unsqueeze(1)

        output, bias = self.linear_proj(core_attn_out)
        if _dump and not getattr(self, "_dumped", False):
            _th.save(mixed_qkv.detach().float().cpu(), _ddir + f"/plugin_mixed_qkv.rank{_RANK}.pt")
            _th.save(query.detach().float().cpu(), _ddir + f"/plugin_query_after_rope.rank{_RANK}.pt")
            _th.save(key.detach().float().cpu(), _ddir + f"/plugin_key_after_rope.rank{_RANK}.pt")
            _th.save(value.detach().float().cpu(), _ddir + f"/plugin_value.rank{_RANK}.pt")
            _th.save(core_attn_out.detach().float().cpu(), _ddir + f"/plugin_core_attn_out.rank{_RANK}.pt")
            _th.save(output.detach().float().cpu(), _ddir + f"/plugin_attention_output.rank{_RANK}.pt")
            self._dumped = True
            print(f"[MG-HOOK] plugin layer 0 dumps saved to {_ddir}", flush=True)
        return output, bias


# ---------------------------------------------------------------------------
# Spec factory
# ---------------------------------------------------------------------------

def get_qwen3_moe_aligned_spec(args, config, vp_stage):
    """Build a Qwen3-30B-A3B transformer block spec that replaces the default
    self_attention with :class:`Qwen3MoeAlignedSelfAttention` while keeping the
    MLP / MoE path (DeepGEMM forward hook aligns it with SGLang).
    """

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    # For every local layer, swap self_attention. input_layernorm stays as the
    # default (which for TE spec is IdentityOp because norm is fused into
    # linear_qkv). We must therefore also swap input_layernorm to a real norm
    # so the FP32-residual-sum RMSNorm branch in transformer_layer can read
    # `.weight`, AND swap linear_qkv to plain ColumnParallelLinear so the
    # normalized input is not re-normalized by TE.
    # Use TENorm which honours config.normalization ("RMSNorm" for Qwen3).
    # v11: TENorm replaced by _SGLangBatchInvariantRMSNorm (batch-invariant kernel).
    from megatron.core.transformer.identity_op import IdentityOp

    self_attn_module_spec = ModuleSpec(
        module=Qwen3MoeAlignedSelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=Qwen3MoeAlignedSelfAttentionSubmodules(
            linear_qkv=ColumnParallelLinear,
            linear_proj=RowParallelLinear,
            q_layernorm=IdentityOp,  # unused; per-head norm is built internally
            k_layernorm=IdentityOp,
        ),
    )

    for layer_id in range(num_layers_to_build):
        layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
        layer_specs.submodules.self_attention = self_attn_module_spec
        # Ensure input_layernorm is a real RMSNorm (has .weight) instead of
        # the TE IdentityOp that fuses into linear_qkv.
        layer_specs.submodules.input_layernorm = _SGLangBatchInvariantRMSNorm
        transformer_layer_spec.layer_specs[layer_id] = layer_specs

    import os as _os
    if _os.environ.get("QWEN3_ALIGNED_PROBE", "0") == "1":
        _norm = transformer_layer_spec.layer_specs[0].submodules.input_layernorm
        _sa = transformer_layer_spec.layer_specs[0].submodules.self_attention
        print(f"[QWEN3-ALIGNED v12-PROBE] spec factory returning: input_layernorm={_norm}, self_attention={_sa}", flush=True)
    return transformer_layer_spec
