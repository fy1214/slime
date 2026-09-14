"""Diagnostic-only backward from actual forward NVFP4 operands.

Forward is unmodified production CUTLASS. Backward uses BF16 dequantized
rowwise operands, actual forward gate/up and down output, identity STE through
quantizers and cast boundaries, stop-gradient scales, and BF16 GEMMs.
Not a claim of equivalence to every TE recipe or the discrete true derivative.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def decode(packed, scales):
    rows, half_k = packed.shape
    k = half_k * 2
    sf = scales.view(torch.float8_e4m3fn).float()
    sf = sf.reshape(rows // 128, k // 64, 32, 4, 4).permute(0, 1, 3, 2, 4)
    sf = sf.reshape(rows // 128, k // 64, 128, 4).permute(0, 2, 1, 3).reshape(rows, k // 16)
    p = packed.int()
    codes = torch.stack([p & 15, p >> 4], -1).flatten(-2)
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., 0., -.5, -1., -1.5, -2., -3., -4., -6.], device=p.device)
    return lut[codes] * sf.repeat_interleave(16, -1)


@contextmanager
def capture_forward():
    # Scoped to one synchronous expert forward in one actor process. Do not use
    # with concurrent forward threads in the same Python process.
    import sglang.srt.layers.quantization.fp4_utils as u
    capture = {'quant': [], 'gemm': []}
    oq, og = u.nvfp4_quantize_pertoken, u.nvfp4_grouped_gemm

    def quant(*args, **kwargs):
        result = oq(*args, **kwargs)
        capture['quant'].append(result)
        return result

    def gemm(*args, **kwargs):
        result = og(*args, **kwargs)
        # FC2 is subsequently compacted/probability-weighted in place.
        capture['gemm'].append(args[0].detach().clone())
        return result

    u.nvfp4_quantize_pertoken, u.nvfp4_grouped_gemm = quant, gemm
    try:
        yield capture
    finally:
        u.nvfp4_quantize_pertoken, u.nvfp4_grouped_gemm = oq, og


def run_captured(module, x, counts, probs, layout):
    from slime.backends.megatron_utils.alignment import cutlass_nvfp4_moe_forward as c
    state = c._build_nvfp4_moe_state(module, layout, module_name='dq_diagnostic')
    with capture_forward() as capture:
        out = c._expert_major_cutlass_nvfp4_moe_forward(module, x, counts, probs, state=state, layout=layout)
    if x.shape[0]:
        assert len(capture['quant']) == len(capture['gemm']) == 2
    capture['state'] = state
    capture['counts'] = tuple(counts.tolist())
    return out, capture


def expert_operands(capture, expert, offset, count):
    state = capture['state']
    pad_offset = sum(((n + 127) // 128) * 128 for n in capture['counts'][:expert])
    padded = ((count + 127) // 128) * 128
    acts = []
    for data, sf, gs in capture['quant']:
        k = data.shape[1] * 2
        raw = decode(data[pad_offset:pad_offset+padded], sf[pad_offset*(k//16):(pad_offset+padded)*(k//16)])
        acts.append((raw[:count] * gs[pad_offset:pad_offset+count,None]).bfloat16())
    weights = []
    for packed, sf, offsets, gs in [
        (state.w1_fp4, state.w1_blockscale_flat, state.w1_scale_offsets, state.w1_weight_scale_2),
        (state.w2_fp4, state.w2_blockscale_flat, state.w2_scale_offsets, state.w2_weight_scale_2),
    ]:
        lo, hi = int(offsets[expert]), int(offsets[expert+1])
        weights.append((decode(packed[expert], sf[lo:hi]) * gs[expert]).bfloat16())
    gu, down = [v[pad_offset:pad_offset+count] for v in capture['gemm']]
    return acts[0], acts[1], weights[0], weights[1], gu, down


def analytic_backward(capture, probs, grad, weights, needs, defer=False):
    n_exp = len(capture['counts'])
    dx = torch.empty_like(grad) if needs[0] else None
    dp = torch.empty_like(probs) if needs[2] and not defer else None
    d1, d2 = [None]*n_exp, [None]*n_exp
    offset = 0
    for e, count in enumerate(capture['counts']):
        if count == 0:
            if needs[6+e]: d1[e] = torch.zeros_like(weights[e])
            if needs[6+n_exp+e]: d2[e] = torch.zeros_like(weights[n_exp+e])
            continue
        xq, hq, w1, w2, gu, down = expert_operands(capture, e, offset, count)
        sl = slice(offset, offset+count)
        g = grad[sl].contiguous().bfloat16()
        if dp is not None:
            dp[sl] = (g.float()*down.float()).sum(-1).reshape_as(dp[sl]).to(dp.dtype)
        dy = g if defer else (g.float()*probs[sl].reshape(-1,1).float()).bfloat16()
        dh = dy @ w2
        gate, up = gu.float().chunk(2,-1)
        sig = gate.sigmoid()
        dg = dh.float()*up*sig*(1+gate*(1-sig))
        du = dh.float()*F.silu(gate)
        dgu = torch.cat([dg,du],-1).bfloat16()
        if dx is not None: dx[sl] = dgu @ w1
        if needs[6+e]:
            d1[e] = (dgu.T @ xq).to(weights[e].dtype)
        if needs[6+n_exp+e]:
            d2[e] = (dy.T @ hq).to(weights[n_exp+e].dtype)
        offset += count
    return dx, dp, d1, d2


class Nvfp4WithDequantizedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, counts, probs, module, layout, module_name, *weights):
        out, ctx.capture = run_captured(module, x, counts, probs, layout)
        ctx.defer = bool(getattr(module, '_slime_defer_router_probabilities', False))
        ctx.save_for_backward(probs, *weights)
        return out

    @staticmethod
    def backward(ctx, grad):
        probs, *weights = ctx.saved_tensors
        dx, dp, d1, d2 = analytic_backward(ctx.capture, probs, grad, weights, ctx.needs_input_grad, ctx.defer)
        return dx, None, dp, None, None, None, *d1, *d2


def install():
    from slime.backends.megatron_utils.alignment import cutlass_nvfp4_moe_forward as c
    c._CutlassNvfp4MoEWithBF16Backward = Nvfp4WithDequantizedBackward
