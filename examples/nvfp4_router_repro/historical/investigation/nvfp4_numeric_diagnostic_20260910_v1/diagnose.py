"""Read-only, single-GPU NVFP4 diagnostics; never performs optimizer updates.

Full-model forward is a separate eager reference, not a formal Megatron
regression. All modes consume identical archived tokens and BF16 HF weights.
STE means identity derivative through quantizers, with scales stop-gradient.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ROOT = Path(__file__).resolve().parent
MODEL = Path('/lustre/fsw/general_sa/shuazhang/models/Qwen/Qwen3-30B-A3B-Base')
BATCH = Path('/lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/investigation/determinism_backward_audit_20260909/full_model_fixed_v1/batch.pt')
INDEX = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
RESULTS = []


def emit(kind, **values):
    row = dict(kind=kind, **values)
    RESULTS.append(row)
    print(json.dumps(row), flush=True)
    (ROOT / f'results_{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(RESULTS, indent=2))


def weight(name):
    with safe_open(MODEL / INDEX[name], framework='pt', device='cpu') as f:
        return f.get_tensor(name).to(device='cuda', dtype=torch.bfloat16)


def stats(a, b):
    a, b = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    return dict(relative_l2=((a-b).norm()/b.norm().clamp_min(1e-30)).item(),
                cosine=F.cosine_similarity(a, b, dim=0).item(),
                norm_ratio=(a.norm()/b.norm().clamp_min(1e-30)).item(),
                max_abs=(a-b).abs().max().item(), finite=bool(torch.isfinite(a).all()))


def quant_ref(x, per_row):
    """Independent RTNE E2M1 + E4M3 block scaling (16 elements).

    FP32 decode value retained for reference GEMMs; no BF16 dequant rounding.
    No production quantizer or dequantizer is called here.
    """
    x = x.detach().float()
    amax = x.abs().amax(dim=-1, keepdim=True) if per_row else x.abs().amax()
    encode = torch.where(amax == 0, 1., 2688. / amax)
    gs = encode.reciprocal()
    blocks = x.reshape(x.shape[0], -1, 16)
    bs = (blocks.abs().amax(-1) * (1./6.) * encode).clamp(max=448).to(torch.float8_e4m3fn).float()
    scale = bs.repeat_interleave(16, -1) * gs
    z = torch.where(scale == 0, 0., x / scale)
    edges = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=x.device)
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
    idx = torch.bucketize(z.abs().contiguous(), edges)
    tie = (idx < 7) & (z.abs() == edges[idx.clamp(max=6)]) & ((idx % 2) == 1)
    idx = idx + tie.long()
    return grid[idx] * z.sign() * scale


def ste(x, q):
    return x.float() + (q - x.float()).detach()


class Experts(torch.nn.Module):
    def __init__(self, layer, ids):
        super().__init__()
        self.linear_fc1, self.linear_fc2 = torch.nn.Module(), torch.nn.Module()
        self.ids = ids
        for i, e in enumerate(ids):
            p = f'model.layers.{layer}.mlp.experts.{e}.'
            w1 = torch.cat([weight(p+'gate_proj.weight'), weight(p+'up_proj.weight')])
            w2 = weight(p+'down_proj.weight')
            self.linear_fc1.register_parameter(f'weight{i}', torch.nn.Parameter(w1, requires_grad=False))
            self.linear_fc2.register_parameter(f'weight{i}', torch.nn.Parameter(w2, requires_grad=False))

    def weights(self):
        return tuple(self.linear_fc1.parameters()), tuple(self.linear_fc2.parameters())


def native_expert(x, w1, w2, mode):
    if mode == 'bf16':
        gu = F.linear(x, w1)
        g, u = gu.chunk(2, -1)
        return F.linear((F.silu(g.float())*u.float()).to(x.dtype), w2)
    q1, q2 = quant_ref(w1, False), quant_ref(w2, False)
    if mode == 'weight_only':
        a = x.float()
    else:
        a = quant_ref(x, True)
    gu = F.linear(a, q1).to(x.dtype)
    g, u = gu.chunk(2, -1)
    h = (F.silu(g.float())*u.float()).to(x.dtype)
    hq = h.float() if mode == 'weight_only' else quant_ref(h, True)
    return F.linear(hq, q2).to(x.dtype)


def backward_probe(module, x, counts, label):
    from slime.backends.megatron_utils.alignment.moe_bf16_expert_backward import moe_bf16_expert_backward
    from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import _MoELayout
    w1, w2 = module.weights()
    probs = torch.linspace(.05, .25, len(x), device='cuda', dtype=torch.float32)
    torch.manual_seed(813)
    upstream = torch.randn_like(x) * .01
    layout = _MoELayout(len(counts), 2048, 768)
    with torch.no_grad():
        got = moe_bf16_expert_backward(hidden_states=x.clone(), permuted_probs=probs,
            grad_output=upstream, fc1_weights=w1, fc2_weights=w2, counts=tuple(counts),
            layout=layout, module_name='diagnostic', needs_hidden=True, needs_probs=True,
            needs_fc1_weights=(True,)*len(counts), needs_fc2_weights=(True,)*len(counts),
            defer_router_probabilities=False, reuse_expert_input_for_grad=False, grad_workspace=None)
    offset = 0
    for e, n in enumerate(counts):
        if not n:
            continue
        sl = slice(offset, offset+n)
        for mode in ['bf16', 'ste']:
            xx = x[sl].detach().clone().requires_grad_()
            a = w1[e].detach().clone().requires_grad_()
            b = w2[e].detach().clone().requires_grad_()
            p = probs[sl].detach().clone().requires_grad_()
            with torch.enable_grad():
                if mode == 'bf16':
                    yy = native_expert(xx, a, b, mode)
                else:
                    qx, qa, qb = ste(xx, quant_ref(xx, True)), ste(a, quant_ref(a, False)), ste(b, quant_ref(b, False))
                    gu = F.linear(qx, qa).to(xx.dtype)
                    g, u = gu.chunk(2, -1)
                    h = (F.silu(g.float())*u.float()).to(xx.dtype)
                    yy = F.linear(ste(h, quant_ref(h, True)), qb).to(xx.dtype)
                yy = (yy.float()*p[:, None]).to(xx.dtype)
                grads = torch.autograd.grad(yy, (xx, p, a, b), upstream[sl])
            prod = (got[0][sl], got[1][sl], got[2][e], got[3][e])
            for name, actual, ref in zip(['input', 'router_probability', 'fc1', 'fc2'], prod, grads):
                emit('backward', label=label, expert=module.ids[e], reference=mode, tensor=name, **stats(actual, ref))
        offset += n


def operator_probe():
    from slime.backends.megatron_utils.alignment import cutlass_nvfp4_moe_forward as C
    from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import _MoELayout
    from slime.backends.megatron_utils.megatron_to_hf.processors.nvfp4_weight_quant import quantize_matrix_nvfp4
    from slime.backends.megatron_utils.hf_to_megatron.nvfp4_dequant import dequantize_nvfp4_weight
    for layer in [0, 23, 47]:
        module = Experts(layer, [0, 17, 63, 127])
        layout = _MoELayout(4, 2048, 768)
        state = C._build_nvfp4_moe_state(module, layout, module_name='diagnostic')
        for wi, group in enumerate(module.weights()):
            for ei, w in enumerate(group):
                packed, bs, gs = quantize_matrix_nvfp4(w)
                dq = dequantize_nvfp4_weight(packed, bs, gs, dtype=torch.float32)
                emit('weight_quantizer', layer=layer, expert=module.ids[ei], weight=wi, **stats(dq, quant_ref(w, False)))
        for case, counts in [('ragged', [1, 17, 0, 129]), ('aligned', [128, 128, 128, 128])]:
            torch.manual_seed(170+layer)
            x = torch.randn(sum(counts), 2048, device='cuda', dtype=torch.bfloat16)
            p = torch.full((len(x),), .125, device='cuda', dtype=torch.float32)
            with torch.no_grad():
                out = C._expert_major_cutlass_nvfp4_moe_forward(module, x.clone(), torch.tensor(counts, device='cuda'), p,
                    state=state, layout=layout)
                for mode in ['bf16', 'oracle']:
                    refs, off = [], 0
                    for n, a, b in zip(counts, *module.weights()):
                        if n:
                            y = native_expert(x[off:off+n], a, b, mode)
                            refs.append((y.float()*p[off:off+n,None]).to(x.dtype))
                        off += n
                    emit('operator_forward', layer=layer, case=case, reference=mode, **stats(out, torch.cat(refs)))
            if case == 'ragged':
                backward_probe(module, x, counts, f'synthetic_layer{layer}')
        del module, state
    emit('OPERATOR_COMPLETE')


def rms(x, w):
    y = x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-6)
    return (y*w.float()).to(x.dtype)


def full_forward():
    from slime.backends.megatron_utils.alignment import cutlass_nvfp4_moe_forward as C
    from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import _MoELayout
    batch = torch.load(BATCH, map_location='cpu', weights_only=False)
    emit('batch_structure', type=str(type(batch)), keys=list(batch)[:30] if isinstance(batch, dict) else [])
    samples = [batch['samples'][i] for i in [0, 8]]
    seq_len = min(512, *(len(s['tokens']) for s in samples))
    ids = torch.stack([torch.as_tensor(s['tokens'][:seq_len], dtype=torch.long) for s in samples]).cuda()
    response_starts = [len(s['tokens'])-s['response_length'] for s in samples]
    emit('fixed_tokens', shape=list(ids.shape), sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
         batch=str(BATCH), sample_indices=[0,8], response_starts=response_starts,
         note='Two archived sequences, up to first 512 tokens; conditional metrics, not rollout reward')
    modes = ['bf16', 'weight_only', 'oracle', 'kernel']
    embedding = weight('model.embed_tokens.weight')
    states = {mode:F.embedding(ids, embedding) for mode in modes}
    del embedding
    bs, seq = ids.shape
    positions = torch.arange(seq, device='cuda').float()
    inv = 1./(1000000.**(torch.arange(0,128,2,device='cuda').float()/128))
    freq = positions[:,None]*inv[None,:]
    cs, sn = torch.cat([freq, freq], -1).cos()[None,None], torch.cat([freq, freq], -1).sin()[None,None]
    def rope(t):
        a,b = t.chunk(2,-1)
        return (t.float()*cs+torch.cat([-b,a],-1).float()*sn).to(t.dtype)
    for layer in range(48):
        prefix=f'model.layers.{layer}.'
        norm1, norm2 = weight(prefix+'input_layernorm.weight'), weight(prefix+'post_attention_layernorm.weight')
        q,k,v,o = [weight(prefix+'self_attn.'+s+'_proj.weight') for s in ['q','k','v','o']]
        qnorm, knorm = weight(prefix+'self_attn.q_norm.weight'), weight(prefix+'self_attn.k_norm.weight')
        gate = weight(prefix+'mlp.gate.weight')
        experts = Experts(layer, list(range(128)))
        layout = _MoELayout(128,2048,768)
        state = C._build_nvfp4_moe_state(experts,layout,module_name='full_reference')
        baseline_selected = None
        for mode in modes:
            x=states[mode]
            h=rms(x,norm1)
            qq=rms(F.linear(h,q).view(bs,seq,32,128),qnorm).transpose(1,2)
            kk=rms(F.linear(h,k).view(bs,seq,4,128),knorm).transpose(1,2)
            vv=F.linear(h,v).view(bs,seq,4,128).transpose(1,2)
            att=F.scaled_dot_product_attention(rope(qq),rope(kk).repeat_interleave(8,1),vv.repeat_interleave(8,1),is_causal=True)
            x=x+F.linear(att.transpose(1,2).reshape(bs,seq,4096),o)
            h=rms(x,norm2).reshape(-1,2048)
            router=F.linear(h,gate).float()
            probs, selected=F.softmax(router,-1).topk(8,-1)
            if mode=='bf16':
                baseline_selected=selected.clone()
            else:
                overlap=(selected[:,:,None]==baseline_selected[:,None,:]).any(-1).float().mean().item()
                emit('routing',layer=layer,mode=mode,top8_set_overlap=overlap)
            probs=probs/probs.sum(-1,keepdim=True)
            selected_flat=selected.reshape(-1)
            order=selected_flat.argsort(stable=True)
            token_rows=order//8
            counts=torch.bincount(selected_flat,minlength=128)
            packed=h[token_rows]
            packed_p=probs.reshape(-1)[order]
            if mode=='kernel':
                y=C._expert_major_cutlass_nvfp4_moe_forward(experts,packed,counts,packed_p,state=state,layout=layout)
            else:
                ys, off=[],0
                for n,a,b in zip(counts.tolist(),*experts.weights()):
                    if n:
                        yy=native_expert(packed[off:off+n],a,b,mode)
                        ys.append((yy.float()*packed_p[off:off+n,None]).to(h.dtype))
                    off+=n
                y=torch.cat(ys)
            # Same deterministic top-k accumulation order across all four modes.
            unpermuted=torch.empty_like(y)
            unpermuted[order]=y
            combined=unpermuted.reshape(-1,8,2048).float().sum(1).to(h.dtype)
            states[mode]=x+combined.reshape(bs,seq,2048)
            if mode!='bf16':
                emit('layer_forward',layer=layer,mode=mode,**stats(states[mode],states['bf16']))
            if mode=='bf16' and layer in [0,23,47]:
                # Real normalized activations; separate local Jacobian probe with
                # fixed synthetic upstream vector, not a full-model RL gradient.
                em=Experts(layer,[0])
                sample=h[(selected==0).any(-1)][:64].clone()
                if len(sample): backward_probe(em,sample,[len(sample)],f'real_activation_layer{layer}')
                del em
        emit('layer_kernel_vs_oracle',layer=layer,**stats(states['kernel'],states['oracle']))
        del experts,state
    nw=weight('model.norm.weight'); lm=weight('lm_head.weight')
    logps={}
    for mode,h in states.items():
        logits=F.linear(rms(h,nw),lm).float()
        lp=F.log_softmax(logits,-1)
        logps[mode]=lp
        entropy=-(lp.exp()*lp).sum(-1)
        next_lp=lp[:,:-1].gather(-1,ids[:,1:,None]).squeeze(-1)
        emit('full_model',mode=mode,entropy=entropy.mean().item(),nll=-next_lp.mean().item(),
             entropy_per_sequence=entropy.mean(-1).tolist())
        response_mask=torch.stack([torch.arange(seq-1,device='cuda')>=max(0,start-1) for start in response_starts])
        if response_mask.any():
            emit('response_tokens',mode=mode,count=int(response_mask.sum()),
                 entropy=entropy[:,:-1][response_mask].mean().item(),nll=-next_lp[response_mask].mean().item())
        if mode!='bf16':
            base=logps['bf16']
            emit('full_model_difference',mode=mode,kl_bf16_to_mode=(base.exp()*(base-lp)).sum(-1).mean().item(),
                 top1_agreement=(lp.argmax(-1)==base.argmax(-1)).float().mean().item())
    emit('FULL_FORWARD_COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['operator','full'],required=True)
    args=parser.parse_args()
    emit('environment',torch=torch.__version__,gpu=torch.cuda.get_device_name(),phase=args.phase,
         model=str(MODEL),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.phase=='operator': operator_probe()
    else:
        with torch.no_grad(): full_forward()
