"""Fixed-forward check and independent autograd for saved-operand backward."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'nvfp4_numeric_diagnostic_20260910_v1'))
import diagnose as d
import dq_backward as q
import torch
import torch.nn.functional as F
from slime.backends.megatron_utils.alignment import cutlass_nvfp4_moe_forward as c
from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import _MoELayout

d.ROOT=Path(__file__).resolve().parent
torch.manual_seed(777)
for layer in [0,23,47]:
    mod=d.Experts(layer,[0,17,63,127])
    weights=tuple(v for group in mod.weights() for v in group)
    for w in weights: w.requires_grad_(True)
    layout=_MoELayout(4,2048,768)
    for counts in [[1,17,0,129],[128]*4]:
        x=torch.randn(sum(counts),2048,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        probs=torch.linspace(.05,.25,len(x),device='cuda',requires_grad=True)
        g=torch.randn_like(x)*.01
        t=torch.tensor(counts,device='cuda')
        original=c._CutlassNvfp4MoEWithBF16Backward.apply(x,t,probs,mod,layout,'original',*weights)
        with torch.no_grad(): refout,cap=q.run_captured(mod,x,t,probs,layout)
        candidate=q.Nvfp4WithDequantizedBackward.apply(x,t,probs,mod,layout,'candidate',*weights)
        assert torch.equal(original,candidate) and torch.equal(original,refout)
        basegrads=torch.autograd.grad(original,(x,probs,*weights),g)
        got=torch.autograd.grad(candidate,(x,probs,*weights),g)
        for name,a,b in zip(['input','router_probability']+[f'weight{i}' for i in range(8)],got,basegrads):
            d.emit('dq_vs_original',layer=layer,counts=counts,tensor=name,**d.stats(a,b))
        off=0
        for e,n in enumerate(counts):
            if not n: continue
            xq,hq,w1,w2,gu,down=q.expert_operands(cap,e,off,n)
            xx=x[off:off+n].detach().clone().requires_grad_()
            pp=probs[off:off+n].detach().clone().requires_grad_()
            a=weights[e].detach().clone().requires_grad_()
            b=weights[4+e].detach().clone().requires_grad_()
            # Value substitution preserves actual captured fprop values and
            # defines identity quantizer/cast derivatives independently.
            sx=xx+(xq-xx).detach();sa=a+(w1-a).detach();sb=b+(w2-b).detach()
            zz=F.linear(sx,sa);zz=zz+(gu-zz).detach()
            gate,up=zz.float().chunk(2,-1)
            h=(F.silu(gate)*up).bfloat16();h=h+(hq-h).detach()
            yy=F.linear(h,sb);yy=yy+(down-yy).detach()
            yy=(yy.float()*pp[:,None]).bfloat16()
            expected=torch.autograd.grad(yy,(xx,pp,a,b),g[off:off+n])
            actual=[got[0][off:off+n],got[1][off:off+n],got[2+e],got[6+e]]
            for name,a,b in zip(['input','router_probability','fc1','fc2'],actual,expected):
                s=d.stats(a,b);d.emit('dq_vs_autograd',layer=layer,counts=counts,expert=e,tensor=name,**s)
                assert s['finite'] and s['relative_l2']<0.001, s
            off+=n
        d.emit('forward_bitwise_equal',layer=layer,counts=counts,passed=True)
d.emit('DQ_GATE_PASS')
