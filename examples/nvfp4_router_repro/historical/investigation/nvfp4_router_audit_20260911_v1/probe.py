import inspect
import json
from pathlib import Path
import torch
from slime.backends.megatron_utils.alignment import deepgemm_moe_forward as m
from router_candidate import install

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(847)
root=Path(__file__).resolve().parent
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate
(root/'moe_fused_gate_runtime.py').write_text(inspect.getsource(moe_fused_gate))
records=[]
old=m._sglang_unbiased_softmax_topk_routing
install()
for n in [1,17,128,513]:
 for scale in [None,1.,.5,2.]:
    x=torch.randn(n,128,device='cuda',dtype=torch.float32,requires_grad=True)
    p,mask=old(x,8,scale)
    fixed,mask2=m._sglang_unbiased_softmax_topk_routing(x,8,scale)
    assert torch.equal(p,fixed) and torch.equal(mask,mask2)
    y=x.detach().clone().requires_grad_()
    v,indices=y.topk(8,-1)
    ref=torch.zeros_like(y).scatter(-1,indices,torch.softmax(v,-1)*(1. if scale is None else scale))
    g=torch.randn_like(p)
    a=torch.autograd.grad(fixed,x,g)[0]
    b=torch.autograd.grad(ref,y,g)[0]
    err=((a-b).norm()/b.norm()).item()
    row=dict(n=n,scale=scale,original_requires_grad=p.requires_grad,original_grad_fn=str(p.grad_fn),
             fixed_requires_grad=fixed.requires_grad,forward_bitwise_equal=True,grad_relative_l2=err,
             grad_norm=a.norm().item(),grad_sum_max=a.sum(-1).abs().max().item())
    print(json.dumps(row),flush=True);records.append(row)
    assert not p.requires_grad and fixed.requires_grad and err<1e-5
(root/'gate.json').write_text(json.dumps(records,indent=2))
print('ROUTER_BACKWARD_GATE_PASS',flush=True)
