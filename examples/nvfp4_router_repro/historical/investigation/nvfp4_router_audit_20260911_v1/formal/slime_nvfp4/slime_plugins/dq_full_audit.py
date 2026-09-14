"""Paired fixed-batch audit, delegating unchanged recompute/norm gate."""
import json
import os
from pathlib import Path
import torch
from slime_plugins import nvfp4_recompute_experiment as original

INSTALLED=False


def before_log_prob(args, model, store_prefix):
    original.before_log_prob(args, model, store_prefix)


def before_train(args, rollout_id, step_id, model, optimizer, scheduler):
    global INSTALLED
    original.before_train(args, rollout_id, step_id, model, optimizer, scheduler)
    if INSTALLED: return
    INSTALLED=True
    variant=os.environ['DQ_VARIANT']
    if variant=='dequantized':
        from slime_plugins.dq_backward import install
        install()
    elif variant!='original': raise ValueError(variant)
    parameters={}
    for ci,chunk in enumerate(model):
        for name,p in chunk.named_parameters():
            if any(s in name for s in ['input_layernorm.weight','router.weight','linear_qkv.weight','linear_fc1.weight0','linear_fc2.weight0']):
                parameters[f'{ci}.{name}']=p
    original_step=optimizer.step

    def step(*a,**kw):
        rank=torch.distributed.get_rank()
        rows={};samples={};before={}
        for name,p in parameters.items():
            g=getattr(p,'main_grad',None)
            if g is None: g=p.grad
            before[name]=p.detach().reshape(-1)[:8192].clone()
            rows[name]={'missing':g is None}
            if g is not None:
                f=g.detach().float().reshape(-1)
                rows[name].update(norm=f.norm().item(),finite=bool(torch.isfinite(f).all()),numel=f.numel())
                samples[name]=f[:8192].cpu()
        result=original_step(*a,**kw)
        for name,p in parameters.items():
            delta=p.detach().reshape(-1)[:8192].float()-before[name].float()
            rows[name]['parameter_prefix_update_l2']=delta.norm().item()
        out=Path(os.environ['AUDIT_OUT']);out.mkdir(parents=True,exist_ok=True)
        report=dict(variant=variant,rank=rank,rollout_id=rollout_id,step_id=step_id,gradients=rows,
                    optimizer_result=str(result),sample_elements=8192)
        (out/f'rank{rank}.json').write_text(json.dumps(report,indent=2))
        torch.save(samples,out/f'rank{rank}_grad_prefix.pt')
        print('DQ_FULL_AUDIT_COMPLETE',json.dumps(dict(variant=variant,rank=rank,parameters=len(rows))),flush=True)
        return result
    optimizer.step=step
