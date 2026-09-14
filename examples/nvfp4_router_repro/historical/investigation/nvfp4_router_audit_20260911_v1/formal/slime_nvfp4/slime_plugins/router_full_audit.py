"""Observe router tensor connectivity, actual hooks, both buffers, updates."""
import json
import os
from pathlib import Path
import torch
from slime_plugins import dq_full_audit as base

INSTALLED=False


def before_log_prob(args,model,store_prefix):
    base.before_log_prob(args,model,store_prefix)


def before_train(args,rollout_id,step_id,model,optimizer,scheduler):
    global INSTALLED
    variant=os.environ['ROUTER_VARIANT']
    os.environ['DQ_VARIANT']='original'
    if variant=='fixed':
        from slime_plugins.router_candidate import install
        install()
    elif variant!='original': raise ValueError(variant)
    base.before_train(args,rollout_id,step_id,model,optimizer,scheduler)
    if INSTALLED: return
    INSTALLED=True
    observations={};params={}

    def stats(t):
        if t is None: return None
        f=t.detach().float()
        return dict(norm=f.norm().item(),finite=bool(torch.isfinite(f).all()))

    def watch(t,key,kind):
        entry=observations[key]
        events=entry.setdefault(kind,{'requires_grad_true':0,'requires_grad_false':0,'hook_count':0,'hook_nonzero':0,'max_grad_norm':0.})
        events['requires_grad_true' if t.requires_grad else 'requires_grad_false']+=1
        if t.requires_grad:
            def hook(g):
                value=stats(g)['norm'];events['hook_count']+=1;events['hook_nonzero']+=int(value>0)
                events['max_grad_norm']=max(events['max_grad_norm'],value)
                return g
            t.register_hook(hook)

    for ci,chunk in enumerate(model):
        for name,m in chunk.named_modules():
            if not name.endswith('mlp.router'): continue
            key=f'{ci}.{name}';observations[key]={};params[key]=m.weight
            watch(m.weight,key,'weight')
            gating,routing=m.gating,m.routing
            def gate(*a,_fn=gating,_key=key,**kw):
                out=_fn(*a,**kw);watch(out,_key,'logits');return out
            def route(*a,_fn=routing,_key=key,**kw):
                out=_fn(*a,**kw);watch(out[0],_key,'probabilities');return out
            m.gating=gate;m.routing=route
    step=optimizer.step

    def audited_step(*a,**kw):
        before={k:p.detach().clone() for k,p in params.items()}
        for k,p in params.items():
            observations[k]['pre_step']=dict(main_grad=stats(getattr(p,'main_grad',None)),grad=stats(p.grad),dtype=str(p.dtype))
        result=step(*a,**kw)
        for k,p in params.items():
            observations[k]['parameter_update']=stats(p.detach().float()-before[k].float())
        rank=torch.distributed.get_rank();out=Path(os.environ['AUDIT_OUT'])
        (out/f'router_rank{rank}.json').write_text(json.dumps(dict(variant=variant,rank=rank,routers=observations),indent=2))
        print('ROUTER_AUDIT_COMPLETE',json.dumps(dict(variant=variant,rank=rank,routers=len(observations))),flush=True)
        return result
    optimizer.step=audited_step
