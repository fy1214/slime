"""Diagnostic-only differentiable wrapper around unchanged SGLang top-k.

Selected top-k indices are locally constant. Normalized selected softmax has
Jacobian diag(p)-p p^T (accounting for an optional output scale).
"""
import torch


def install():
    from slime.backends.megatron_utils.alignment import deepgemm_moe_forward as m
    if getattr(m,'_router_backward_diagnostic_installed',False): return
    original=m._sglang_unbiased_softmax_topk_routing

    class Routing(torch.autograd.Function):
        @staticmethod
        def forward(ctx,logits,topk,scaling_factor):
            probs,mask=original(logits,topk,scaling_factor)
            ctx.scale=1.0 if scaling_factor is None else float(scaling_factor)
            ctx.save_for_backward(probs)
            ctx.mark_non_differentiable(mask)
            return probs,mask

        @staticmethod
        def backward(ctx,grad_probs,grad_mask):
            (p,)=ctx.saved_tensors
            if grad_probs is None: return None,None,None
            if ctx.scale==0: return torch.zeros_like(p),None,None
            pf,g=p.float(),grad_probs.float()
            dx=pf*(g-(g*pf).sum(-1,keepdim=True)/ctx.scale)
            return dx.to(p.dtype),None,None

    m._sglang_unbiased_softmax_topk_routing=lambda logits,topk,scaling_factor=None: Routing.apply(logits,topk,scaling_factor)
    m._router_backward_diagnostic_installed=True
