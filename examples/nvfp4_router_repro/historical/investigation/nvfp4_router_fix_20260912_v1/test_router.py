"""GPU regression gate; run only inside the recorded Slurm container."""
import json
import sys
from pathlib import Path

import torch

source, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(source))
from slime.backends.megatron_utils.alignment import deepgemm_moe_forward as m

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(847)
records = []
for n in [0, 1, 17, 128, 513]:
    for scale in [None, 0., .5, 1., 2.]:
        for topk in [1, 8]:
            x = torch.randn(n, 128, device='cuda', dtype=torch.float32, requires_grad=True)
            actual, mask = m._sglang_unbiased_softmax_topk_routing(x, topk, scale)
            with torch.no_grad():
                original, original_mask = m._sglang_unbiased_softmax_topk_routing_forward(x, topk, scale)
                inference, _ = m._sglang_unbiased_softmax_topk_routing(x, topk, scale)
            assert actual.requires_grad and not mask.requires_grad
            assert not inference.requires_grad
            assert torch.equal(actual, original) and torch.equal(mask, original_mask)
            assert torch.equal(actual, inference)
            y = x.detach().clone().requires_grad_()
            values, indices = y.topk(topk, dim=-1)
            ref = torch.zeros_like(y).scatter(-1, indices, values.softmax(-1) * (1. if scale is None else scale))
            upstream = torch.randn_like(actual)
            grad = torch.autograd.grad(actual, x, upstream)[0]
            expected = torch.autograd.grad(ref, y, upstream)[0]
            relative = ((grad - expected).norm() / expected.norm().clamp_min(1e-12)).item()
            absolute = (grad - expected).abs().max().item() if n else 0.
            assert torch.isfinite(grad).all()
            assert absolute < 2e-6 and (relative < 1e-5 or absolute < 2e-7), (n, scale, topk, relative, absolute)
            if topk == 1 or scale == 0.:
                assert torch.count_nonzero(grad) == 0
            assert torch.count_nonzero(grad[~mask]) == 0
            records.append(dict(tokens=n, scale=scale, topk=topk, forward_bitwise=True,
                                gradient_relative_l2=relative, gradient_max_abs=absolute))
output.write_text(json.dumps(dict(source=str(source), cases=records), indent=2))
print('PRODUCTION_ROUTER_GATE_PASS', source, len(records), flush=True)
