"""Same-packed-operands GEMM checks and runtime TE capability inventory."""
import inspect
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'nvfp4_numeric_diagnostic_20260910_v1'))
import diagnose as D
import torch
import torch.nn.functional as F

D.ROOT = BASE
emit, stats = D.emit, D.stats


def inventory():
    import transformer_engine as te
    import transformer_engine.common.recipe as recipe
    from transformer_engine.pytorch import NVFP4Quantizer
    emit('te_version', version=te.__version__, path=te.__file__,
         recipe_signature=str(inspect.signature(recipe.NVFP4BlockScaling)),
         quantizer_signature=str(inspect.signature(NVFP4Quantizer)))
    root = Path(te.__file__).parent
    needles = ['backward_override', 'nvfp4_4over6', 'rowwise_dequantized']
    hits=[]
    for folder in [root/'common'/'recipe', root/'pytorch']:
        for path in folder.rglob('*.py'):
            lines=path.read_text().splitlines()
            for i,line in enumerate(lines):
                if any(n in line for n in needles):
                    hits.append(dict(path=str(path.relative_to(root)),line=i+1,text='\n'.join(lines[max(0,i-1):i+3])))
    (BASE/'te_runtime_features.json').write_text(json.dumps(hits,indent=2))
    emit('te_features', hits=len(hits), by_keyword={n:sum(n in h['text'] for h in hits) for n in needles})
    for kwargs in [dict(backward_override='dequantized'),dict(nvfp4_4over6='all',disable_rht=True)]:
        try:
            r=recipe.NVFP4BlockScaling(**kwargs)
            emit('recipe_acceptance',kwargs=kwargs,accepted=True,recipe=str(r))
        except (TypeError,ValueError,AssertionError) as e:
            emit('recipe_acceptance',kwargs=kwargs,accepted=False,error=str(e))


def decode(packed, scales):
    """Decode one expert's physical row-major FP4 and tcgen05 swizzled SF."""
    rows,k2=packed.shape
    k=k2*2
    sf=scales.view(torch.float8_e4m3fn).float()
    sf=sf.reshape(rows//128,k//64,32,4,4).permute(0,1,3,2,4)
    sf=sf.reshape(rows//128,k//64,128,4).permute(0,2,1,3).reshape(rows,k//16)
    p=packed.to(torch.int32)
    codes=torch.stack([p&15,p>>4],-1).flatten(-2)
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
    return lut[codes]*sf.repeat_interleave(16,-1)


def main():
    inventory()
    from slime.backends.megatron_utils.alignment import cutlass_nvfp4_moe_forward as C
    from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import _MoELayout
    from slime.backends.megatron_utils.megatron_to_hf.processors.nvfp4_weight_quant import quantize_matrix_nvfp4
    from slime.backends.megatron_utils.hf_to_megatron.nvfp4_dequant import dequantize_nvfp4_weight
    import sglang.srt.layers.quantization.fp4_utils as U
    import sgl_kernel
    original_gemm=U.nvfp4_grouped_gemm
    original_quant=U.nvfp4_quantize_pertoken
    original_silu=sgl_kernel.silu_and_mul
    context={}

    def gemm(out,a,b,sa,sb,ar,br,aso,bso,ne,alpha,host):
        original_gemm(out,a,b,sa,sb,ar,br,aso,bso,ne,alpha,host)
        ar,br,aso,bso=[t.tolist() for t in [ar,br,aso,bso]]
        refs=[]
        for e in range(ne):
            if ar[e]==ar[e+1]: continue
            aa=decode(a[ar[e]:ar[e+1]],sa[aso[e]:aso[e+1]])
            bb=decode(b[br[e]:br[e+1]],sb[bso[e]:bso[e+1]])
            refs.append(((aa@bb.T)*alpha[ar[e]:ar[e+1],None]).to(out.dtype))
        ref=torch.cat(refs)
        emit('same_bytes_gemm',**context,stage=context['gemm_stage'],**stats(out,ref))
        context['gemm_stage']+=1

    def quant(x,pad,unpad,sfo,total,k):
        result=original_quant(x,pad,unpad,sfo,total,k)
        data,sf,gs=result
        pp,uu,ss=[t.tolist() for t in [pad,unpad,sfo]]
        decoded, inputs=[],[]
        for e in range(len(pp)-1):
            n=uu[e+1]-uu[e]
            if not n:continue
            q=decode(data[pp[e]:pp[e+1]],sf[ss[e]:ss[e+1]])
            q=q*gs[pp[e]:pp[e+1],None]
            decoded.append(q[:n]);inputs.append(x[uu[e]:uu[e+1]])
        actual=torch.cat(decoded);xx=torch.cat(inputs)
        emit('activation_quant',**context,k=k,against='independent',**stats(actual,D.quant_ref(xx,True)))
        emit('activation_quant',**context,k=k,against='unquantized',**stats(actual,xx))
        return result

    def silu(x,out):
        result=original_silu(x,out)
        a,b=x.chunk(2,-1)
        ref=(F.silu(a.float())*b.float()).to(x.dtype)
        emit('silu',**context,**stats(out,ref))
        return result

    U.nvfp4_grouped_gemm=gemm
    U.nvfp4_quantize_pertoken=quant
    sgl_kernel.silu_and_mul=silu
    for layer in [0,23,47]:
        mod=D.Experts(layer,[0,17,63,127])
        layout=_MoELayout(4,2048,768)
        state=C._build_nvfp4_moe_state(mod,layout,module_name='same_bytes')
        for wi,weights in enumerate(mod.weights()):
            for ei,w in enumerate(weights):
                packed,bs,gs=quantize_matrix_nvfp4(w)
                dq=dequantize_nvfp4_weight(packed,bs,gs,dtype=torch.float32)
                from transformer_engine.pytorch import NVFP4Quantizer
                tq=NVFP4Quantizer(rowwise=True,columnwise=False,with_amax_reduction=False,amax_reduction_group=None,
                    with_rht=False,with_post_rht_amax=False,with_2d_quantization=False,stochastic_rounding=False)(w)
                emit('te_dequant',layer=layer,expert=ei,weight=wi,**stats(dq,tq.dequantize(dtype=torch.float32)))
                emit('te_quantizer_settings',layer=layer,expert=ei,weight=wi,
                     metadata_keys=list(tq.get_metadata()),quantizer=str(tq._quantizer))
        for case,counts in [('ragged',[1,17,0,129]),('aligned',[128]*4)]:
            context.clear();context.update(layer=layer,case=case,gemm_stage=1)
            torch.manual_seed(170+layer)
            x=torch.randn(sum(counts),2048,device='cuda',dtype=torch.bfloat16)
            p=torch.full((len(x),),.125,device='cuda',dtype=torch.float32)
            C._expert_major_cutlass_nvfp4_moe_forward(mod,x,torch.tensor(counts,device='cuda'),p,state=state,layout=layout)
    emit('STAGE_PROBE_COMPLETE')


if __name__=='__main__':
    with torch.no_grad(): main()
