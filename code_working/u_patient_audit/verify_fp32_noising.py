"""CPU validation of saved FP32 scheduler noising with cancellation-aware bounds."""
import math
import numpy as np
import torch

def verify_fp32_noising(actual,z,epsilon,alphas,timestep=140):
    """Require exact CPU FP32 replay and an operand-scaled bound to ideal FP64.
    
    Let a,b be ideal sqrt coefficients, ah,bh the actual CPU FP32 coefficients,
    u=2^-24 and gamma2=2u/(1-2u). Each term undergoes a product and final addition.
    |y-(az+b epsilon)| <= |ah-a||z|+|bh-b||epsilon|
                         + gamma2*(|ah*z|+|bh*epsilon|).
    A small explicit FP64 evaluation allowance is added. CPU equality is an
    empirical check on these saved tensors, not a universal CPU/CUDA guarantee.
    """
    for value in (actual,z,epsilon):
        assert value.device.type=='cpu' and value.dtype==torch.float32
        assert value.shape==z.shape and bool(torch.isfinite(value).all())
    aa=alphas.detach().cpu().to(torch.float32)[torch.tensor([timestep],dtype=torch.long)]
    a32=aa**.5;b32=(1-aa)**.5
    while a32.ndim<z.ndim:a32=a32.unsqueeze(-1);b32=b32.unsqueeze(-1)
    # Literal installed DDPMScheduler.add_noise arithmetic: FP32 pow/products/add.
    cpu=a32*z+b32*epsilon
    assert torch.equal(actual,cpu),'Saved noising differs from literal CPU FP32 scheduler'
    av=float(aa.item());a=math.sqrt(av);b=math.sqrt(1-av)
    ah=float(a32.reshape(-1)[0]);bh=float(b32.reshape(-1)[0])
    zd=z.double().numpy();ed=epsilon.double().numpy();yd=actual.double().numpy()
    ideal=a*zd+b*ed;error=np.abs(yd-ideal)
    operands=np.abs(ah*zd)+np.abs(bh*ed)
    coeff=abs(ah-a)*np.abs(zd)+abs(bh-b)*np.abs(ed)
    unit=2.**-24;gamma2=2*unit/(1-2*unit)
    round_bound=coeff+gamma2*operands+8*np.finfo(np.float64).eps*operands
    assert np.isfinite(round_bound).all() and np.all(error<=round_bound),'Outside coefficient/product/sum rounding bound'
    ratio=np.divide(error,round_bound,out=np.zeros_like(error),where=round_bound>0)
    old_tolerance=1e-7+2e-6*np.abs(ideal)
    failed=error>old_tolerance
    first=None
    if failed.any():
        loc=tuple(np.argwhere(failed)[0])
        first=dict(index=list(map(int,loc)),z=float(zd[loc]),epsilon=float(ed[loc]),
            actual_FP32=float(yd[loc]),CPU_FP32=float(cpu[loc]),ideal_FP64=float(ideal[loc]),
            absolute_error=float(error[loc]),old_tolerance=float(old_tolerance[loc]),
            absolute_operand_sum=float(operands[loc]),coefficient_error_bound=float(coeff[loc]),
            total_rounding_bound=float(round_bound[loc]))
    return dict(CPU_FP32_exact=True,elements=actual.numel(),old_tolerance_failed_elements=int(failed.sum()),
        max_absolute_error_to_FP64=float(error.max()),max_error_to_rounding_bound_ratio=float(ratio.max()),
        coefficients=dict(alpha=av,a_FP32=ah,b_FP32=bh,a_FP64=a,b_FP64=b),
        first_old_failure=first)

def self_test():
    alphas=torch.zeros(1000,dtype=torch.float32);alphas[140]=.8417623043060303
    z=torch.tensor([[[[-1.1494140625]]]],dtype=torch.float32)
    eps=torch.tensor([[[[2.627894878387451]]]],dtype=torch.float32)
    a=alphas[140]**.5;b=(1-alphas[140])**.5;y=a*z+b*eps
    r=verify_fp32_noising(y,z,eps,alphas)
    assert r['CPU_FP32_exact'] and r['old_tolerance_failed_elements']==1
    try:verify_fp32_noising(y+1e-5,z,eps,alphas)
    except AssertionError:pass
    else:raise AssertionError('Changed noising must fail')
    return r

