"""Synthetic CPU regression tests, never bioacoustic benchmark evidence."""
import numpy as np
import pytest
import torch
from torch import nn
from akr_closing.backend import ProjectionIntervention


def inject(d,alpha=.01,dtype=torch.float32,**kw):
    m=nn.Identity(); x=torch.ones(1,4,3,dtype=dtype); mask=torch.tensor([[0,1,1,0]],dtype=torch.bool)
    with ProjectionIntervention({(0,'k'):m},{(0,'k'):d},mask,alpha,**kw) as h:
        y=m(x);again=m(x);h.assert_applied()
    return x,y,again,h


def test_bfloat16_actual_update_is_not_the_planned_update():
    x,y,_,h=inject(np.ones(3),alpha=.001,dtype=torch.bfloat16)
    n=h.norms[str((0,'k'))]
    assert n['delta_frobenius']>0 and n['effective_delta_frobenius']==0
    assert n['unchanged_fraction']==1 and torch.equal(x,y)


def test_cap_does_not_amplify_small_direction():
    x,y,_,h=inject(np.ones(3)*1e-5,scale_mode='relative_cap')
    _,old,_,_=inject(np.ones(3)*1e-5)
    assert 0<float((y-x).norm())<1e-4
    assert float((old-x).norm())>100*float((y-x).norm())


def test_default_matches_historical_arithmetic():
    d=np.array([.2,-.7,.5],dtype=np.float32);x,y,again,h=inject(d)
    delta=torch.zeros_like(x);delta[:,1:3]=torch.tensor(d)
    expected=x+delta*(.01*x[:,1:3].norm()/delta[:,1:3].norm())
    assert torch.equal(expected,y) and torch.equal(x,again)
    assert torch.equal(x[:,[0,3]],y[:,[0,3]])


@pytest.mark.parametrize('mode',['relative_norm','relative_cap','raw'])
def test_zero_noop(mode):
    x,y,_,h=inject(np.zeros(3),scale_mode=mode,raw_eta=300.)
    assert torch.equal(x,y) and h.norms[str((0,'k'))]['effective_delta_frobenius']==0


def test_cap_on_large_and_raw_scale():
    _,_,_,h=inject(np.ones(3)*100,scale_mode='relative_cap',raw_eta=300.)
    assert h.ratios[str((0,'k'))]==pytest.approx(.01,rel=1e-5)
    _,_,_,h=inject(np.ones(3),scale_mode='raw',raw_eta=.2)
    assert h.ratios[str((0,'k'))]==pytest.approx(.2)


@pytest.mark.parametrize('kw',[{'scale_mode':'bad'},{'raw_eta':-1},{'raw_eta':float('nan')}])
def test_invalid_modes_and_scales(kw):
    with pytest.raises(ValueError):inject(np.ones(3),**kw)


def test_unapplied_hook_is_not_reported_as_success():
    m=nn.Identity();mask=torch.ones(1,4,dtype=torch.bool)
    with ProjectionIntervention({(0,'k'):m},{(0,'k'):np.ones(3)},mask,.01) as h:
        m(torch.ones(1,1,3))
        with pytest.raises(RuntimeError,match='prefill'):h.assert_applied()
    assert not m._forward_hooks


def test_gain_map_identity_zero_and_dimensions():
    from akr_final.rebuttal_math import GainMap
    from akr_closing.repair import RepositoryGradientRouter
    rng=np.random.default_rng(1);x=rng.normal(size=(12,4));g=rng.normal(size=(12,9))
    r=RepositoryGradientRouter.fit(x,g,rank=4,alpha=10.);gm=GainMap(r)
    np.testing.assert_allclose(gm.predict(x,np.ones(5)),r.predict(x))
    assert np.count_nonzero(gm.predict(x,np.zeros(5)))==0
    with pytest.raises(ValueError):gm.predict(x,np.ones(4))


def test_gain_guard_veto_no_reselection():
    from akr_final.rebuttal_math import fit_gains
    calls=[]
    def loss(a,part):
        calls.append((part,a))
        return np.array([2.]) if a is None else np.array([5. if part=='guard' else 1.+np.square(a-.5).mean()])
    r=fit_gains(3,loss,steps=[.5,.25],regularization=0)
    assert not r['accepted'] and r['reason']=='guard_veto'
    assert r['gains']==[0.]*3
    assert len([a for p,a in calls if p=='guard' and a is not None])==1


def test_gain_scalar_and_nonfinite():
    from akr_final.rebuttal_math import fit_gains
    def loss(a,part):return np.array([3.]) if a is None else np.array([1.+np.square(a-.5).mean()])
    r=fit_gains(5,loss,steps=[.5,.25],scalar=True,regularization=0)
    assert r['accepted'] and len(set(r['gains']))==1
    with pytest.raises(ValueError,match='finite'):fit_gains(2,lambda *a:np.array([np.nan]))


def toy_backend():
    from types import SimpleNamespace
    from akr_closing.backend import OmniBackend
    b=OmniBackend.__new__(OmniBackend);b.torch=torch;b.device='cpu';b.max_context_tokens=32
    b.audio_token_id=9;b.layer_ids=[0];b.layout=[((0,'k'),3)]
    class Toy(nn.Module):
        def __init__(self):super().__init__();self.proj=nn.Identity();self.seen=None
        def forward(self,**kw):
            self.seen=(kw,torch.is_grad_enabled())
            z=self.proj(torch.ones(1,kw['input_ids'].shape[1],3,dtype=torch.bfloat16))
            return SimpleNamespace(loss=z.float().mean())
    b.thinker=Toy();b.modules={(0,'k'):b.thinker.proj}
    b.processor=SimpleNamespace(tokenizer=lambda *a,**kw:{'input_ids':torch.tensor([[3,4]])})
    b._prepare=lambda *a,**k:{'input_ids':torch.tensor([[1,9,9,2]]),'attention_mask':torch.ones(1,4,dtype=torch.long)}
    return b


def test_support_answer_loss_is_answer_only_and_forward_only():
    from akr_closing.core import Query
    b=toy_backend();out=b.support_answer_loss(Query('cal','unused','g'),'target','p',direction=np.ones(3),alpha=.01)
    kw,grad=b.thinker.seen
    assert not grad and kw['labels'].tolist()==[[-100,-100,-100,-100,3,4]]
    assert out['answer_tokens']==2 and out['effective_relative_norms']
    assert not b.thinker.proj._forward_hooks


def test_support_answer_loss_context_overflow_fails_before_model():
    from akr_closing.backend import ContextLimit
    from akr_closing.core import Query
    b=toy_backend();b.max_context_tokens=5
    with pytest.raises(ContextLimit):b.support_answer_loss(Query('cal','unused','g'),'target','p')
    assert b.thinker.seen is None


def test_legacy_hook_actual_update_telemetry():
    from types import SimpleNamespace
    from animal_omni.kv_hooks import KVDeltaHooks
    module=nn.Identity();attn=SimpleNamespace(k_proj=module,v_proj=nn.Identity())
    thinker=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]))
    with KVDeltaHooks(thinker,{(0,'k'):torch.ones(1,4,3)*.001}) as h:
        out=module(torch.ones(1,4,3,dtype=torch.bfloat16))
    assert h.effective_relative_norms[(0,'k')]==0 and h.unchanged_fractions[(0,'k')]==1
    assert torch.equal(out,torch.ones_like(out))


def test_calibration_without_improvement_does_not_query_guard():
    from akr_final.rebuttal_math import fit_gains
    parts=[]
    def evaluate(a,part):
        parts.append(part);return np.array([1.]) if a is None else np.array([2.])
    r=fit_gains(3,evaluate,steps=[.5])
    assert r['reason']=='no_calibration_improvement' and r['gains']==[0.]*3
    assert 'guard' not in parts
