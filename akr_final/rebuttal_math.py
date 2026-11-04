"""Support-only bounded gain calibration on the real inference arithmetic.

GainMap never accepts a query label. Coordinate search is deliberately derivative-
free: rounding through the final BF16 addition is not treated as differentiable.
This is a new optional algorithm, not a retrospective change to paper results.
"""
from __future__ import annotations
import numpy as np

class GainMap:
    def __init__(self,router):
        self.router=router;obj=getattr(router,'implementation',router)
        self.mean=np.asarray(obj.gradient_mean,dtype=float)
        self.basis=np.asarray(obj.basis,dtype=float);self.size=1+len(self.basis)
    def predict(self,features,gains):
        a=np.asarray(gains,dtype=float)
        if a.shape!=(self.size,) or not np.isfinite(a).all():raise ValueError('finite mean + coordinate gains required')
        pred=np.asarray(self.router.predict(features),dtype=float)
        if np.all(a==1):return pred
        if np.all(a==0):return np.zeros_like(pred)
        return a[0]*self.mean+(((pred-self.mean)@self.basis.T)*a[1:])@self.basis


def fit_gains(size,evaluate,*,steps=(.5,.25),regularization=.01,upper=2.,
              min_improvement=1e-6,guard_tolerance=0.,scalar=False):
    """evaluate(gains_or_None, partition) gives per-recording answer CE.

    None is native. Fit selects one candidate; guard may veto but never select
    a different candidate. Input/model errors propagate, not native fallbacks.
    """
    if isinstance(size,bool) or not isinstance(size,int) or size<1:raise ValueError('positive coordinate count required')
    if (not steps or any(not np.isfinite(s) or s<=0 for s in steps) or not np.isfinite(upper) or upper<1
        or any(not np.isfinite(s) or s<0 for s in [regularization,min_improvement,guard_tolerance])):
        raise ValueError('invalid finite calibration settings')
    def losses(a,part):
        v=np.asarray(evaluate(a,part),dtype=float)
        if v.ndim!=1 or not len(v) or not np.isfinite(v).all():raise ValueError('finite nonempty per-recording losses required')
        return v
    native=losses(None,'calibration');trace=[];cache={}
    def objective(a):
        key=tuple(float(x) for x in a)
        if key not in cache:
            v=losses(a,'calibration')
            if len(v)!=len(native):raise ValueError('calibration population changed')
            penalty=regularization*float(np.square(a-1).mean())
            record={'gains':list(key),'mean_ce':float(v.mean()),'penalty':penalty,'objective':float(v.mean())+penalty}
            trace.append(record);cache[key]=record
        return cache[key]
    current=np.ones(size);best=objective(current)
    for step in steps:
        for coordinate in range(1 if scalar else size):
            selected=current;selected_record=best
            for sign in [-1.,1.]:
                trial=current.copy()
                if scalar:trial[:]=np.clip(current[0]+sign*step,0,upper)
                else:trial[coordinate]=np.clip(current[coordinate]+sign*step,0,upper)
                record=objective(trial)
                if record['objective']<selected_record['objective']-min_improvement:selected,selected_record=trial,record
            current,best=selected,selected_record
    result={'optimizer':'bounded_actual_path_coordinate_search','scalar_control':scalar,
            'candidate_count':len(trace),'trace':trace,'fit_selected_gains':current.tolist(),
            'gains':[0.]*size,'native_calibration_mean_ce':float(native.mean()),
            'calibration_mean_ce':best['mean_ce'],'accepted':False,'query_labels_used':False,
            'guard_candidate_comparisons':0}
    if best['mean_ce']>=float(native.mean())-min_improvement:
        result['reason']='no_calibration_improvement';return result
    gn=losses(None,'guard');gc=losses(current,'guard')
    if len(gn)!=len(gc):raise ValueError('guard population changed')
    result.update(native_guard_mean_ce=float(gn.mean()),guard_mean_ce=float(gc.mean()),guard_candidate_comparisons=1)
    if float(gc.mean())>float(gn.mean())+guard_tolerance:
        result['reason']='guard_veto';return result
    result.update(accepted=True,gains=current.tolist(),reason='accepted');return result
