"""Frozen-representation readouts and continuous/class-pooled corrections.

Support fitting is supervised. No function here accepts a query target.
The continuous route follows the existing repo's centered-SVD + ridge design.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def standardize(x):
    mean=x.mean(0); scale=x.std(0); scale=np.where(scale<1e-6,1.,scale)
    return (x-mean)/scale,mean,scale


@dataclass
class RidgeMap:
    mean: np.ndarray
    scale: np.ndarray
    support: np.ndarray
    dual: np.ndarray
    intercept: np.ndarray
    @classmethod
    def fit(cls,x,y,alpha):
        x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
        if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not len(x) or alpha<=0:
            raise ValueError('finite aligned 2-D support arrays and alpha>0 required')
        if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('nonfinite support')
        z,mean,scale=standardize(x); intercept=y.mean(0)
        dual=np.linalg.solve(z@z.T+alpha*np.eye(len(z)),y-intercept)
        return cls(mean,scale,z,dual,intercept)
    def predict(self,x):
        x=np.atleast_2d(np.asarray(x,dtype=np.float64))
        return ((x-self.mean)/self.scale)@self.support.T@self.dual+self.intercept


@dataclass
class RidgeReadout:
    mapping: RidgeMap
    @classmethod
    def fit(cls,x,y,n_class,alpha=10.):
        y=np.asarray(y,dtype=int)
        if set(y)!=set(range(n_class)): raise ValueError('every class must be in support')
        return cls(RidgeMap.fit(x,np.eye(n_class)[y],alpha))
    def predict(self,x): return self.mapping.predict(x).argmax(1)
    def probabilities(self,x,temperature=1.):
        if temperature<=0: raise ValueError('temperature must be positive')
        z=self.mapping.predict(x)/temperature; z-=z.max(1,keepdims=True)
        p=np.exp(z); return p/p.sum(1,keepdims=True)


@dataclass
class GradientRouter:
    gradient_mean: np.ndarray
    basis: np.ndarray
    mapping: RidgeMap | None
    rank: int
    @classmethod
    def fit(cls,x,g,rank=4,alpha=10.):
        x=np.asarray(x,float); g=np.asarray(g,float)
        if x.ndim!=2 or g.ndim!=2 or len(x)!=len(g) or not len(x): raise ValueError('support shapes')
        mean=g.mean(0); centered=g-mean
        # Gram eigensolver avoids a wide p x p covariance or full wide SVD workspace.
        gram=centered@centered.T; vals,vec=np.linalg.eigh(gram); order=np.argsort(vals)[::-1]
        vals,vec=vals[order],vec[:,order]
        tol=max(float(vals[0]) if len(vals) else 0,1e-30)*1e-10
        rank=min(max(int(rank),0),len(x)-1,int((vals>tol).sum()))
        if rank:
            basis=(vec[:,:rank].T@centered)/np.sqrt(vals[:rank])[:,None]
            mapping=RidgeMap.fit(x,centered@basis.T,alpha)
        else: basis=np.empty((0,g.shape[1])); mapping=None
        return cls(mean,basis,mapping,rank)
    def predict(self,x):
        n=len(np.atleast_2d(x))
        if not self.rank: return np.repeat(self.gradient_mean[None,:],n,0)
        return self.gradient_mean+self.mapping.predict(x)@self.basis


def matched_random(field,seed):
    field=np.asarray(field); rng=np.random.default_rng(seed); out=rng.standard_normal(field.shape)
    return out*(np.linalg.norm(field)/max(np.linalg.norm(out),1e-30))


def geometry(g,y):
    g=np.asarray(g,dtype=float); y=np.asarray(y); classes=np.unique(y)
    mu={c:g[y==c].mean(0) for c in classes}; fit=np.stack([mu[c] for c in y])
    denom=max(float(np.square(g).sum()),1e-30)
    globalmean=g.mean(0); total=float(np.square(g-globalmean).sum())
    within=float(np.square(g-fit).sum()); between=float(np.square(fit-globalmean).sum())
    loo_pred=[]; loo_error=[]
    for i in range(len(g)):
        mask=np.arange(len(g))!=i
        if sum(mask&(y==y[i]))==0: continue
        centroids=np.stack([g[mask&(y==c)].mean(0) if np.any(mask&(y==c)) else np.zeros(g.shape[1]) for c in classes])
        scores=centroids@g[i]/(np.maximum(np.linalg.norm(centroids,axis=1),1e-30)*max(np.linalg.norm(g[i]),1e-30))
        loo_pred.append(classes[scores.argmax()]==y[i]); loo_error.append(float(np.square(g[i]-g[mask&(y==y[i])].mean(0)).sum()))
    vals=np.maximum(np.linalg.eigvalsh((g-globalmean)@(g-globalmean).T),0); p=vals/max(vals.sum(),1e-30)
    positive=p[p>1e-12]
    return {'n_support':len(g),'n_classes':len(classes),
            'in_sample_centroid_energy_fraction':float(np.square(fit).sum()/denom),
            'in_sample_relative_squared_error':within/denom,
            'loo_relative_squared_error':sum(loo_error)/denom if loo_error else None,
            'centered_class_variance_fraction':between/max(total,1e-30),
            'between_within_ratio':between/max(within,1e-30),
            'effective_rank':float(np.exp(-(positive*np.log(positive)).sum())) if len(positive) else 0.,
            'loo_label_accuracy':float(np.mean(loo_pred)) if loo_pred else None,
            'interpretation':'Targets were used to compute gradients; label clustering is not independent evidence of acoustic grounding.'}

@dataclass
class RepositoryGradientRouter:
    """Production adapter; the primary fit is the existing repo implementation.

    There is deliberately no import-error fallback to class routing or another
    solver. CPU math tests use the explicitly named reference implementation.
    """
    implementation: object
    rank: int
    @classmethod
    def fit(cls,x,g,rank=4,alpha=10.):
        from animal_omni.conditional_kv import ConditionalGradientRouter
        model=ConditionalGradientRouter.fit(np.asarray(x,dtype=np.float32),
             np.asarray(g,dtype=np.float32),rank=rank,alpha=alpha)
        return cls(model,model.rank)
    def predict(self,x):
        return np.stack([self.implementation.predict(row) for row in np.atleast_2d(x)])
