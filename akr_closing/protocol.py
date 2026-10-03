"""Small, testable protocol operations; no model or query-label access."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from .core import file_hash


def matched_audio(reference: Path, donor: Path | None, output: Path) -> Path:
    """Match reference duration/sample rate by anti-aliased resampling + crop/pad.

    No time stretching, loudness normalization, looping or synthetic content.
    Silence has identical sample count. Hashes are stored to reject stale reuse.
    """
    import soundfile as sf
    from scipy.signal import resample_poly
    from .core import atomic_json
    x,sr=sf.read(reference,dtype='float32',always_2d=True)
    x=x.mean(1)
    meta={'reference_sha256':file_hash(reference),
          'donor_sha256':file_hash(donor) if donor else None,
          'samples':len(x),'sample_rate':sr,'policy':'mono/resample/prefix-crop-or-tail-zero-pad; no RMS normalization'}
    sidecar=Path(str(output)+'.json')
    if output.exists():
        if not sidecar.exists() or json.loads(sidecar.read_text())!=meta:
            raise RuntimeError(f'stale null audio: {output}')
        return output
    if donor is None: y=np.zeros_like(x)
    else:
        y,ysr=sf.read(donor,dtype='float32',always_2d=True);y=y.mean(1)
        if ysr!=sr:
            div=math.gcd(int(sr),int(ysr)); y=resample_poly(y,sr//div,ysr//div)
        y=np.pad(y[:len(x)],(0,max(0,len(x)-len(y))))
    if not np.isfinite(y).all(): raise ValueError('non-finite control audio')
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.stem+'.tmp.wav'); sf.write(tmp,y,sr,subtype='FLOAT');tmp.replace(output)
    atomic_json(sidecar,meta)
    return output


def shuffle_support_labels(y, *, seed: int):
    y=np.asarray(y)
    if y.ndim!=1 or len(np.unique(y))<2: raise ValueError('need at least two support classes')
    rng=np.random.default_rng(seed)
    for _ in range(64):
        p=rng.permutation(len(y))
        if np.any(y[p]!=y):return y[p].copy()
    return np.roll(y,1)


def _cos(a,b):
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    good=den>1e-20
    return np.sum(a[good]*b[good],axis=1)/den[good]


def compare_gradient_targets(real,control,y):
    """Compare same target on different audio, including class-residual directions."""
    a=np.asarray(real,dtype=float);b=np.asarray(control,dtype=float);y=np.asarray(y)
    if a.ndim!=2 or a.shape!=b.shape or len(a)!=len(y): raise ValueError('aligned fields required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite field')
    ar=a.copy();br=b.copy()
    for c in np.unique(y):
        idx=y==c;ar[idx]-=a[idx].mean(0);br[idx]-=b[idx].mean(0)
    cos=_cos(a,b);rcos=_cos(ar,br)
    return {'n':len(y),'median_same_target_cosine':float(np.median(cos)) if len(cos) else None,
       'relative_error':float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)),
       'residual_cosine_median':float(np.median(rcos)) if len(rcos) else None,
       'n_nonzero_residual_pairs':len(rcos),
       'interpretation':'label-conditioned gradients are not_acoustic_semantics; inspect held-out repair and target-controlled residuals'}


def probe_to_text(predictions):
    return [{'prediction':p,'raw_prediction':p,'model_forward_calls':0,
             'source':'deterministic classifier-to-label string'} for p in predictions]


def require_external_gate(path: Path):
    payload=json.loads(Path(path).read_text())
    gate=payload.get('gate',payload)
    if gate.get('passed') is not True:
        raise RuntimeError('External registered gate did not pass; no test execution is allowed')
    return payload
