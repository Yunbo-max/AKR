"""Support-only, recording-grouped correction predictability diagnostics.

F1 distinguishes truncation error from feature-to-gradient regression error.
All normalizers, bases and regressors are refit INSIDE each training fold.
The true-class centroid is explicitly an oracle diagnostic, never deployment.
"""
from __future__ import annotations
import numpy as np


def fit_repository(x, g, rank, alpha):
    from akr_closing.repair import RepositoryGradientRouter
    return RepositoryGradientRouter.fit(x, g, rank=rank, alpha=alpha)


def derangement(n: int, seed: int) -> np.ndarray:
    if n < 2:
        raise ValueError('A derangement needs at least two queries.')
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    result = np.empty(n, dtype=int)
    result[order] = np.roll(order, 1)
    return result


def group_folds(groups, n_splits=5, seed=0):
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError('At least two real recording groups are required.')
    if n_splits < 2:
        raise ValueError('n_splits must be >= 2')
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    output = []
    for test_groups in np.array_split(unique, min(n_splits, len(unique))):
        test = np.flatnonzero(np.isin(groups, test_groups))
        train = np.flatnonzero(~np.isin(groups, test_groups))
        if len(train) < 2:
            raise ValueError('A grouped training fold has fewer than two examples.')
        output.append((train, test))
    return output


def error_decomposition(truth, prediction, mean, basis):
    """basis has orthonormal ROWS, as in ConditionalGradientRouter.basis."""
    truth, prediction, mean, basis = map(np.asarray, (truth, prediction, mean, basis))
    centered = truth - mean
    coordinates = centered @ basis.T
    reconstruction = mean + coordinates @ basis
    projected_prediction = (prediction - mean) @ basis.T
    return {
        'total_sse': float(np.square(truth-prediction).sum()),
        'projection_sse': float(np.square(truth-reconstruction).sum()),
        'coefficient_sse': float(np.square(coordinates-projected_prediction).sum()),
    }


def _model_parts(router):
    model = getattr(router, 'implementation', router)
    return np.asarray(model.gradient_mean), np.asarray(model.basis)


def _cosines(a, b):
    norms = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    keep = norms > 1e-20
    return ((a[keep]*b[keep]).sum(1)/norms[keep]).tolist()


def grouped_predictability(features, gradients, groups, labels, ranks, penalty,
                           n_splits=5, seed=0, *, router_fit=None):
    fit = router_fit or fit_repository
    x, g = np.asarray(features, dtype=np.float32), np.asarray(gradients, dtype=np.float32)
    groups, labels = np.asarray(groups).astype(str), np.asarray(labels)
    if (x.ndim != 2 or g.ndim != 2 or len(x) != len(g) or len(x) != len(groups)
            or len(x) != len(labels) or not np.isfinite(x).all() or not np.isfinite(g).all()):
        raise ValueError('Aligned finite support feature, gradient, group and label arrays required.')
    if penalty <= 0 or not ranks or any(int(r) != r or r < 1 for r in ranks):
        raise ValueError('Positive ranks and ridge penalty required.')
    folds = group_folds(groups, n_splits, seed)
    summaries, fold_records = [], []
    for fi, (train, test) in enumerate(folds):
        fold_records.append({'fold': fi, 'train_indices': train.tolist(), 'heldout_indices': test.tolist(),
                             'train_groups': sorted(set(groups[train])), 'heldout_groups': sorted(set(groups[test]))})
    for rank in ranks:
        fixed_sse = pred_sse = projection_sse = coef_sse = shuffled_sse = oracle_sse = 0.
        cosine, per_fold, unavailable_oracle = [], [], 0
        for fi, (train, test) in enumerate(folds):
            model = fit(x[train], g[train], rank=rank, alpha=penalty)
            prediction = model.predict(x[test])
            mean, basis = _model_parts(model)
            dec = error_decomposition(g[test].astype(float), prediction.astype(float), mean.astype(float), basis.astype(float))
            f_sse = float(np.square(g[test].astype(float)-mean).sum())
            shuffled = fit(x[train][derangement(len(train), seed+fi)], g[train], rank=rank, alpha=penalty)
            s_sse = float(np.square(g[test].astype(float)-shuffled.predict(x[test])).sum())
            oracle = []
            for label in labels[test]:
                mask = labels[train] == label
                if mask.any(): oracle.append(g[train][mask].mean(0))
                else:
                    unavailable_oracle += 1
                    oracle.append(g[train].mean(0))
            o_sse = float(np.square(g[test].astype(float)-np.stack(oracle)).sum())
            fixed_sse += f_sse; pred_sse += dec['total_sse']
            projection_sse += dec['projection_sse']; coef_sse += dec['coefficient_sse']
            shuffled_sse += s_sse; oracle_sse += o_sse
            cosine += _cosines(g[test], prediction)
            per_fold.append({'fold':fi, 'actual_rank':int(model.rank), 'n_train':len(train), 'n_heldout':len(test),
                             **dec, 'fixed_mean_sse':f_sse, 'shuffled_sse':s_sse, 'label_oracle_sse':o_sse})
        ratio = lambda value: value/fixed_sse if fixed_sse > 1e-20 else None
        summaries.append({'rank':int(rank), 'n':len(x), 'predicted_nmse':ratio(pred_sse),
                          'projection_nmse':ratio(projection_sse), 'coefficient_nmse':ratio(coef_sse),
                          'shuffled_nmse':ratio(shuffled_sse), 'label_oracle_nmse':ratio(oracle_sse),
                          'fixed_mean_nmse':1.0 if fixed_sse > 1e-20 else None,
                          'mean_cosine':float(np.mean(cosine)) if cosine else None,
                          'nonzero_cosine_n':len(cosine), 'label_oracle_fallback_n':unavailable_oracle,
                          'decomposition_residual':pred_sse-projection_sse-coef_sse, 'fold_results':per_fold})
    return {'complete':True, 'n_support':len(x), 'n_recording_groups':len(set(groups)), 'folds':fold_records,
            'rows':summaries, 'query_gradients_computed':0, 'hyperparameter_selection':False,
            'interpretation':'Held-out SUPPORT gradient prediction; not recognition accuracy. True-class centroid uses held-out support labels and is a diagnostic only.'}
