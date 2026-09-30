"""Routing feature transforms and labeled-observation validation."""
import numpy as np

def guard_ids(dataset, ids, split, metadata):
    ids = list(ids)
    m = metadata[dataset]
    if split != 'train' or len(set(ids)) != len(ids):
        raise ValueError('fit requires unique explicitly identified train nodes')
    if not set(ids) <= set(m['gold_ids']) or set(ids) & (set(m['validation_ids']) | set(m['query_ids'])):
        raise ValueError('validation/query/test nodes cannot enter fitting')

def exact_mask(score, ids, k):
    score = np.asarray(score, float)
    ids = np.asarray(ids)
    if score.ndim != 1 or score.shape != ids.shape or (not np.isfinite(score).all()) or (len(np.unique(ids)) != len(ids)):
        raise ValueError('invalid ranking arrays')
    if int(k) != k or not 0 <= k <= len(ids):
        raise ValueError('invalid K')
    mask = np.zeros(len(ids), bool)
    mask[np.lexsort((ids, -score))[:int(k)]] = True
    return mask

def precision_profile(score, correct, ids, half, fraction):
    score = np.asarray(score, float)
    correct = np.asarray(correct, bool)
    ids = np.asarray(ids)
    half = np.asarray(half, bool)
    fractions = sorted(set((min(1.0, x) for x in (0.5 * fraction, fraction, 2 * fraction))))

    def metrics(mask):
        s = score[mask]
        y = correct[mask]
        i = ids[mask]
        order = np.lexsort((i, -s))
        values = []
        counts = []
        for f in fractions:
            k = int(np.ceil(len(s) * f))
            if k >= 10:
                counts.append(k)
                values.append(float(y[order[:k]].mean()))
        return dict(mean=float(np.mean(values)) if values else float(y.mean()) if len(y) else 0.0, counts=counts, precision=values)
    full = metrics(np.ones(len(ids), bool))
    halves = [metrics(half), metrics(~half)]
    return dict(full=full, halves=halves, robust=0.5 * full['mean'] + 0.5 * min((m['mean'] for m in halves)), fractions=fractions)
