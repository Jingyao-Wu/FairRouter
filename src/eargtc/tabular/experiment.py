"""Classical routing estimators fitted on labeled support observations."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
import sklearn
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from eargtc.targets.core import array, binary_target
PROBABILITY_INDICES = list(range(16)) + list(range(27, 43)) + list(range(54, 63))

def _uses_sources(cfg):
    return cfg['kind'] == 'ridge' and cfg.get('training_scope', 'source_target') == 'source_target'

def fit_tabular(banks, head, cfg):
    """All supervised selection/scaling sees exclusively supplied training rows."""
    if not _uses_sources(cfg) and len(banks) != 1:
        raise ValueError('target-only tabular candidate requires exactly one target training bank')
    xs, ys, records = ([], [], {})
    for bank in banks:
        mask, y = binary_target(bank['state'], head)
        xs.append(np.asarray(array(bank['tab']), dtype=float)[mask])
        ys.append(y)
        domain = bank.get('dataset', bank.get('domain', 'unknown'))
        records[domain] = array(bank['ids'])[mask].tolist()
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    active = sum((bool(len(v)) for v in ys))
    weights = np.concatenate([np.full(len(v), len(y) / (active * len(v))) if len(v) else np.empty(0) for v in ys])
    mean = np.average(x, axis=0, weights=weights) if len(x) else np.zeros(70)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)) if len(x) else np.ones(70)
    scale[scale < 1e-08] = 1.0
    scaled = np.clip((x - mean) / scale, -8, 8)
    indices = np.arange(70)
    if cfg['feature_mode'] == 'prob':
        indices = np.array(PROBABILITY_INDICES)
    elif cfg['feature_mode'] == 'top16':
        correlations = np.abs(scaled.T @ (weights * (y - np.average(y, weights=weights)))) if len(y) else np.zeros(70)
        indices = np.lexsort((np.arange(70), -correlations))[:16]
    fitted = dict(kind=cfg['kind'], head=head, config=cfg, mean=mean, scale=scale, indices=indices, sklearn_version=sklearn.__version__, training_diagnostics=dict(training_ids=records, feature_fit_ids=records, seed=42, domain_loss_mass={name: 1 / active for name, values in records.items() if values}, sample_weight_sum=float(weights.sum()), normalization='equal-domain supervised rows'))
    if len(np.unique(y)) < 2:
        fitted['constant_logit'] = float(logit((y.sum() + 0.5) / (len(y) + 1)))
    elif cfg['kind'] == 'ridge':
        estimator = LogisticRegression(C=cfg['C'], solver='lbfgs', max_iter=2000, random_state=42)
        estimator.fit(scaled[:, indices], y, sample_weight=weights)
        fitted.update(coef=estimator.coef_[0], intercept=float(estimator.intercept_[0]))
    elif cfg['kind'] == 'hist':
        estimator = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=cfg['max_depth'], l2_regularization=1.0, min_samples_leaf=10, early_stopping=False, random_state=42)
        estimator.fit(scaled[:, indices], y)
        fitted['estimator'] = estimator
    else:
        estimator = ExtraTreesClassifier(n_estimators=200, max_depth=cfg['max_depth'], min_samples_leaf=5, random_state=42, n_jobs=1)
        estimator.fit(scaled[:, indices], y)
        fitted['trees'] = [dict(children_left=t.tree_.children_left, children_right=t.tree_.children_right, feature=t.tree_.feature, threshold=t.tree_.threshold, value=t.tree_.value[:, 0, :]) for t in estimator.estimators_]
    return fitted

def predict_tabular(fitted, bank):
    x = np.clip((np.asarray(array(bank['tab']), float) - fitted['mean']) / fitted['scale'], -8, 8)
    x = x[:, fitted['indices']]
    if 'constant_logit' in fitted:
        return np.full(len(x), fitted['constant_logit'], dtype=float)
    if fitted['kind'] == 'ridge':
        return x @ fitted['coef'] + fitted['intercept']
    if fitted['kind'] == 'hist':
        if fitted['sklearn_version'] != sklearn.__version__:
            raise ValueError('HGB checkpoint sklearn version mismatch')
        return np.asarray(fitted['estimator'].decision_function(x), dtype=float)
    p = np.zeros(len(x))
    for tree in fitted['trees']:
        nodes = np.zeros(len(x), dtype=int)
        active = tree['children_left'][nodes] >= 0
        while active.any():
            rows = np.flatnonzero(active)
            current = nodes[rows]
            left = x[rows, tree['feature'][current]] <= tree['threshold'][current]
            nodes[rows] = np.where(left, tree['children_left'][current], tree['children_right'][current])
            active = tree['children_left'][nodes] >= 0
        value = tree['value'][nodes]
        p += value[:, 1] / value.sum(axis=1)
    return logit(np.clip(p / len(fitted['trees']), 1e-07, 1 - 1e-07))
