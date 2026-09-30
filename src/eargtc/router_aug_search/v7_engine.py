"""Fit routing estimators on equally weighted augmented support views."""
from copy import deepcopy
import numpy as np
from eargtc.router_v8_v7_explore import core_v2 as historical
DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
HEADS = ('agreement', 'trust', 'preference')

def candidate_grid(dataset: str, head: str, stage='original') -> list[dict]:
    if dataset not in DATASETS or head not in HEADS:
        raise ValueError((dataset, head))
    if stage != 'original':
        raise ValueError(f'Unknown training stage: {stage}')
    return deepcopy(historical.candidate_grid())

def _check_bank(bank, dataset):
    if bank.get('split') != 'train':
        raise ValueError('fit_candidate accepts training banks only (split=train)')
    if bank.get('dataset') != dataset:
        raise ValueError('Training dataset key mismatch')
    ids = np.asarray(bank['ids'])
    parents = np.asarray(bank['node_ids'])
    if ids.ndim != 1 or parents.shape != ids.shape or len(np.unique(ids)) != len(ids):
        raise ValueError('Training sample IDs must be unique and aligned with parent IDs')
    weights = np.asarray(bank.get('weight', np.ones(len(ids))))
    if weights.shape != ids.shape or not np.all(weights == 1):
        raise ValueError('Every augmented view must have equal unit weight')
    if len(bank['tab']) != len(ids):
        raise ValueError('Training feature/sample lengths differ')

def fit_candidate(cfg, head, banks: dict, target: str) -> dict:
    if head not in HEADS or target not in DATASETS:
        raise ValueError((target, head))
    if set(banks) != set(DATASETS):
        raise ValueError('Routing requires exactly five training datasets')
    for dataset, bank in banks.items():
        _check_bank(bank, dataset)
    cfg = deepcopy(cfg)
    scope, mode = (cfg['scope'], cfg['mode'])
    ordered = [banks[d] for d in DATASETS if d != target] + [banks[target]]
    if scope not in ('source', 'target', 'pooled', 'transfer'):
        raise ValueError(f'Unknown training scope: {scope}')
    use = ordered[:-1] if scope == 'source' else ordered[-1:] if scope == 'target' else ordered
    xs, ys, ws, ds = ([], [], [], [])
    fitids, parentids, featureids, domain_masses = ({}, {}, {}, {})
    for j, bank in enumerate(use):
        labels = np.asarray(bank['target'] if head == 'agreement' else bank['state'])
        if labels.shape != (len(bank['ids']),):
            raise ValueError('Training labels must align with sample IDs')
        if not np.isin(labels, (0, 1) if head == 'agreement' else (0, 1, 2)).all():
            raise ValueError('Unexpected training target/state')
        mask = labels != 2 if head == 'preference' else np.ones(len(labels), bool)
        y = labels[mask] == 0 if head == 'preference' else labels[mask] != 2 if head == 'trust' else labels[mask].astype(bool)
        if not len(y):
            continue
        xs.append(historical.raw_features(bank, mode)[mask])
        ys.append(y)
        ds.append(np.full(len(y), j))
        dataset = bank['dataset']
        fitids[dataset] = np.asarray(bank['ids'])[mask].tolist()
        parentids[dataset] = np.asarray(bank['node_ids'])[mask].tolist()
        featureids[dataset] = np.asarray(bank['ids']).tolist()
        mass = (0.25 if dataset == target else 0.75 / 4) if scope in ('pooled', 'transfer') else 1 / len(use)
        domain_masses[dataset] = mass
        ws.append(np.full(len(y), mass / len(y)))
    if not xs:
        raise ValueError('No eligible training rows for requested head')
    x, y, w, domains = map(np.concatenate, (xs, ys, ws, ds))
    w *= len(w) / w.sum()
    transform = historical.Transform.fit(x, w, cfg['rank'])
    xx = transform.apply(x)
    cfg['target_domain'] = len(use) - 1 if scope == 'transfer' else -1
    estimator = historical.fit_estimator(xx, y, w, domains, cfg)
    train_z = historical.predict_estimator(estimator, xx)
    center = float(np.average(train_z, weights=w))
    scale = max(float(np.sqrt(np.average((train_z - center) ** 2, weights=w))), 0.1)
    audit = dict(fit_sample_ids=fitids, fit_parent_ids=parentids, transform_fit_sample_ids=deepcopy(fitids), transform_fit_parent_ids=deepcopy(parentids), raw_feature_sample_ids=featureids, domain_masses=domain_masses, fit_rows=len(y), validation_training_rows=0, query_training_rows=0, validation_loss_evaluations=0, view_weights='all unit; configured dataset mass retained', environment_features=False, target_domain=cfg['target_domain'], normalization_fit_sample_ids=deepcopy(fitids))
    return dict(version='Router-v8(v7)', head=head, target=target, config=cfg, transform=transform, estimator=estimator, mode=mode, center=center, scale=scale, fit_ids=fitids, fit_audit=audit)

def predict_candidate(model, bank) -> np.ndarray:
    x = historical.raw_features(bank, model['mode'])
    return np.asarray(historical.predict_estimator(model['estimator'], model['transform'].apply(x)), float).reshape(-1)

def describe_space():
    return dict(version='Router-v8(v7)', original_source='eargtc/router_v8_v7_explore/core_v2.py', original_candidates=168, stages=['original'], incumbent='fit the configured estimator on augmented support observations', fit_boundary='train only; all six views share unit weight; parent identity retained in audit')
