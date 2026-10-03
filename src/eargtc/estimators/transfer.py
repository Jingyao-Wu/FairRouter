"""Support-supervised routing with classical and neural estimators."""
from __future__ import annotations
import copy
from collections import OrderedDict
import hashlib
import json
DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
HEADS = ('trust', 'preference', 'agreement')
_SOURCE_CACHE = OrderedDict()
_SOURCE_CACHE_LIMIT = 32

def _identity(dataset, head):
    if dataset not in DATASETS or head not in HEADS:
        raise ValueError('Unknown dataset or head')

def _prepare(banks, target, head):
    import torch
    if target not in banks or set(banks) - set(DATASETS):
        raise ValueError('Missing target or unknown training dataset')
    clean = {}
    for dataset, bank in banks.items():
        if bank.get('dataset') != dataset or bank.get('split') != 'train':
            raise ValueError('All supplied banks must identify their dataset and train split')
        needed = ('ids', 'node_ids', 'environment_index', 'weight', 'tab', 'g', 'l', 'target' if head == 'agreement' else 'state')
        if any((k not in bank for k in needed)):
            raise ValueError('Prepared bank is missing features, targets or provenance')
        n = len(bank['ids'])
        for key in ('ids', 'node_ids', 'environment_index', 'weight'):
            if torch.as_tensor(bank[key]).shape != (n,):
                raise ValueError('Invalid observation provenance shape')
        ids = torch.as_tensor(bank['ids'])
        parents = torch.as_tensor(bank['node_ids'])
        env = torch.as_tensor(bank['environment_index'])
        if ids.dtype != torch.int64 or parents.dtype != torch.int64 or ids.unique().numel() != n:
            raise ValueError('Sample IDs must be unique int64 values; parents must be int64')
        if (ids < 0).any() or (parents < 0).any() or ((env < 0) | (env >= 6)).any():
            raise ValueError('Invalid sample, parent or six-view environment identity')
        if not torch.equal(torch.as_tensor(bank['weight']), torch.ones(n)):
            raise ValueError('Every augmented observation must have unit weight')
        if any((k in bank for k in ('gold_target', 'gold_labels', 'labels', 'query_truth', 'validation_truth'))):
            raise ValueError('Raw truth is not a model input')
        label_key = 'target' if head == 'agreement' else 'state'
        labels = torch.as_tensor(bank[label_key])
        valid = (labels == 0) | (labels == 1)
        if head != 'agreement':
            valid |= labels == 2
        if labels.shape != (n,) or not valid.all():
            raise ValueError('Invalid branch supervision')
        for key, width in (('tab', 43 if head == 'agreement' else 70), ('g', 128), ('l', 4096)):
            if torch.as_tensor(bank[key]).shape != (n, width):
                raise ValueError('Invalid routing feature dimensions')
        clean[dataset] = {k: bank[k] for k in ('dataset', 'split', 'ids', 'tab', 'g', 'l', label_key)}
    return clean

def _parent_ids(ids_by_dataset, banks):
    result = {}
    for dataset, ids in ids_by_dataset.items():
        mapping = dict(zip(banks[dataset]['ids'].tolist(), banks[dataset]['node_ids'].tolist()))
        result[dataset] = [int(mapping[int(i)]) for i in ids]
    return result

def _diagnostics(stages, banks, *, mass=None):
    """Union source and adaptation parameter IDs; retain true transform scope."""
    input_ids, supervised, transforms = ({}, {}, {})
    for stage in stages:
        for dest, names in ((input_ids, ('input_ids', 'training_ids')), (supervised, ('supervised_ids', 'training_ids')), (transforms, ('transform_fit_ids', 'feature_fit_ids'))):
            values = next((stage[k] for k in names if k in stage), {})
            for dataset, ids in values.items():
                dest[dataset] = list(dict.fromkeys(dest.get(dataset, []) + list(ids)))
    return dict(input_ids=input_ids, supervised_ids=supervised, transform_fit_ids=transforms, input_parent_ids=_parent_ids(input_ids, banks), supervised_parent_ids=_parent_ids(supervised, banks), transform_fit_parent_ids=_parent_ids(transforms, banks), domain_loss_mass=mass if mass is not None else copy.deepcopy(stages[-1].get('domain_loss_mass', {})), stages=copy.deepcopy(stages), augmented_observations_equal_weight=True, environment_features=False, validation_training_rows=0, validation_transform_rows=0, query_training_rows=0, calibration='none; caller owns parent-group OOF')

def _source_key(cfg, head, sources, banks):
    """Content-hash every source input and target, not mutable object pointers.

    Rehashing detects in-place edits (including edits through NumPy aliases) and
    prevents source-fold or dataset substitutions. Target-only OOF changes do
    not invalidate a source fit. Environment metadata never enters the key.
    """
    import numpy as np
    source_cfg = {k: v for k, v in cfg.items() if k not in ('id', 'target_steps')}
    digest = hashlib.sha256(json.dumps(dict(head=head, config=source_cfg), sort_keys=True).encode())
    for dataset in sources:
        digest.update(dataset.encode())
        for key in ('ids', 'node_ids', 'tab', 'g', 'l', 'state'):
            value = np.ascontiguousarray(np.asarray(banks[dataset][key]))
            digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
            digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()

def fit_candidate(cfg: dict, head: str, banks: dict[str, dict], target: str) -> dict:
    """Refit a recipe using training banks only; no validation argument exists."""
    _identity(target, head)
    cfg = copy.deepcopy(cfg)
    prepared = _prepare(banks, target, head)
    sources = [d for d in DATASETS if d != target]
    scope = cfg.get('scope')
    if head != 'agreement' and cfg['kind'] == 'neural':
        if float(cfg.get('consistency', 0)) != 0:
            raise ValueError('Unlabeled consistency is outside the augmented training protocol')
        if any((d not in prepared for d in sources)):
            raise ValueError('Neural source training requires all other datasets')
        from eargtc.neural.model import fit_head
        cache_key = _source_key(cfg, head, sources, banks)
        cache_hit = cache_key in _SOURCE_CACHE
        if cache_hit:
            _SOURCE_CACHE.move_to_end(cache_key)
        else:
            source = fit_head([prepared[d] for d in sources], head, cfg, device='cpu', steps=cfg['source_steps'], unlabeled=[])
            _SOURCE_CACHE[cache_key] = copy.deepcopy(source)
            if len(_SOURCE_CACHE) > _SOURCE_CACHE_LIMIT:
                _SOURCE_CACHE.popitem(last=False)
        fitted = copy.deepcopy(_SOURCE_CACHE[cache_key])
        fitted['config'] = copy.deepcopy(cfg)
        stages = [fitted['training_diagnostics']]
        if cfg['target_steps']:
            fitted = fit_head([prepared[target]], head, cfg, device='cpu', initial=fitted, steps=cfg['target_steps'], unlabeled=[])
            stages.append(fitted['training_diagnostics'])
        diagnostics = _diagnostics(stages, banks)
        diagnostics.update(source_cache_hit=cache_hit, source_cache_key=cache_key)
    else:
        if scope not in ('source', 'pooled', 'target'):
            raise ValueError('Training scope must be source, pooled or target')
        selected = [target] if scope == 'target' else sources if scope == 'source' else [target] + sources if head == 'agreement' else sources + [target]
        if any((d not in prepared for d in selected)):
            raise ValueError('Missing source training dataset')
        if head == 'agreement':
            import numpy as np
            from eargtc.agreement.models import fit_predict
            selected = [d for d in selected if len(prepared[d]['ids'])]
            if not selected:
                raise ValueError('Agreement recipe has no training observations')
            xs = {k: np.concatenate([np.asarray(prepared[d][k]) for d in selected]) for k in ('tab', 'g', 'l')}
            y = np.concatenate([np.asarray(prepared[d]['target'], dtype=bool) for d in selected])
            mass = {d: (0.5 if d == target else 0.5 / len(sources)) if scope == 'pooled' else 1 / len(selected) for d in selected}
            weight = np.concatenate([np.full(len(prepared[d]['ids']), mass[d] / len(prepared[d]['ids'])) for d in selected])
            weight *= len(weight) / weight.sum()
            _, fitted = fit_predict(cfg['config'], xs, y, xs, sample_weight=weight, seed=42, device='cpu')
            ids = {d: prepared[d]['ids'].tolist() for d in selected}
            supervised = {} if fitted['kind'] == 'baseline' else ids
            transform_ids = {} if fitted['kind'] in ('baseline', 'constant') else ids
            stage = dict(fitted.get('training_diagnostics', {}), input_ids=ids, supervised_ids=supervised, transform_fit_ids=transform_ids, domain_loss_mass=mass)
            diagnostics = _diagnostics([stage], banks, mass=mass)
        else:
            from eargtc.tabular.experiment import fit_tabular
            fitted = fit_tabular([prepared[d] for d in selected], head, cfg)
            stage = dict(fitted['training_diagnostics'], input_ids={d: prepared[d]['ids'].tolist() for d in selected})
            diagnostics = _diagnostics([stage], banks)
    return dict(schema='transfer_candidate', version='transfer', head=head, target=target, config=cfg, fitted=fitted, training_diagnostics=diagnostics)

def predict_candidate(model: dict, bank: dict):
    """Return raw disagreement logits or original agreement probability scores."""
    import numpy as np
    if model.get('schema') != 'transfer_candidate':
        raise ValueError('Unknown fitted estimator schema')
    fitted = model['fitted']
    features = {k: bank[k] for k in ('tab', 'g', 'l')}
    if model['head'] == 'agreement':
        from eargtc.agreement.models import predict_model
        result = predict_model(fitted, features)
    elif fitted.get('kind', 'neural') == 'neural':
        from eargtc.neural.model import predict_head
        result = predict_head(fitted, dict(features, ids=bank['ids']), device='cpu')
    else:
        from eargtc.tabular.experiment import predict_tabular
        result = predict_tabular(fitted, features)
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (len(bank['ids']),) or not np.isfinite(result).all():
        raise ValueError('Invalid raw candidate predictions')
    return result
