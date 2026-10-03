"""Prepare residual-classifier data for each dataset and split."""
import torch
from eargtc.validation.artifacts import validate_features, validate_supervision, validate_pools
from eargtc.supervision.adapter import build_router_features_and_pools
from eargtc.calibration.calibration import deterministic_calibration_threshold_split

def validation_halves(dataset, valid, agreement):
    selection, threshold = ([], [])
    for module in ('conflict', 'agreement'):
        ids = valid[agreement[valid] if module == 'agreement' else ~agreement[valid]]
        split = deterministic_calibration_threshold_split(ids, dataset, module)
        selection.append(ids[split.calibration])
        threshold.append(ids[split.threshold])
    return (torch.cat(selection).sort().values, torch.cat(threshold).sort().values)

def prepare_arrays(split, upstream, embeddings, safe, router, dataset):
    """Strict schemas prevent query truth from entering the trainer adapter."""
    if set(upstream) != {'gnn_logits', 'llm_logits'} or set(embeddings) != {'embeddings', 'node_ids'}:
        raise ValueError('unexpected upstream schema')
    if set(safe) != {'support_ids', 'support_targets', 'valid_ids', 'valid_targets'}:
        raise ValueError('unsafe supervision schema')
    core = {'node_ids', 'choose_g', 'trust', 'accepted', 'selected_pred', 'agreement'}
    extras = {'preference_score', 'logp_g', 'logp_l', 'selected_score'}
    if not core.issubset(router) or not set(router).issubset(core | extras):
        raise ValueError('unsafe Router schema')
    n = split['num_nodes']
    if not torch.equal(embeddings['node_ids'], torch.arange(n)):
        raise ValueError('embedding node alignment differs')
    for key in set(router) - core:
        value = router[key]
        expected = upstream['gnn_logits'].shape if key in ('logp_g', 'logp_l') else (n,)
        if value.shape != expected or not bool(torch.isfinite(value).all()):
            raise ValueError('invalid safe Router auxiliary field: ' + key)
        if key in ('logp_g', 'logp_l'):
            source = upstream['gnn_logits' if key == 'logp_g' else 'llm_logits']
            if not torch.allclose(value.exp(), source.float().softmax(1), atol=2e-06):
                raise ValueError('Router auxiliary expert differs from current upstream')
    ids = {key: torch.tensor(split[key], dtype=torch.long) for key in ('support_ids', 'valid_ids', 'unlabeled_ids', 'standard_eval_ids')}
    standard = ids['standard_eval_ids']
    if len(standard) != 1000 or len(standard.unique()) != 1000 or not bool(torch.isin(standard, ids['unlabeled_ids']).all()):
        raise ValueError('standard evaluation IDs must be 1000 distinct query nodes')
    for key in ('support_ids', 'valid_ids'):
        if not torch.equal(safe[key], ids[key]):
            raise ValueError('safe supervision differs from outer split')
    if safe['support_targets'].dtype != torch.long or safe['support_targets'].shape != ids['support_ids'].shape:
        raise ValueError('invalid support targets')
    if safe['valid_targets'].dtype != torch.long or safe['valid_targets'].shape != ids['valid_ids'].shape:
        raise ValueError('invalid validation targets')
    if not torch.equal(router['node_ids'], torch.arange(n)):
        raise ValueError('Router node alignment differs')
    if router['accepted'].dtype != torch.bool or router['accepted'].shape != (n,):
        raise ValueError('accepted must be a full canonical Boolean vector')
    protected = torch.cat([ids['support_ids'], ids['valid_ids']])
    if bool(router['accepted'][protected].any()):
        raise ValueError('Router accepted protected nodes')
    f = {'node_ids': router['node_ids'], 'gnn_embeddings': embeddings['embeddings'].float(), 'logp_g': upstream['gnn_logits'].float().log_softmax(1), 'logp_l': upstream['llm_logits'].float().log_softmax(1), **{key: router[key] for key in ('choose_g', 'trust', 'selected_pred', 'agreement')}}
    validate_features(f)
    selection, threshold = validation_halves(dataset, ids['valid_ids'], f['agreement'])
    lookup = dict(zip(ids['valid_ids'].tolist(), safe['valid_targets'].tolist()))
    s = {'gold_ids': ids['support_ids'], 'gold_targets': safe['support_targets'], 'selection_ids': selection, 'selection_targets': torch.tensor([lookup[i] for i in selection.tolist()]), 'threshold_ids': threshold, 'test_ids': ids['unlabeled_ids']}
    validate_supervision(s, n, f['logp_g'].shape[1])
    query = s['test_ids']
    decision = {'node_ids': query, 'accepted': router['accepted'][query], 'selected_pred': router['selected_pred'][query], 'module_code': torch.where(f['agreement'][query], 1, 2).to(torch.int8)}
    f, p = build_router_features_and_pools(f, query, decision, f['choose_g'], f['trust'])
    validate_pools(p, f, s)
    accepted = torch.cat([p['teach_g'], p['teach_l'], p['agreement']]).sort().values
    if not torch.equal(accepted, torch.where(router['accepted'])[0]):
        raise ValueError('pool builder changed Router acceptance')
    return (f, s, p)
