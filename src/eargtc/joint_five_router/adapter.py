"""Build residual-classifier inputs and pseudo-label supervision pools."""
from __future__ import annotations
import torch
from eargtc.joint_cross_repair.artifacts import validate_features

def _tensor(value, *, dtype, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous().view(-1)
    if result.numel() == 0 or not bool(torch.isfinite(result.float()).all()):
        raise ValueError(f'invalid {name}')
    return result

def build_router_features_and_pools(base: dict, test_ids: torch.Tensor, decision: dict[str, torch.Tensor], choose_g: torch.Tensor, trust: torch.Tensor | None=None):
    """Apply a frozen route/decision to base features and form exact accepted pools."""
    n = base['node_ids'].numel()
    if not torch.equal(base['node_ids'], torch.arange(n)):
        raise ValueError('base node IDs are not canonical')
    test = _tensor(test_ids, dtype=torch.long, name='test IDs')
    if not torch.equal(decision['node_ids'], test):
        raise ValueError('Routing node IDs differ from query IDs')
    route = _tensor(choose_g, dtype=torch.bool, name='router route')
    if route.numel() != n:
        raise ValueError('router route does not cover all nodes')
    g, l = (base['logp_g'].argmax(1), base['logp_l'].argmax(1))
    agreement = g == l
    selected = torch.where(route, g, l)
    if not torch.equal(selected[test], decision['selected_pred']):
        raise ValueError('selected prediction differs from frozen route')
    code = decision['module_code']
    if not bool(torch.isin(code, torch.tensor([0, 1, 2], dtype=torch.int8)).all()):
        raise ValueError('invalid frozen module code')
    if bool((code == 1).any()) and (not bool(agreement[test][code == 1].all())):
        raise ValueError('agreement module code differs from expert agreement')
    if bool((code == 2).any()) and (not bool((~agreement[test][code == 2]).all())):
        raise ValueError('conflict module code differs from expert agreement')
    full_trust = torch.zeros(n, dtype=torch.float32) if trust is None else _tensor(trust, dtype=torch.float32, name='router trust')
    if full_trust.numel() != n or bool((full_trust < 0).any()) or bool((full_trust > 1).any()):
        raise ValueError('invalid router trust')
    features = {'node_ids': base['node_ids'], 'gnn_embeddings': base['gnn_embeddings'], 'logp_g': base['logp_g'], 'logp_l': base['logp_l'], 'choose_g': route, 'agreement': agreement, 'selected_pred': selected, 'trust': full_trust}
    accepted = torch.zeros(n, dtype=torch.bool)
    accepted[test] = decision['accepted']
    pools = {'teach_g': torch.nonzero(accepted & ~agreement & ~route, as_tuple=False).view(-1), 'teach_l': torch.nonzero(accepted & ~agreement & route, as_tuple=False).view(-1), 'agreement': torch.nonzero(accepted & agreement, as_tuple=False).view(-1)}
    pools['keep_g'] = torch.cat([pools['teach_l'], pools['agreement']]).sort().values
    pools['keep_l'] = torch.cat([pools['teach_g'], pools['agreement']]).sort().values
    selected_ids = torch.cat([pools['teach_g'], pools['teach_l'], pools['agreement']])
    pools['unselected'] = test[~torch.isin(test, selected_ids)].sort().values
    validate_features(features)
    return (features, pools)
