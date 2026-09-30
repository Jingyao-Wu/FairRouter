"""Joint trust and preference training with task-specific supervision."""
from copy import deepcopy
import numpy as np
from eargtc.router_aug_search import v7_engine as original
from eargtc.router_v8_v7_explore import core_v2 as historical

def _network(inputs, width):
    from torch import nn
    return nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Dropout(0.15), nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 2))

def joint_loss(logits, state, preference_logits=None, preference_state=None, *, trust_active=True, preference_active=True):
    """Sum independent BCE terms; neither never becomes a preference negative.

    Optional separate preference observations support task-specific domain-mass
    sampler. The decisive mask remains explicit even for that eligible sampler.
    """
    from torch.nn import functional as F
    if logits.ndim != 2 or logits.shape[1] != 2 or state.shape != logits.shape[:1]:
        raise ValueError('Joint loss requires [N,2] logits and aligned state')
    pz = logits if preference_logits is None else preference_logits
    ps = state if preference_state is None else preference_state
    if pz.ndim != 2 or pz.shape[1] != 2 or ps.shape != pz.shape[:1]:
        raise ValueError('Preference logits/state must align')
    trust = F.binary_cross_entropy_with_logits(logits[:, 0], (state != 2).float()) if trust_active else logits[:, 0].sum() * 0.0
    decisive = ps != 2
    preference = F.binary_cross_entropy_with_logits(pz[decisive, 1], (ps[decisive] == 0).float()) if preference_active and decisive.any() else pz[:, 1].sum() * 0.0
    return trust + preference

def _plan(state, domains, cfg):
    source = np.flatnonzero(domains != cfg['target_domain'])
    target = np.flatnonzero(domains == cfg['target_domain'])
    if not len(source):
        source = np.arange(len(state))
    plan = [dict(name='source', eligible=source, steps=cfg['steps'], lr=0.002, trust_active=True, preference_active=bool(np.any(state[source] != 2)))]
    if len(target):
        trust_active = len(np.unique(state[target] != 2)) == 2
        decisive = target[state[target] != 2]
        preference_active = len(np.unique(state[decisive] == 0)) == 2
        if trust_active or preference_active:
            plan.append(dict(name='target', eligible=target, steps=cfg['adapt_steps'], lr=0.0005, trust_active=trust_active, preference_active=preference_active))
    return plan

def _fit_estimator(x, state, trust_weights, preference_weights, plan, cfg):
    import torch
    tx = torch.tensor(x, dtype=torch.float32)
    ts = torch.tensor(state, dtype=torch.long)
    states = []
    for seed in cfg['seeds']:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = _network(x.shape[1], cfg['width'])
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.05)
        for stage in plan:
            eligible = stage['eligible']
            pref = eligible[state[eligible] != 2]
            tp = trust_weights[eligible] / trust_weights[eligible].sum()
            pp = preference_weights[pref] / preference_weights[pref].sum() if len(pref) else None
            for group in optimizer.param_groups:
                group['lr'] = stage['lr']
            for _ in range(stage['steps']):
                ti = rng.choice(eligible, 128, p=tp)
                z = net(tx[ti])
                if stage['preference_active']:
                    pi = rng.choice(pref, 128, p=pp)
                    pz, ps = (net(tx[pi]), ts[pi])
                else:
                    pz, ps = (z, ts[ti])
                loss = joint_loss(z, ts[ti], pz, ps, trust_active=stage['trust_active'], preference_active=stage['preference_active'])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()
        states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    return dict(kind='joint_shared', states=states, seeds=list(cfg['seeds']), config=deepcopy(cfg), target_adapted=any((s['name'] == 'target' for s in plan)))

def _predict_pair(estimator, x):
    import torch
    if not len(x):
        return np.empty((0, 2), dtype=float)
    net = _network(x.shape[1], estimator['config']['width'])
    scores = []
    for state in estimator['states']:
        net.load_state_dict(state)
        net.eval()
        outputs = []
        with torch.inference_mode():
            for i in range(0, len(x), 4096):
                outputs.append(net(torch.tensor(x[i:i + 4096], dtype=torch.float32)).numpy())
        scores.append(np.concatenate(outputs))
    return np.mean(scores, axis=0)

def fit_candidate(cfg, head, banks, target):
    original._slurm()
    if head not in ('trust', 'preference') or target not in original.DATASETS:
        raise ValueError('Joint experiment supports disagreement heads only')
    if set(banks) != set(original.DATASETS):
        raise ValueError('Joint transfer requires exactly the five training datasets')
    cfg = deepcopy(cfg)
    if cfg.get('family') != 'joint_shared' or cfg.get('scope') != 'transfer' or cfg.get('mode') not in ('compact', 'context', 'embed') or (cfg.get('rank') is not False) or (cfg.get('width') not in (32, 64)) or (cfg.get('seeds') != [42, 137, 2026]):
        raise ValueError('Unregistered joint shared-trunk recipe')
    if any((not isinstance(cfg.get(k), int) or cfg[k] < 0 for k in ('steps', 'adapt_steps'))):
        raise ValueError('Joint training steps must be nonnegative integers')
    ordered = [d for d in original.DATASETS if d != target] + [target]
    arrays, states, tw, pw, domains = ([], [], [], [], [])
    sample_ids, parent_ids, preference_ids, preference_parents = ({}, {}, {}, {})
    masses, offsets = ({}, {})
    cursor = 0
    for j, dataset in enumerate(ordered):
        bank = banks[dataset]
        original._check_bank(bank, dataset)
        if 'weight' not in bank:
            raise ValueError('Explicit unit view weights are required')
        state = np.asarray(bank['state'])
        if state.shape != (len(bank['ids']),) or not np.isin(state, (0, 1, 2)).all():
            raise ValueError('Invalid disagreement states')
        if np.asarray(bank['tab']).ndim != 2 or np.asarray(bank['tab']).shape[1] != 70:
            raise ValueError('Joint experiment requires the disagreement feature bank')
        n = len(state)
        if not n:
            continue
        x = historical.raw_features(bank, cfg['mode'])
        decisive = state != 2
        mass = 0.25 if dataset == target else 0.75 / 4
        arrays.append(x)
        states.append(state)
        domains.append(np.full(n, j))
        tw.append(np.full(n, mass / n))
        pw.append(np.where(decisive, mass / max(int(decisive.sum()), 1), 0.0))
        masses[dataset] = mass
        sample_ids[dataset] = np.asarray(bank['ids']).tolist()
        parent_ids[dataset] = np.asarray(bank['node_ids']).tolist()
        preference_ids[dataset] = np.asarray(bank['ids'])[decisive].tolist()
        preference_parents[dataset] = np.asarray(bank['node_ids'])[decisive].tolist()
        offsets[dataset] = slice(cursor, cursor + n)
        cursor += n
    if not arrays:
        raise ValueError('No training observations')
    x, state, trust_weights, preference_weights, domains = map(np.concatenate, (arrays, states, tw, pw, domains))
    if head == 'preference' and (not np.any(state != 2)):
        raise ValueError('No decisive training observations for preference')
    trust_weights *= len(x) / trust_weights.sum()
    transform = historical.Transform.fit(x, trust_weights, False)
    xx = transform.apply(x)
    cfg['target_domain'] = len(ordered) - 1
    plan = _plan(state, domains, cfg)
    estimator = _fit_estimator(xx, state, trust_weights, preference_weights, plan, cfg)
    train_z = _predict_pair(estimator, xx)[:, int(head == 'preference')]
    eligible = state != 2 if head == 'preference' else np.ones(len(state), bool)
    weights = preference_weights if head == 'preference' else trust_weights
    center = float(np.average(train_z[eligible], weights=weights[eligible]))
    scale = max(float(np.sqrt(np.average((train_z[eligible] - center) ** 2, weights=weights[eligible]))), 0.1)
    active = {h: np.zeros(len(state), bool) for h in ('trust', 'preference')}
    for stage in plan:
        if not stage['steps']:
            continue
        for h in active:
            if stage[h + '_active']:
                rows = stage['eligible']
                if h == 'preference':
                    rows = rows[state[rows] != 2]
                active[h][rows] = True
    loss_ids, loss_parents = ({}, {})
    for h in active:
        loss_ids[h] = {d: np.asarray(sample_ids[d])[active[h][rows]].tolist() for d, rows in offsets.items()}
        loss_parents[h] = {d: np.asarray(parent_ids[d])[active[h][rows]].tolist() for d, rows in offsets.items()}
    head_ids = preference_ids if head == 'preference' else sample_ids
    head_parents = preference_parents if head == 'preference' else parent_ids
    audit = dict(fit_sample_ids=deepcopy(head_ids), fit_parent_ids=deepcopy(head_parents), shared_fit_sample_ids=sample_ids, shared_fit_parent_ids=parent_ids, trust_loss_sample_ids=loss_ids['trust'], trust_loss_parent_ids=loss_parents['trust'], preference_loss_sample_ids=loss_ids['preference'], preference_loss_parent_ids=loss_parents['preference'], transform_fit_sample_ids=deepcopy(sample_ids), transform_fit_parent_ids=deepcopy(parent_ids), raw_feature_sample_ids=deepcopy(sample_ids), normalization_fit_sample_ids=deepcopy(head_ids), normalization_fit_parent_ids=deepcopy(head_parents), domain_masses=masses, fit_rows=int(eligible.sum()), shared_fit_rows=len(state), target_domain=cfg['target_domain'], stages=[{k: v for k, v in s.items() if k != 'eligible'} for s in plan], validation_training_rows=0, query_training_rows=0, validation_loss_evaluations=0, environment_features=False, view_weights='all unit; configured per-head domain masses retained', preference_sampling='decisive only, domain mass divided by decisive count', neither='trust BCE and shared trunk through trust only; zero preference BCE gradient', shared_transform='all training disagreement rows; no validation or query inputs', early_stopping=False)
    return dict(version='Router-v8(v7)', head=head, target=target, config=cfg, transform=transform, estimator=estimator, mode=cfg['mode'], center=center, scale=scale, fit_ids=head_ids, fit_audit=audit)

def predict_candidate(model, bank):
    original._slurm()
    x = historical.raw_features(bank, model['mode'])
    return np.asarray(_predict_pair(model['estimator'], model['transform'].apply(x))[:, int(model['head'] == 'preference')], float)
