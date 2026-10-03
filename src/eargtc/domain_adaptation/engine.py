"""Fit agreement, trust and preference estimators on labeled observations."""
from copy import deepcopy
import hashlib
import json

import numpy as np
from scipy.special import logit

from eargtc.feature_transforms import core as reference


DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
HEADS = ('agreement', 'trust', 'preference')
STAGES = ('regularization', 'transfer', 'training')
SEEDS = [42, 137, 2026]


def _model_config(base, head):
    """Convert a fitted estimator record into a training configuration."""
    if not isinstance(base, dict) or base.get('engine') not in ('transfer', 'tabular', 'ensemble', 'adaptation'):
        raise ValueError('base must contain a supported fitted routing model')
    model = base.get('model')
    if not isinstance(model, dict):
        raise ValueError('base.model must be a fitted model dictionary')
    cfg = model.get('config')
    if base['engine'] == 'transfer' and isinstance(cfg, dict):
        inner = cfg.get('config') if isinstance(cfg.get('config'), dict) else None
        kind = cfg.get('kind')
        if kind == 'neural':
            seed = _defaults('neural', 'context', 'transfer')
            seed.update({k: deepcopy(cfg[k]) for k in ('width', 'dropout', 'lr', 'weight_decay') if k in cfg})
            seed['steps'] = int(cfg.get('source_steps', seed['steps']))
            seed['adapt_steps'] = int(cfg.get('target_steps', cfg.get('adapt_steps', seed['adapt_steps'])))
            return seed
        if inner and inner.get('family') == 'extra_trees':
            seed = _defaults('trees', 'full', cfg.get('scope', 'target'))
            seed.update(depth=int(inner.get('max_depth', seed['depth'])),
                        leaf=int(inner.get('min_samples_leaf', seed['leaf'])),
                        n_estimators=int(inner.get('n_estimators', seed['n_estimators'])))
            return seed
        if inner and inner.get('family') == 'lr':
            seed = _defaults('lr', 'prob' if inner.get('feature_mode') == 'prob' else 'context', cfg.get('scope', 'pooled'))
            seed['C'] = float(inner.get('C', seed['C']))
            return seed
        if kind == 'ridge':
            seed = _defaults('lr', 'prob' if cfg.get('feature_mode') == 'prob' else 'context', cfg.get('scope', 'pooled'))
            seed['C'] = float(cfg.get('C', seed['C']))
            return seed
        return _defaults('lr', 'context', 'pooled')
    if base['engine'] == 'transfer' and (not isinstance(cfg, dict) or cfg.get('family') not in
                                   ('mlp', 'neural', 'residual', 'joint_shared')):
        return _defaults('lr', 'context', 'pooled')
    if not isinstance(cfg, dict):
        return _defaults('lr', 'context', 'pooled')
    family = cfg.get('family', 'lr')
    if family in ('mlp', 'residual'):
        family = 'neural'
    if family not in ('lr', 'trees', 'svc', 'neural', 'joint_shared'):
        family = 'lr'
    scope = cfg.get('scope', 'transfer' if family in ('neural', 'joint_shared') else 'pooled')
    seed = _defaults(family, cfg.get('mode', 'context'), scope)
    seed.update({k: deepcopy(v) for k, v in cfg.items() if k in seed})
    seed['family'] = family
    seed.pop('id', None)
    return seed


def _defaults(family, mode='context', scope='pooled'):
    common = dict(family=family, mode=mode, scope=scope, rank=False, seed=42,
                  target_mass=.25, source_profile='uniform')
    if family == 'lr':
        common.update(C=.3)
    elif family == 'trees':
        common.update(depth=8, leaf=12, n_estimators=256)
    elif family == 'svc':
        common.update(C=1., gamma=.1)
    elif family in ('neural', 'joint_shared'):
        common.update(width=32, steps=280, adapt_steps=60, weight_decay=.05,
                      dropout=.15, lr=.002, seeds=list(SEEDS))
        common['scope'] = scope
        if family == 'joint_shared':
            common.update(preference_loss_weight=1., trust_loss_weight=1.)
    return common


def _identified(rows, stage):
    out, seen = [], set()
    for cfg in rows:
        cfg = deepcopy(cfg)
        cfg['stage'] = stage
        payload = json.dumps(cfg, sort_keys=True, separators=(',', ':'))
        ident = 'candidate_' + stage[:3] + '_' + hashlib.sha1(payload.encode()).hexdigest()[:12]
        if ident in seen:
            continue
        seen.add(ident)
        cfg['id'] = ident
        out.append(cfg)
    if len(out) > 60:
        raise AssertionError('Router configuration count exceeds the supported bound')
    return out


def _vary(seed, key, values):
    rows = []
    for value in values:
        row = deepcopy(seed)
        row[key] = value
        rows.append(row)
    return rows


def candidate_grid(dataset, head, stage, base):
    if dataset not in DATASETS or head not in HEADS or stage not in STAGES:
        raise ValueError((dataset, head, stage))
    seed = _model_config(base, head)
    family = seed['family']
    rows = []
    if stage == 'regularization':
        if family == 'lr':
            modes = ('compact', 'context', 'embed') if (head == 'agreement' and dataset in ('cora', 'citeseer')) else ('prob', 'compact', 'context', 'embed')
            for mode in modes:
                for c in (.03, .3, 3.):
                    row = deepcopy(seed); row.update(mode=mode, C=c)
                    rows.append(row)
        elif family == 'trees':
            rows += _vary(seed, 'depth', (4, 8, 12))
            rows += _vary(seed, 'leaf', (6, 12, 24))
        elif family == 'svc':
            rows += _vary(seed, 'C', (.1, 1., 10.))
            rows += _vary(seed, 'gamma', (.01, .1, 'scale'))
        else:
            rows += _vary(seed, 'weight_decay', (.01, .05, .15))
    elif stage == 'transfer':
        # Each row changes one transfer choice from the selected seed.
        pooled = deepcopy(seed); pooled['scope'] = 'pooled'
        rows += _vary(pooled, 'target_mass', (.05, .1, .5, .75))
        rows += _vary(seed, 'source_profile', ('small_graphs', 'large_graphs'))
        rows += _vary(seed, 'scope', ('source', 'pooled', 'target'))
    elif family in ('neural', 'joint_shared'):
        rows += _vary(seed, 'steps', (140, 560))
        rows += _vary(seed, 'adapt_steps', (0, 20, 120))
        rows += _vary(seed, 'weight_decay', (.01, .15))
        rows += _vary(seed, 'dropout', (0., .3))
        if family == 'joint_shared':
            rows += _vary(seed, 'preference_loss_weight', (.25, .5, 2., 4.))
    elif family == 'trees':
        rows += _vary(seed, 'leaf', (2, 4, 16, 32))
        rows += _vary(seed, 'n_estimators', (256, 768))
    elif family in ('lr', 'svc'):
        c = float(seed.get('C', 1.))
        rows += _vary(seed, 'C', (.3 * c, 3. * c))
    return _identified(rows, stage)


def _check_bank(bank, dataset, head):
    required = {'dataset', 'split', 'ids', 'node_ids', 'weight', 'tab', 'g', 'l'}
    required.add('target' if head == 'agreement' else 'state')
    if not isinstance(bank, dict) or not required.issubset(bank):
        raise ValueError('Unknown training-bank schema')
    if bank['split'] != 'train':
        raise ValueError('fit_candidate accepts split=train only')
    if bank['dataset'] != dataset:
        raise ValueError('Training dataset key mismatch')
    ids = np.asarray(bank['ids'])
    parents = np.asarray(bank['node_ids'])
    n = len(ids)
    if ids.ndim != 1 or parents.shape != (n,) or len(np.unique(ids)) != n:
        raise ValueError('Training IDs and parents must be aligned; IDs must be unique')
    weights = np.asarray(bank['weight'])
    if weights.shape != (n,) or not np.all(weights == 1):
        raise ValueError('Six augmented views must have equal unit weights')
    tab = np.asarray(bank['tab'])
    expected_width = 43 if head == 'agreement' else 70
    if tab.ndim != 2 or tab.shape != (n, expected_width):
        raise ValueError('Unknown feature-bank schema')
    if np.asarray(bank['g']).ndim != 2 or len(bank['g']) != n or np.asarray(bank['l']).ndim != 2 or len(bank['l']) != n:
        raise ValueError('Embedding rows must align with sample IDs')
    label = np.asarray(bank['target' if head == 'agreement' else 'state'])
    allowed = (0, 1) if head == 'agreement' else (0, 1, 2)
    if label.shape != (n,) or not np.isin(label, allowed).all():
        raise ValueError('Invalid training label schema')


def _domain_masses(use, target, cfg):
    scope = cfg['scope']
    if scope == 'target':
        return {target: 1.}
    sources = [d for d in use if d != target]
    profile = cfg.get('source_profile', 'uniform')
    if profile not in ('uniform', 'small_graphs', 'large_graphs'):
        raise ValueError('Unknown source_profile')
    small = {'cora', 'citeseer'}
    raw = {}
    for dataset in sources:
        raw[dataset] = (2. if (profile == 'small_graphs' and dataset in small) or
                               (profile == 'large_graphs' and dataset not in small) else 1.)
    total = sum(raw.values())
    source_mass = 1. if scope == 'source' else 1. - float(cfg.get('target_mass', .25))
    if not 0. < source_mass < 1. and scope in ('pooled', 'transfer'):
        raise ValueError('target_mass must be strictly between zero and one')
    masses = {d: source_mass * raw[d] / total for d in sources}
    if scope in ('pooled', 'transfer'):
        masses[target] = float(cfg.get('target_mass', .25))
    return masses


def _network(inputs, cfg, outputs=1):
    from torch import nn
    width = int(cfg.get('width', 32))
    return nn.Sequential(nn.Linear(inputs, width), nn.GELU(),
                         nn.Dropout(float(cfg.get('dropout', .15))),
                         nn.Linear(width, max(width // 2, 1)), nn.GELU(),
                         nn.Linear(max(width // 2, 1), outputs))


def joint_loss(logits, state, preference_logits=None, preference_state=None, *,
               trust_loss_weight=1., preference_loss_weight=1.,
               trust_active=True, preference_active=True):
    from torch.nn import functional as F
    if logits.ndim != 2 or logits.shape[1] != 2 or state.shape != logits.shape[:1]:
        raise ValueError('Joint loss requires [N,2] logits and aligned state')
    pz = logits if preference_logits is None else preference_logits
    ps = state if preference_state is None else preference_state
    if pz.ndim != 2 or pz.shape[1] != 2 or ps.shape != pz.shape[:1]:
        raise ValueError('Preference logits/state must align')
    trust = F.binary_cross_entropy_with_logits(logits[:, 0], (state != 2).float()) if trust_active else logits[:, 0].sum() * 0.
    decisive = ps != 2
    pref = F.binary_cross_entropy_with_logits(pz[decisive, 1], (ps[decisive] == 0).float()) if preference_active and decisive.any() else pz[:, 1].sum() * 0.
    return float(trust_loss_weight) * trust + float(preference_loss_weight) * pref


def _fit_sklearn(x, y, w, cfg):
    if len(np.unique(y)) < 2:
        return dict(kind='constant', logit=float(logit((y.sum() + .5) / (len(y) + 1))))
    family = cfg['family']
    if family == 'lr':
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression(C=float(cfg['C']), max_iter=1500, random_state=int(cfg.get('seed', 42)))
    elif family == 'trees':
        from sklearn.ensemble import ExtraTreesClassifier
        estimator = ExtraTreesClassifier(n_estimators=int(cfg.get('n_estimators', 256)),
            max_depth=int(cfg['depth']), min_samples_leaf=int(cfg['leaf']), max_features=1.,
            random_state=int(cfg.get('seed', 42)), n_jobs=1)
    elif family == 'svc':
        from sklearn.svm import SVC
        estimator = SVC(C=float(cfg['C']), gamma=cfg['gamma'], kernel='rbf',
                        random_state=int(cfg.get('seed', 42)), cache_size=512)
    else:
        raise ValueError('Unknown nonneural family')
    estimator.fit(x, y.astype(int), sample_weight=w)
    return dict(kind='sklearn', estimator=estimator)


def _binary_plan(y, domains, cfg):
    target_domain = cfg.get('target_domain', -1)
    source = np.flatnonzero(domains != target_domain)
    target = np.flatnonzero(domains == target_domain)
    if not len(source):
        source = np.arange(len(y))
    plan = [('source', source, int(cfg.get('steps', 280)), float(cfg.get('lr', .002)))]
    if len(target) and len(np.unique(y[target])) == 2 and int(cfg.get('adapt_steps', 60)):
        plan.append(('target', target, int(cfg['adapt_steps']), float(cfg.get('adapt_lr', .0005))))
    return plan


def _fit_neural(x, y, w, domains, cfg):
    import torch
    from torch.nn import functional as F
    tx = torch.tensor(x, dtype=torch.float32); ty = torch.tensor(y, dtype=torch.float32)
    states = []
    plan = _binary_plan(y, domains, cfg)
    for seed in cfg.get('seeds', SEEDS):
        torch.manual_seed(int(seed)); rng = np.random.default_rng(seed)
        net = _network(x.shape[1], cfg, 1)
        optimizer = torch.optim.AdamW(net.parameters(), lr=float(cfg.get('lr', .002)),
                                      weight_decay=float(cfg.get('weight_decay', .05)))
        for _, eligible, steps, lr in plan:
            for group in optimizer.param_groups: group['lr'] = lr
            p = w[eligible] / w[eligible].sum()
            for _ in range(steps):
                chosen = rng.choice(eligible, 128, p=p)
                loss = F.binary_cross_entropy_with_logits(net(tx[chosen]).squeeze(1), ty[chosen])
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
        states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    return dict(kind='neural', states=states, config=deepcopy(cfg),
                target_adapted=any(name == 'target' for name, *_ in plan))


def _joint_plan(state, domains, cfg):
    td = cfg.get('target_domain', -1)
    source = np.flatnonzero(domains != td); target = np.flatnonzero(domains == td)
    if not len(source): source = np.arange(len(state))
    result = [('source', source, int(cfg.get('steps', 280)), float(cfg.get('lr', .002)), True, bool(np.any(state[source] != 2)))]
    if len(target) and int(cfg.get('adapt_steps', 60)):
        trust = len(np.unique(state[target] != 2)) == 2
        decisive = target[state[target] != 2]
        pref = len(decisive) > 0 and len(np.unique(state[decisive] == 0)) == 2
        if trust or pref:
            result.append(('target', target, int(cfg['adapt_steps']), float(cfg.get('adapt_lr', .0005)), trust, pref))
    return result


def _fit_joint(x, state, trust_w, preference_w, domains, cfg):
    import torch
    tx = torch.tensor(x, dtype=torch.float32); ts = torch.tensor(state, dtype=torch.long)
    plan = _joint_plan(state, domains, cfg); states = []
    for seed in cfg.get('seeds', SEEDS):
        torch.manual_seed(int(seed)); rng = np.random.default_rng(seed)
        net = _network(x.shape[1], cfg, 2)
        optimizer = torch.optim.AdamW(net.parameters(), lr=float(cfg.get('lr', .002)),
                                      weight_decay=float(cfg.get('weight_decay', .05)))
        for _, eligible, steps, lr, trust_active, pref_active in plan:
            for group in optimizer.param_groups: group['lr'] = lr
            tp = trust_w[eligible] / trust_w[eligible].sum()
            pref = eligible[state[eligible] != 2]
            pp = preference_w[pref] / preference_w[pref].sum() if len(pref) else None
            for _ in range(steps):
                ti = rng.choice(eligible, 128, p=tp); tz = net(tx[ti])
                if pref_active:
                    pi = rng.choice(pref, 128, p=pp); pz, ps = net(tx[pi]), ts[pi]
                else:
                    pz, ps = tz, ts[ti]
                loss = joint_loss(tz, ts[ti], pz, ps, trust_active=trust_active,
                    preference_active=pref_active, trust_loss_weight=cfg.get('trust_loss_weight', 1.),
                    preference_loss_weight=cfg.get('preference_loss_weight', 1.))
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
        states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    return dict(kind='joint_shared', states=states, config=deepcopy(cfg),
                target_adapted=any(row[0] == 'target' for row in plan),
                plan=[dict(name=n, steps=s, lr=lr, trust_active=t, preference_active=p)
                      for n, _, s, lr, t, p in plan])


def _predict_estimator(estimator, x):
    if estimator['kind'] == 'constant': return np.full(len(x), estimator['logit'])
    if estimator['kind'] == 'sklearn':
        fitted = estimator['estimator']
        if hasattr(fitted, 'decision_function'): return np.asarray(fitted.decision_function(x), float)
        return logit(np.clip(fitted.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6))
    import torch
    outputs = 2 if estimator['kind'] == 'joint_shared' else 1
    net = _network(x.shape[1], estimator['config'], outputs); scores = []
    for state in estimator['states']:
        net.load_state_dict(state); net.eval(); parts = []
        with torch.inference_mode():
            for start in range(0, len(x), 4096):
                parts.append(net(torch.tensor(x[start:start + 4096], dtype=torch.float32)).numpy())
        scores.append(np.concatenate(parts) if parts else np.empty((0, outputs)))
    mean = np.mean(scores, axis=0)
    return mean[:, 0] if outputs == 1 else mean


def fit_candidate(cfg, head, banks, target):
    if head not in HEADS or target not in DATASETS or set(banks) != set(DATASETS):
        raise ValueError('Routing requires a valid head, target and five training datasets')
    cfg = deepcopy(cfg); family = cfg.get('family')
    if family == 'joint_shared' and head == 'agreement':
        raise ValueError('joint_shared supports disagreement heads only')
    if family not in ('lr', 'trees', 'svc', 'neural', 'joint_shared'):
        raise ValueError('Unknown router estimator family')
    if cfg.get('mode') not in ('prob', 'compact', 'context', 'embed', 'full') or cfg.get('scope') not in ('source', 'pooled', 'target', 'transfer'):
        raise ValueError('Unknown router feature mode or training scope')
    for dataset, bank in banks.items(): _check_bank(bank, dataset, head)
    ordered = [d for d in DATASETS if d != target] + [target]
    use = ordered[:-1] if cfg['scope'] == 'source' else ordered[-1:] if cfg['scope'] == 'target' else ordered
    masses = _domain_masses(use, target, cfg)
    raw, labels, domains, transform_w, fit_w, preference_w = [], [], [], [], [], []
    raw_ids, raw_parents, fit_ids, fit_parents = {}, {}, {}, {}
    for di, dataset in enumerate(use):
        bank = banks[dataset]; n = len(bank['ids'])
        if n == 0:
            continue
        x = reference.raw_features(bank, cfg['mode'])
        state = np.asarray(bank['state']) if head != 'agreement' else None
        eligible = state != 2 if head == 'preference' else np.ones(n, bool)
        y = np.asarray(bank['target']).astype(bool) if head == 'agreement' else (state != 2 if head == 'trust' else state == 0)
        raw.append(x); labels.append(state if family == 'joint_shared' else y)
        domains.append(np.full(n, di)); transform_w.append(np.full(n, masses[dataset] / n))
        fit_w.append(np.where(eligible, masses[dataset] / max(int(eligible.sum()), 1), 0.))
        decisive = state != 2 if state is not None else np.ones(n, bool)
        preference_w.append(np.where(decisive, masses[dataset] / max(int(decisive.sum()), 1), 0.))
        raw_ids[dataset] = np.asarray(bank['ids']).tolist(); raw_parents[dataset] = np.asarray(bank['node_ids']).tolist()
        fit_ids[dataset] = np.asarray(bank['ids'])[eligible].tolist(); fit_parents[dataset] = np.asarray(bank['node_ids'])[eligible].tolist()
    if not raw:
        raise ValueError('No eligible training rows for requested head')
    x = np.concatenate(raw); state_or_y = np.concatenate(labels); domains = np.concatenate(domains)
    transform_w = np.concatenate(transform_w); fit_w = np.concatenate(fit_w); preference_w = np.concatenate(preference_w)
    transform_mask = np.ones(len(x), bool) if family == 'joint_shared' else fit_w > 0
    transform = reference.Transform.fit(x[transform_mask], transform_w[transform_mask], bool(cfg.get('rank', False)))
    xx = transform.apply(x); cfg['target_domain'] = len(use) - 1 if cfg['scope'] == 'transfer' else -1
    if family == 'joint_shared':
        state = state_or_y.astype(int); decisive = state != 2
        estimator = _fit_joint(xx, state, transform_w, preference_w, domains, cfg)
        pair_payload = json.dumps({'config': cfg, 'target': target,
                                   'ids': raw_ids}, sort_keys=True, separators=(',', ':'))
        pair_id = hashlib.sha256(pair_payload.encode()).hexdigest()[:20]
        all_z = _predict_estimator(estimator, xx)[:, int(head == 'preference')]
        eligible = decisive if head == 'preference' else np.ones(len(state), bool)
        norm_w = preference_w if head == 'preference' else transform_w
    else:
        eligible = fit_w > 0
        y = state_or_y[eligible].astype(bool); wx = fit_w[eligible]
        wx *= len(wx) / wx.sum()
        estimator = _fit_neural(xx[eligible], y, wx, domains[eligible], cfg) if family == 'neural' else _fit_sklearn(xx[eligible], y, wx, cfg)
        all_z = _predict_estimator(estimator, xx[eligible]); norm_w = wx; pair_id = None
    z = all_z[eligible] if family == 'joint_shared' else all_z
    nw = norm_w[eligible] if family == 'joint_shared' else norm_w
    center = float(np.average(z, weights=nw)); scale = max(float(np.sqrt(np.average((z - center) ** 2, weights=nw))), .1)
    transform_ids = raw_ids if family == 'joint_shared' else fit_ids
    transform_parents = raw_parents if family == 'joint_shared' else fit_parents
    diagnostics = dict(fit_sample_ids=fit_ids, fit_parent_ids=fit_parents,
        shared_fit_sample_ids=deepcopy(raw_ids) if family == 'joint_shared' else None,
        shared_fit_parent_ids=deepcopy(raw_parents) if family == 'joint_shared' else None,
        raw_feature_sample_ids=raw_ids, raw_feature_parent_ids=raw_parents,
        transform_fit_sample_ids=deepcopy(transform_ids), transform_fit_parent_ids=deepcopy(transform_parents),
        normalization_fit_sample_ids=deepcopy(fit_ids), normalization_fit_parent_ids=deepcopy(fit_parents),
        requested_domain_masses=deepcopy(masses), effective_domain_masses=deepcopy(masses),
        fit_rows=sum(map(len, fit_ids.values())), transform_fit_rows=int(transform_mask.sum()),
        validation_training_rows=0, query_training_rows=0, validation_loss_evaluations=0,
        environment_features=False, view_weights='all unit within each dataset mass',
        transform_order='raw_features on full branch before decisive projection',
        normalization_source='gold train predictions only', early_stopping=False)
    if family in ('neural', 'joint_shared') and cfg['scope'] == 'transfer':
        diagnostics['target_mass_effect'] = 'transform and train-score normalization only; source and target stages use separate samplers'
    empty_domains = [d for d in use if not len(banks[d]['ids'])]
    if empty_domains:
        diagnostics['empty_domains_skipped'] = empty_domains
        active_mass = sum(masses[d] for d in raw_ids)
        diagnostics['effective_domain_masses'] = {d: masses[d] / active_mass for d in raw_ids}
        diagnostics['empty_domain_compatibility'] = 'skip zero rows; preserve original domain indices and requested nonempty masses before native normalization'
    model = dict(version='domain_adaptation_router', config=cfg, head=head, target=target,
                 transform=transform, estimator=estimator, mode=cfg['mode'], center=center,
                 scale=scale, fit_diagnostics=diagnostics)
    if pair_id is not None: model['shared_pair_id'] = pair_id
    return model


def predict_candidate(model, bank):
    if not isinstance(model, dict) or model.get('version') != 'domain_adaptation_router':
        raise ValueError('Unknown serialized router model schema')
    x = reference.raw_features(bank, model['mode'])
    scores = _predict_estimator(model['estimator'], model['transform'].apply(x))
    if model['estimator']['kind'] == 'joint_shared':
        scores = scores[:, int(model['head'] == 'preference')]
    return np.asarray(scores, float).reshape(-1)
