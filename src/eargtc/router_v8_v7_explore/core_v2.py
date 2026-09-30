"""Routing feature transforms and estimator families."""
"""Compact gold-only router models. No data loaders or query truth access."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from eargtc.router_v8_v6.core import guard_ids, exact_mask, precision_profile

def check_train(bank, metadata):
    guard_ids(bank['dataset'], np.asarray(bank['ids']).tolist(), bank['split'], metadata)

def raw_features(bank, mode):
    t = np.asarray(bank['tab'], dtype=np.float64)
    dis = t.shape[1] == 70
    if mode == 'full':
        return t
    if mode == 'prob':
        idx = list(range(16)) + list(range(27, 43)) + list(range(54, 63)) if dis else list(range(16)) + list(range(27, 36))
        return t[:, idx]
    if dis:
        idx = [0, 1, 2, 3, 5, 10, 11, 13, 14, 15, 27, 28, 30, 31, 54, 55, 56, 57]
        if mode == 'context':
            idx += [16, 43, 63, 64]
        x = t[:, idx]
        delta = np.column_stack([t[:, 1] - t[:, 28], t[:, 3] - t[:, 30], t[:, 0] - t[:, 27], t[:, 2] - t[:, 29]])
        x = np.column_stack([x, delta, delta[:, :2] * (1 - t[:, [10, 13]])])
    else:
        idx = [0, 1, 2, 3, 10, 11, 13, 14, 27, 28, 29, 30]
        if mode == 'context':
            idx += [16, 36, 37]
        x = t[:, idx]
        x = np.column_stack([x, t[:, 0] * t[:, 2], np.minimum(t[:, 0], t[:, 2]), np.abs(t[:, 10] - t[:, 13])])
    if mode == 'embed':
        parts = [x]
        for name, width, seed in [('g', 8, 171), ('l', 16, 173)]:
            a = np.asarray(bank[name])
            rng = np.random.default_rng(seed)
            bucket = rng.integers(0, width, a.shape[1])
            sign = rng.choice([-1.0, 1.0], a.shape[1])
            z = np.zeros((len(a), width), np.float32)
            for start in range(0, len(a), 2048):
                block = np.asarray(a[start:start + 2048], np.float32)
                for j in range(width):
                    z[start:start + len(block), j] = block[:, bucket == j] @ sign[bucket == j].astype(np.float32)
            z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-08)
            parts.append(z)
        x = np.column_stack(parts)
    if mode not in ('compact', 'context', 'embed'):
        raise ValueError(mode)
    return x

@dataclass
class Transform:
    mean: np.ndarray
    scale: np.ndarray
    indices: np.ndarray
    references: object = None

    @classmethod
    def fit(cls, x, w, rank=False):
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        if not len(x) or not np.isfinite(x).all():
            raise ValueError('empty/nonfinite training features')
        mean = np.average(x, axis=0, weights=w)
        scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
        active = np.flatnonzero(scale > 1e-08)
        scale[scale < 1e-08] = 1.0
        z = (x - mean) / scale
        chosen = []
        for j in active:
            if all((abs(np.average(z[:, j] * z[:, k], weights=w)) < 0.995 for k in chosen)):
                chosen.append(j)
        if not chosen:
            chosen = [0]
        ix = np.array(chosen, int)
        refs = [np.sort(x[:, j]) for j in ix] if rank else None
        return cls(mean, scale, ix, refs)

    def apply(self, x):
        x = np.asarray(x, float)
        if self.references is None:
            return np.clip((x[:, self.indices] - self.mean[self.indices]) / self.scale[self.indices], -8, 8)
        return np.column_stack([(np.searchsorted(r, x[:, j], side='left') + np.searchsorted(r, x[:, j], side='right')) / len(r) - 1 for j, r in zip(self.indices, self.references)])

def mix_rows(x, y, domains, seed):
    rng = np.random.default_rng(seed)
    parent = np.arange(len(x))
    for d in np.unique(domains):
        ids = np.flatnonzero(domains == d)
        parent[ids] = rng.permutation(ids)
    lam = rng.beta(0.4, 0.4, len(x))
    lam = np.maximum(lam, 1 - lam)
    return (lam[:, None] * x + (1 - lam[:, None]) * x[parent], lam * y + (1 - lam) * y[parent], parent, lam)

def rank_metrics(z, y, half):
    z = np.asarray(z, float)
    y = np.asarray(y, bool)
    half = np.asarray(half, bool)

    def auc(m):
        return float(roc_auc_score(y[m], z[m])) if len(np.unique(y[m])) == 2 else None
    full = auc(np.ones(len(y), bool))
    hs = [auc(half), auc(~half)]
    valid = [v for v in hs if v is not None]
    return dict(auc=full, half_auc=hs, robust=0.5 * full + 0.5 * min(valid) if full is not None and valid else full if full is not None else 0.5)

def fit_estimator(x, y, w, domains, cfg):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    if len(np.unique(y)) < 2:
        return dict(kind='constant', logit=float(logit((y.sum() + 0.5) / (len(y) + 1))))
    family = cfg['family']
    if family == 'mlp':
        return fit_neural(x, y, w, domains, cfg)
    if family == 'lr':
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=cfg['C'], max_iter=1500, random_state=cfg['seed'])
    elif family == 'trees':
        from sklearn.ensemble import ExtraTreesClassifier
        m = ExtraTreesClassifier(n_estimators=200, max_depth=cfg['depth'], min_samples_leaf=cfg['leaf'], max_features=1.0, random_state=cfg['seed'], n_jobs=1)
    else:
        raise ValueError(family)
    m.fit(x, y.astype(int), sample_weight=w)
    return dict(kind='sklearn', estimator=m)

def predict_estimator(m, x):
    x = np.asarray(x, float)
    if m['kind'] == 'constant':
        return np.full(len(x), m['logit'])
    if m['kind'] == 'sklearn':
        e = m['estimator']
        if hasattr(e, 'decision_function'):
            return np.asarray(e.decision_function(x), float)
        return logit(np.clip(e.predict_proba(x)[:, 1], 1e-06, 1 - 1e-06))
    import torch
    net = network(x.shape[1])
    out = []
    for state in m['states']:
        net.load_state_dict(state)
        net.eval()
        parts = []
        with torch.inference_mode():
            for i in range(0, len(x), 4096):
                parts.append(net(torch.tensor(x[i:i + 4096], dtype=torch.float32)).squeeze(1).numpy())
        out.append(np.concatenate(parts) if parts else np.empty(0))
    return np.mean(out, axis=0)

def network(width):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Dropout(0.15), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

def fit_neural(x, y, w, domains, cfg):
    import torch
    from torch.nn import functional as F
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    domains = np.asarray(domains)
    target = cfg.get('target_domain', -1)
    source = np.flatnonzero(domains != target)
    target_rows = np.flatnonzero(domains == target)
    if not len(source):
        source = np.arange(len(x))
    states = []
    augment = cfg['variant'] in ('aug', 'ssl')
    ranking = cfg['variant'] != 'plain'
    for seed in cfg.get('seeds', [42, 137, 2026]):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = network(x.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.05)
        p = w[source] / w[source].sum()
        if cfg['variant'] == 'ssl':
            encoder = net[:5]
            for step in range(cfg.get('pretrain_steps', 80)):
                ids = rng.choice(source, 128, p=p)
                a = tx[ids]
                donor = ids.copy()
                for d in np.unique(domains[ids]):
                    mask = np.flatnonzero(domains[ids] == d)
                    donor[mask] = rng.choice(source[domains[source] == d], len(mask))
                b = torch.where(torch.rand_like(a) < 0.25, tx[donor], a)
                za = F.normalize(encoder(a), dim=1)
                zb = F.normalize(encoder(b), dim=1)
                logits = za @ zb.T / 0.2
                labels = torch.arange(len(a))
                loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
                opt.zero_grad()
                loss.backward()
                opt.step()
        stages = [(source, cfg.get('steps', 280), 0.002)]
        if len(target_rows) and len(np.unique(y[target_rows])) == 2:
            stages.append((target_rows, cfg.get('adapt_steps', 60), 0.0005))
        for eligible, steps, lr in stages:
            for group in opt.param_groups:
                group['lr'] = lr
            p = w[eligible] / w[eligible].sum()
            for step in range(steps):
                ids = rng.choice(eligible, 128, p=p)
                a = tx[ids]
                b = ty[ids]
                dd = domains[ids]
                z = net(a).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(z, b)
                if ranking:
                    pair = np.arange(len(ids))
                    for d in np.unique(dd):
                        jj = np.flatnonzero(dd == d)
                        pair[jj] = rng.permutation(jj)
                    delta = b - b[pair]
                    valid = delta != 0
                    if valid.any():
                        loss = loss + 0.35 * F.softplus(-(z - z[pair])[valid] * delta[valid]).mean()
                if augment:
                    mx, my, _, _ = mix_rows(a.numpy(), b.numpy(), dd, int(rng.integers(2 ** 31)))
                    ma = torch.tensor(mx, dtype=torch.float32)
                    ma = ma * (torch.rand_like(ma) > 0.1) + 0.03 * torch.randn_like(ma)
                    loss = loss + 0.5 * F.binary_cross_entropy_with_logits(net(ma).squeeze(1), torch.tensor(my, dtype=torch.float32))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
        states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    return dict(kind='mlp', states=states, seeds=cfg.get('seeds', [42, 137, 2026]), target_adapted=bool(len(target_rows) and len(np.unique(y[target_rows])) == 2))

def candidate_grid():
    rows = []
    for mode in ('prob', 'full', 'compact', 'context', 'embed'):
        for scope in ('source', 'pooled', 'target'):
            for rank in (False, True):
                for c in (0.01, 0.1, 1.0, 10.0):
                    rows.append(dict(family='lr', mode=mode, scope=scope, rank=rank, C=c, seed=42))
            if mode != 'embed':
                for depth, leaf in ((2, 5), (4, 5), (8, 3)):
                    rows.append(dict(family='trees', mode=mode, scope=scope, rank=False, depth=depth, leaf=leaf, seed=42))
    for mode in ('compact', 'context', 'embed'):
        for variant in ('plain', 'rank', 'aug', 'ssl'):
            rows.append(dict(family='mlp', mode=mode, scope='transfer', rank=False, variant=variant, seed=42, seeds=[42, 137, 2026], steps=280, adapt_steps=60, pretrain_steps=80))
    for i, cfg in enumerate(rows):
        cfg['id'] = f"c{i:03d}_{cfg['family']}_{cfg['mode']}_{cfg['scope']}"
    return rows
