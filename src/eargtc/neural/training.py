"""Support-supervised optimization for neural routing scorers."""
from __future__ import annotations
import copy
import hashlib
import os
import numpy as np
import torch
from torch.nn import functional as F
from .model import LearnedHead, PROBABILITY_INDICES, _features
_ALLOWED = {'ids', 'tab', 'g', 'l', 'state', 'cal_mask', 'policy_mask', 'gnn_pred', 'llm_pred', 'dataset', 'domain', 'split'}

def _validate(bank, labeled):
    unknown = set(bank) - _ALLOWED
    if unknown:
        raise ValueError(f'Unrecognized bank fields (possible label leakage): {sorted(unknown)}')
    if labeled and bank.get('split') != 'train':
        raise ValueError('Only identified support observations may enter the training loss')
    if ('state' in bank) != labeled:
        raise ValueError('Supervised banks need state; unlabeled banks must not contain state')
    n = len(bank['ids'])
    ids = torch.as_tensor(bank['ids'])
    if ids.ndim != 1 or ids.unique().numel() != n:
        raise ValueError('Bank IDs must be unique one-dimensional values')
    for key in ('tab', 'g', 'l'):
        value = torch.as_tensor(bank[key])
        if value.ndim != 2 or len(value) != n or (not torch.isfinite(value).all()):
            raise ValueError(f'Invalid {key} features')
    if torch.as_tensor(bank['tab']).shape[1] != 70:
        raise ValueError('tab must contain exactly 70 features')
    if labeled:
        state = torch.as_tensor(bank['state'])
        if state.shape != (n,) or not ((state == 0) | (state == 1) | (state == 2)).all():
            raise ValueError('Invalid supervision state')
    elif {'cal_mask', 'policy_mask'} & set(bank):
        raise ValueError('Unlabeled banks must not include labeled split masks')

def _name(bank, i):
    return str(bank.get('dataset', bank.get('domain', f'domain_{i}')))

def _subset(bank, indices):
    return {k: torch.as_tensor(bank[k])[indices].cpu() for k in ('ids', 'tab', 'g', 'l', 'state') if k in bank}

def _fit_transform(domains, config, shape_bank):
    populated = [b['tab'].double() for _, b in domains if len(b['ids'])]
    if populated:
        mean = torch.stack([x.mean(0) for x in populated]).mean(0)
        second = torch.stack([(x * x).mean(0) for x in populated]).mean(0)
        scale = (second - mean.square()).clamp_min(1e-12).sqrt().clamp_min(1e-06)
    else:
        mean, scale = (torch.zeros(70), torch.ones(70))
    mask = torch.ones(70, dtype=torch.bool)
    if config['feature_mode'] == 'prob':
        mask.zero_()
        mask[list(PROBABILITY_INDICES)] = True
    return dict(mean=mean.float(), scale=scale.float(), mask=mask, g_dim=int(torch.as_tensor(shape_bank['g']).shape[1]), l_dim=int(torch.as_tensor(shape_bank['l']).shape[1]))

def _sample(domains, count, rng, transform, device):
    chunks, targets, groups = ([], [], [])
    count = max(count, len(domains))
    for d, (_, bank) in enumerate(domains):
        size = count // len(domains) + int(d < count % len(domains))
        idx = torch.as_tensor(rng.integers(0, len(bank['ids']), size=size))
        chunks.append(_features(bank, idx, transform, device))
        if 'state' in bank:
            targets.append(bank['state'][idx].to(device))
        groups.append(slice(sum((x.stop - x.start for x in groups)), sum((x.stop - x.start for x in groups)) + size))
    features = [torch.cat([chunk[k] for chunk in chunks]) for k in range(3)]
    return (features, torch.cat(targets) if targets else None, groups)

def train_head(banks, head, config, device, initial, steps, unlabeled):
    head = head.lower()
    if head not in ('trust', 'preference') or not banks:
        raise ValueError('A valid head and at least one supervised bank are required')
    config = copy.deepcopy(config)
    if config['feature_mode'] not in ('prob', 'all', 'gated'):
        raise ValueError('Unknown feature mode')
    if config['embedding_mode'] not in ('none', 'gnn', 'llm', 'both'):
        raise ValueError('Unknown embedding mode')
    seed = int(config.get('seed', 42))
    if seed != 42:
        raise ValueError('This preregistered experiment uses seed42 only')
    steps = int(config['source_steps' if initial is None else 'adapt_steps'] if steps is None else steps)
    if steps < 0:
        raise ValueError('steps must be nonnegative')
    batch_size = min(256, int(config.get('batch_size', 256)))
    if batch_size < 1 or len(banks) > batch_size:
        raise ValueError('Invalid bounded batch size/domain count')
    rng = np.random.default_rng(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    domains, supervised_ids, input_ids = ([], {}, {})
    for i, bank in enumerate(banks):
        _validate(bank, True)
        name = _name(bank, i)
        if name in input_ids:
            raise ValueError('Each domain must appear as one bank')
        state = torch.as_tensor(bank['state'])
        selected = torch.where(state != 2)[0] if head == 'preference' else torch.arange(len(state))
        filtered = _subset(bank, selected)
        input_ids[name] = torch.as_tensor(bank['ids']).tolist()
        supervised_ids[name] = filtered['ids'].tolist()
        domains.append((name, filtered))
    transform = _fit_transform(domains, config, banks[0]) if initial is None else copy.deepcopy(initial['transform'])
    if initial is not None:
        if initial['head'] != head:
            raise ValueError('Cannot initialize a head from another head')
        for key in ('feature_mode', 'embedding_mode', 'width', 'dropout'):
            if initial['config'][key] != config[key]:
                raise ValueError('Adaptation architecture must match source')
    for bank in banks:
        if torch.as_tensor(bank['g']).shape[1] != transform['g_dim'] or torch.as_tensor(bank['l']).shape[1] != transform['l_dim']:
            raise ValueError('Embedding dimensions must match source transform')
    model = LearnedHead(config, transform).to(device)
    if initial is not None:
        model.load_state_dict(initial['state_dict'])
    uq_domains, unlabeled_ids = ([], {})
    for i, bank in enumerate(unlabeled or []):
        _validate(bank, False)
        name = _name(bank, i)
        if name not in input_ids:
            raise ValueError('Unlabeled domain is outside supplied training domains')
        if name in unlabeled_ids:
            raise ValueError('Repeated unlabeled domain')
        ids = torch.as_tensor(bank['ids']).tolist()
        if set(ids) & set(input_ids[name]):
            raise ValueError('Unlabeled query overlaps supervised bank IDs')
        indices = sorted(range(len(ids)), key=lambda j: (hashlib.sha256(f'42:{ids[j]}'.encode()).digest(), ids[j]))[:4096]
        subset = _subset(bank, torch.tensor(indices, dtype=torch.long))
        if subset['g'].shape[1] != transform['g_dim'] or subset['l'].shape[1] != transform['l_dim']:
            raise ValueError('Unlabeled embedding dimensions do not match source')
        unlabeled_ids[name] = sorted(subset['ids'].tolist())
        if len(indices):
            uq_domains.append((name, subset))
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    active = [(name, bank) for name, bank in domains if len(bank['ids'])]
    gradients = dict(gnn_projection=0.0, llm_projection=0.0, feature_gates=0.0)
    trace = []
    model.train()
    actual_steps = steps if active else 0
    for step in range(actual_steps):
        x, state, groups = _sample(active, batch_size, rng, transform, device)
        y = (state != 2).float() if head == 'trust' else (state == 0).float()
        original_logits = model(*x)
        logits, target = (original_logits, y)
        if config['mixup']:
            perm = torch.arange(len(y), device=device)
            lam = torch.ones(len(y), device=device)
            for group in groups:
                size = group.stop - group.start
                perm[group] = torch.as_tensor(rng.permutation(size) + group.start, device=device)
                lam[group] = float(rng.beta(config['mixup'], config['mixup']))
            mixed = [lam[:, None] * v + (1 - lam[:, None]) * v[perm] for v in x]
            target = lam * y + (1 - lam) * y[perm]
            logits = model(*mixed)
        loss = torch.stack([F.binary_cross_entropy_with_logits(logits[g], target[g]) for g in groups]).mean()
        if config['ranking']:
            ranking = []
            for group in groups:
                positive = torch.where(y[group] == 1)[0] + group.start
                negative = torch.where(y[group] == 0)[0] + group.start
                if len(positive) and len(negative):
                    n = min(len(positive), len(negative), 64)
                    pi = torch.as_tensor(rng.integers(0, len(positive), size=n), device=device)
                    ni = torch.as_tensor(rng.integers(0, len(negative), size=n), device=device)
                    ranking.append(F.softplus(-(original_logits[positive[pi]] - original_logits[negative[ni]])).mean())
                else:
                    ranking.append(original_logits[group].sum() * 0.0)
            loss = loss + config['ranking'] * torch.stack(ranking).mean()
        if config['consistency'] and uq_domains:
            ux, _, ugroups = _sample(uq_domains, batch_size, rng, transform, device)
            with torch.no_grad():
                teacher_prob = teacher(*ux).sigmoid()
            noisy = [v + 0.02 * torch.randn_like(v) for v in ux]
            student_logits = model(*noisy)
            consistency = torch.stack([F.binary_cross_entropy_with_logits(student_logits[g], teacher_prob[g]) for g in ugroups]).mean()
            loss = loss + config['consistency'] * consistency
        if model.gate_logits is not None:
            loss = loss + float(config.get('gate_penalty', 0.001)) * model.gate_logits.sigmoid().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for label, param in (('gnn_projection', None if model.gnn is None else model.gnn[1].weight), ('llm_projection', None if model.llm is None else model.llm[1].weight), ('feature_gates', model.gate_logits)):
            if param is not None and param.grad is not None:
                gradients[label] = max(gradients[label], float(param.grad.detach().norm()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if initial is None:
            with torch.no_grad():
                for tp, p in zip(teacher.parameters(), model.parameters()):
                    tp.lerp_(p, 0.05)
        if step == 0 or step + 1 == actual_steps or (step + 1) % 20 == 0:
            trace.append(dict(step=step + 1, loss=float(loss.detach())))
    diagnostics = dict(seed=seed, stage='source' if initial is None else 'adaptation', requested_steps=steps, actual_steps=actual_steps, input_ids=input_ids, supervised_ids=supervised_ids, transform_fit_ids=copy.deepcopy(supervised_ids) if initial is None else copy.deepcopy(initial['training_diagnostics']['transform_fit_ids']), unlabeled_ids=unlabeled_ids, teacher_kind='source_training_ema' if initial is None else 'fixed_source', domain_loss_mass={name: 1 / len(active) for name, _ in active}, max_supervised_batch_size=batch_size if actual_steps else 0, max_unlabeled_batch_size=batch_size if actual_steps and uq_domains and config['consistency'] else 0, gradient_norms=gradients, training_trace=trace, preference_n_excluded_before_transform_mixup_ranking=True, transductive_query_consistency=bool(actual_steps and uq_domains and config['consistency']))
    return dict(head=head, config=config, transform=transform, training_diagnostics=diagnostics, state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
