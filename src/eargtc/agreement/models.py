"""Agreement scorers and their training objectives."""
from __future__ import annotations
import copy
from typing import Any, Iterator
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F
TAB_DIM = 43
G_DIM = 128
L_DIM = 4096
PROBABILITY_COLUMNS = tuple(range(16)) + tuple(range(27, 36))
_PREDICT_BATCH_SIZE = 2048
_PROJECTION_SEED = 20260908

def _validate_inputs(inputs: dict[str, Any], *, name: str) -> dict[str, np.ndarray]:
    dimensions = {'tab': TAB_DIM, 'g': G_DIM, 'l': L_DIM}
    result: dict[str, np.ndarray] = {}
    count: int | None = None
    for key, width in dimensions.items():
        if key not in inputs:
            raise ValueError(f'{name} is missing {key}')
        value = np.asarray(inputs[key])
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f'{name} {key} must have shape [N, {width}]')
        if count is None:
            count = len(value)
        elif len(value) != count:
            raise ValueError(f'{name} feature row counts differ')
        if not np.isfinite(value).all():
            raise ValueError(f'{name} {key} contains non-finite values')
        result[key] = value
    return result

def _targets_and_weights(y: Any, sample_weight: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(y)
    if target.ndim != 1 or len(target) != count:
        raise ValueError('y must have one binary target per training row')
    if not np.isfinite(target).all() or not np.isin(target, (0, 1)).all():
        raise ValueError('y must contain binary targets')
    target = target.astype(np.float64, copy=False)
    if sample_weight is None:
        weight = np.ones(count, dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64)
        if weight.ndim != 1 or len(weight) != count:
            raise ValueError('sample_weight must have one value per training row')
        weight = weight.copy()
    if not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError('sample_weight must be finite, nonnegative, and have positive mass')
    return (target, weight)

def _baseline(kind: str, tab: np.ndarray) -> np.ndarray:
    gnn = tab[:, 0].astype(np.float64)
    llm = tab[:, 2].astype(np.float64)
    if kind == 'gnn_probability':
        score = gnn
    elif kind == 'llm_probability':
        score = llm
    elif kind == 'min_probability':
        score = np.minimum(gnn, llm)
    elif kind == 'geomean_probability':
        score = np.sqrt(np.clip(gnn, 0.0, 1.0) * np.clip(llm, 0.0, 1.0))
    elif kind == 'min_margin':
        score = np.minimum(tab[:, 11], tab[:, 14])
    elif kind == 'negative_entropy':
        score = 1.0 - 0.5 * (tab[:, 10] + tab[:, 13])
    elif kind == 'prototype_signal':
        score = 0.5 * (tab[:, 19] + 1.0)
    else:
        raise ValueError(f'unknown baseline kind: {kind}')
    return np.clip(np.asarray(score, dtype=np.float64), 0.0, 1.0)

def _projection_indices(width: int, output: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(width, size=output, replace=False))

def _make_linear_transform(mode: str, seed: int) -> dict[str, Any]:
    transform: dict[str, Any] = {'feature_mode': mode}
    if mode in ('tab_g', 'tab_gl'):
        transform['g_indices'] = _projection_indices(G_DIM, 16, seed + 17)
    if mode in ('tab_l', 'tab_gl'):
        transform['l_indices'] = _projection_indices(L_DIM, 32, seed + 31)
    return transform

def _linear_raw(arrays: dict[str, np.ndarray], transform: dict[str, Any], rows: slice | np.ndarray) -> np.ndarray:
    mode = transform['feature_mode']
    tab = np.asarray(arrays['tab'][rows], dtype=np.float32)
    if mode == 'prob':
        return tab[:, PROBABILITY_COLUMNS]
    pieces = [tab]
    if mode in ('tab_g', 'tab_gl'):
        pieces.append(np.asarray(arrays['g'][rows, transform['g_indices']], dtype=np.float32))
    if mode in ('tab_l', 'tab_gl'):
        pieces.append(np.asarray(arrays['l'][rows, transform['l_indices']], dtype=np.float32))
    return np.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0]

def _linear_batches(arrays: dict[str, np.ndarray], transform: dict[str, Any], batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(arrays['tab']), batch_size):
        rows = slice(start, min(start + batch_size, len(arrays['tab'])))
        yield _linear_raw(arrays, transform, rows)

def _fit_linear_transform(arrays: dict[str, np.ndarray], mode: str, weight: np.ndarray, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    transform = _make_linear_transform(mode, seed)
    raw = _linear_raw(arrays, transform, slice(None)).astype(np.float64)
    mean = np.average(raw, axis=0, weights=weight)
    variance = np.average((raw - mean) ** 2, axis=0, weights=weight)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-07] = 1.0
    transform['mean'] = mean.astype(np.float64)
    transform['scale'] = scale.astype(np.float64)
    return (transform, np.clip((raw - mean) / scale, -12.0, 12.0))

def _positive_probability(estimator: Any, x: np.ndarray) -> np.ndarray:
    raw = estimator.predict_proba(x)
    classes = np.asarray(estimator.classes_)
    if 1 not in classes:
        return np.full(len(x), float(len(classes) and classes[0] == 1), dtype=np.float64)
    return np.asarray(raw[:, int(np.flatnonzero(classes == 1)[0])], dtype=np.float64)

def _predict_sklearn(model: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
    outputs: list[np.ndarray] = []
    transform = model['transform']
    batch_size = int(model.get('predict_batch_size', _PREDICT_BATCH_SIZE))
    if batch_size < 1:
        raise ValueError('predict_batch_size must be positive')
    for raw in _linear_batches(arrays, transform, batch_size):
        x = np.clip((raw - transform['mean']) / transform['scale'], -12.0, 12.0)
        outputs.append(_positive_probability(model['estimator'], x))
    return np.clip(np.concatenate(outputs) if outputs else np.empty(0), 0.0, 1.0)

class _AgreementMLP(nn.Module):

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        mode = config['feature_mode']
        self.g_projection = nn.Sequential(nn.LayerNorm(G_DIM), nn.Linear(G_DIM, int(config['g_projection_dim'])), nn.GELU()) if mode in ('tab_g', 'tab_gl') else None
        self.l_projection = nn.Sequential(nn.LayerNorm(L_DIM), nn.Linear(L_DIM, int(config['l_projection_dim'])), nn.GELU()) if mode in ('tab_l', 'tab_gl') else None
        input_width = TAB_DIM
        input_width += int(config['g_projection_dim']) if self.g_projection is not None else 0
        input_width += int(config['l_projection_dim']) if self.l_projection is not None else 0
        width = int(config['width'])
        self.fusion = nn.Sequential(nn.Linear(input_width, width), nn.GELU(), nn.Dropout(float(config['dropout'])), nn.Linear(width, 1))

    def forward(self, tab: torch.Tensor, g: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        pieces = [tab]
        if self.g_projection is not None:
            pieces.append(self.g_projection(g))
        if self.l_projection is not None:
            pieces.append(self.l_projection(l))
        return self.fusion(torch.cat(pieces, dim=1)).squeeze(1)

def _mlp_features(arrays: dict[str, np.ndarray], rows: slice | np.ndarray, transform: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tab = torch.as_tensor(np.asarray(arrays['tab'][rows], dtype=np.float32), device=device)
    mean = torch.as_tensor(transform['mean'], dtype=torch.float32, device=device)
    scale = torch.as_tensor(transform['scale'], dtype=torch.float32, device=device)
    tab = ((tab - mean) / scale).clamp(-12.0, 12.0)
    g = torch.as_tensor(np.asarray(arrays['g'][rows], dtype=np.float32), device=device)
    l = torch.as_tensor(np.asarray(arrays['l'][rows], dtype=np.float32), device=device)
    return (tab, g, l)

def _fit_mlp(config: dict[str, Any], arrays: dict[str, np.ndarray], y: np.ndarray, weight: np.ndarray, seed: int, device: str) -> dict[str, Any]:
    torch_device = torch.device(device)
    if torch_device.type == 'cuda' and (not torch.cuda.is_available()):
        raise ValueError('CUDA device requested but unavailable')
    tab = np.asarray(arrays['tab'], dtype=np.float64)
    mean = np.average(tab, axis=0, weights=weight)
    variance = np.average((tab - mean) ** 2, axis=0, weights=weight)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-07] = 1.0
    transform = {'mean': mean.astype(np.float32), 'scale': scale.astype(np.float32), 'g_dim': G_DIM, 'l_dim': L_DIM}
    torch.manual_seed(int(seed))
    if torch_device.type == 'cuda':
        torch.cuda.manual_seed_all(int(seed))
    network = _AgreementMLP(config).to(torch_device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
    rng = np.random.default_rng(int(seed))
    count = len(y)
    batch_size = min(int(config['batch_size']), count)
    steps = int(config['steps'])
    if batch_size < 1 or steps < 1:
        raise ValueError('MLP batch_size and steps must be positive')
    target_all = y.astype(np.float32)
    sample_probability = weight / weight.sum()
    network.train()
    final_loss = float('nan')
    for _ in range(steps):
        index = rng.choice(count, size=batch_size, replace=True, p=sample_probability)
        x = _mlp_features(arrays, index, transform, torch_device)
        target = torch.as_tensor(target_all[index], device=torch_device)
        original_logits = network(*x)
        logits = original_logits
        used_target = target
        mixup = float(config.get('mixup', 0.0))
        if mixup > 0.0:
            permutation = torch.as_tensor(rng.permutation(batch_size), device=torch_device)
            lam = float(rng.beta(mixup, mixup))
            mixed = tuple((lam * value + (1.0 - lam) * value[permutation] for value in x))
            logits = network(*mixed)
            used_target = lam * target + (1.0 - lam) * target[permutation]
        loss = F.binary_cross_entropy_with_logits(logits, used_target)
        ranking_weight = float(config.get('ranking_weight', 0.0))
        if ranking_weight > 0.0:
            positive = torch.where(target == 1)[0]
            negative = torch.where(target == 0)[0]
            if len(positive) and len(negative):
                pairs = min(len(positive), len(negative), 64)
                pindex = torch.as_tensor(rng.integers(0, len(positive), pairs), device=torch_device)
                nindex = torch.as_tensor(rng.integers(0, len(negative), pairs), device=torch_device)
                ranking = F.softplus(-(original_logits[positive[pindex]] - original_logits[negative[nindex]])).mean()
                loss = loss + ranking_weight * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return {'kind': 'mlp', 'family': 'mlp', 'config': copy.deepcopy(config), 'transform': transform, 'state_dict': {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}, 'training_diagnostics': {'seed': int(seed), 'training_rows': count, 'sample_weight_sum': float(weight.sum()), 'sampling': 'sample_weight_probability_with_replacement', 'sample_probability_sum': float(sample_probability.sum()), 'steps': steps, 'batch_size': batch_size, 'final_loss': final_loss, 'transform_fit': 'training_rows_only', 'inference_device': 'cpu'}}

def _predict_mlp(model: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
    device = torch.device('cpu')
    network = _AgreementMLP(model['config']).to(device)
    network.load_state_dict(model['state_dict'])
    network.eval()
    outputs: list[np.ndarray] = []
    batch_size = int(model['config'].get('predict_batch_size', _PREDICT_BATCH_SIZE))
    if batch_size < 1:
        raise ValueError('predict_batch_size must be positive')
    with torch.inference_mode():
        for start in range(0, len(arrays['tab']), batch_size):
            rows = slice(start, min(start + batch_size, len(arrays['tab'])))
            values = _mlp_features(arrays, rows, model['transform'], device)
            outputs.append(network(*values).sigmoid().double().numpy())
    return np.clip(np.concatenate(outputs) if outputs else np.empty(0), 0.0, 1.0)

def fit_predict(config: dict[str, Any], train: dict[str, Any], y: Any, predict: dict[str, Any], sample_weight: Any=None, seed: int=42, device: str='cpu') -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one candidate and return probabilities plus a joblib-safe record."""
    config = copy.deepcopy(config)
    training = _validate_inputs(train, name='train')
    prediction = _validate_inputs(predict, name='predict')
    target, weight = _targets_and_weights(y, sample_weight, len(training['tab']))
    if not len(target):
        raise ValueError('at least one training row is required')
    family = config.get('family')
    if family == 'baseline':
        model = {'kind': 'baseline', 'family': family, 'config': config}
        return (predict_model(model, prediction), model)
    effective = target[weight > 0]
    if np.unique(effective).size < 2:
        probability = float((np.dot(weight, target) + 0.5) / (weight.sum() + 1.0))
        model = {'kind': 'constant', 'family': family, 'config': config, 'probability': probability, 'training_diagnostics': {'seed': int(seed), 'training_rows': len(target), 'sample_weight_sum': float(weight.sum()), 'transform_fit': 'not_required_single_class'}}
        return (predict_model(model, prediction), model)
    if family == 'mlp':
        model = _fit_mlp(config, training, target, weight, int(seed), device)
        return (predict_model(model, prediction), model)
    if family not in ('lr', 'extra_trees', 'hgb'):
        raise ValueError(f'unknown model family: {family}')
    mode = str(config.get('feature_mode', 'tab'))
    if mode not in ('prob', 'tab', 'tab_g', 'tab_l', 'tab_gl'):
        raise ValueError(f'unknown feature mode: {mode}')
    projection_seed = int(config.get('projection_seed', _PROJECTION_SEED))
    transform, x = _fit_linear_transform(training, mode, weight, projection_seed)
    labels = target.astype(np.int64)
    if family == 'lr':
        estimator = LogisticRegression(C=float(config['C']), solver='lbfgs', max_iter=2000, random_state=int(seed))
    elif family == 'extra_trees':
        estimator = ExtraTreesClassifier(n_estimators=int(config['n_estimators']), max_depth=int(config['max_depth']), min_samples_leaf=int(config['min_samples_leaf']), max_features='sqrt', random_state=int(seed), n_jobs=1)
    else:
        estimator = HistGradientBoostingClassifier(max_iter=int(config['max_iter']), learning_rate=float(config['learning_rate']), max_depth=int(config['max_depth']), l2_regularization=float(config['l2_regularization']), min_samples_leaf=int(config['min_samples_leaf']), early_stopping=False, random_state=int(seed))
    estimator.fit(x, labels, sample_weight=weight)
    model = {'kind': 'sklearn', 'family': family, 'config': config, 'transform': transform, 'estimator': estimator, 'predict_batch_size': int(config.get('predict_batch_size', _PREDICT_BATCH_SIZE)), 'training_diagnostics': {'seed': int(seed), 'training_rows': len(target), 'sample_weight_sum': float(weight.sum()), 'transform_fit': 'training_rows_only_weighted'}}
    return (predict_model(model, prediction), model)

def predict_model(model: dict[str, Any], inputs: dict[str, Any]) -> np.ndarray:
    """Replay probabilities from a fitted model record using bounded batches."""
    arrays = _validate_inputs(inputs, name='predict')
    kind = model.get('kind')
    if kind == 'baseline':
        return _baseline(str(model['config']['kind']), arrays['tab'])
    if kind == 'constant':
        return np.full(len(arrays['tab']), float(model['probability']), dtype=np.float64)
    if kind == 'sklearn':
        return _predict_sklearn(model, arrays)
    if kind == 'mlp':
        return _predict_mlp(model, arrays)
    raise ValueError(f'unknown fitted model kind: {kind}')
