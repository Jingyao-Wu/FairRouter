from __future__ import annotations
import math
import torch
EPS = 1e-12
PROBABILITY_CANDIDATE_NAMES = ['gnn_probability', 'gnn_log_probability', 'llm_probability', 'llm_log_probability', 'gnn_rank_normalized', 'llm_rank_normalized', 'gnn_top1_probability_gap', 'llm_top1_probability_gap', 'gnn_candidate_minus_top2_probability', 'llm_candidate_minus_top2_probability', 'gnn_entropy', 'gnn_top1_margin', 'gnn_top1_probability', 'llm_entropy', 'llm_top1_margin', 'llm_top1_probability']
PROBABILITY_GLOBAL_NAMES = ['expert_js_divergence', 'expert_bhattacharyya_overlap', 'expert_probability_l1', 'expert_probability_l2', 'expert_top1_probability_min', 'expert_top1_probability_max', 'expert_top1_probability_abs_difference', 'expert_entropy_sum', 'expert_entropy_abs_difference']
STRUCTURAL_CANDIDATE_NAMES = ['neighbor_gnn_vote_rate', 'neighbor_gnn_mean_probability', 'neighbor_gnn_mean_probability_missing']
STRUCTURAL_GLOBAL_NAMES = ['neighbor_prediction_entropy', 'log_degree', 'isolated_node']

def _validate_probabilities(probability: torch.Tensor, *, name: str) -> torch.Tensor:
    value = probability.detach().cpu().float()
    if value.ndim != 2 or value.size(1) < 2:
        raise ValueError(f'{name} must have shape [N, C] with C >= 2')
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError(f'{name} must contain finite nonnegative probabilities')
    sums = value.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-05, rtol=1e-05):
        raise ValueError(f'{name} rows must sum to one')
    return value

def _distribution_shape(probability: torch.Tensor) -> tuple[torch.Tensor, ...]:
    class_count = probability.size(1)
    sorted_probability = probability.sort(dim=1, descending=True).values
    entropy = -(probability * probability.clamp_min(EPS).log()).sum(dim=1)
    entropy = entropy / math.log(class_count)
    top1 = sorted_probability[:, 0]
    top2 = sorted_probability[:, 1]
    return (entropy, top1 - top2, top1, top2)

def _candidate_columns(gnn: torch.Tensor, llm: torch.Tensor, candidate: torch.Tensor, gnn_shape: tuple[torch.Tensor, ...], llm_shape: tuple[torch.Tensor, ...]) -> torch.Tensor:
    candidate = candidate.detach().cpu().view(-1).long()
    if candidate.numel() != gnn.size(0):
        raise ValueError('candidate row count differs from probability rows')
    if candidate.numel() and (int(candidate.min()) < 0 or int(candidate.max()) >= gnn.size(1)):
        raise ValueError('candidate class outside probability width')
    index = candidate[:, None]
    gnn_candidate = gnn.gather(1, index).view(-1)
    llm_candidate = llm.gather(1, index).view(-1)
    gnn_rank = (gnn > gnn_candidate[:, None]).sum(dim=1).float() / max(gnn.size(1) - 1, 1)
    llm_rank = (llm > llm_candidate[:, None]).sum(dim=1).float() / max(llm.size(1) - 1, 1)
    gnn_entropy, gnn_margin, gnn_top1, gnn_top2 = gnn_shape
    llm_entropy, llm_margin, llm_top1, llm_top2 = llm_shape
    return torch.stack([gnn_candidate, gnn_candidate.clamp_min(EPS).log(), llm_candidate, llm_candidate.clamp_min(EPS).log(), gnn_rank, llm_rank, gnn_top1 - gnn_candidate, llm_top1 - llm_candidate, gnn_candidate - gnn_top2, llm_candidate - llm_top2, gnn_entropy, gnn_margin, gnn_top1, llm_entropy, llm_margin, llm_top1], dim=1)

def probability_evidence(gnn_probability: torch.Tensor, llm_probability: torch.Tensor, candidate_g: torch.Tensor, candidate_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    gnn = _validate_probabilities(gnn_probability, name='gnn_probability')
    llm = _validate_probabilities(llm_probability, name='llm_probability')
    if gnn.shape != llm.shape:
        raise ValueError('expert probability shapes differ')
    gnn_shape = _distribution_shape(gnn)
    llm_shape = _distribution_shape(llm)
    g = _candidate_columns(gnn, llm, candidate_g, gnn_shape, llm_shape)
    l = _candidate_columns(gnn, llm, candidate_l, gnn_shape, llm_shape)
    mixture = 0.5 * (gnn + llm)
    js = 0.5 * ((gnn * (gnn.clamp_min(EPS).log() - mixture.clamp_min(EPS).log())).sum(dim=1) + (llm * (llm.clamp_min(EPS).log() - mixture.clamp_min(EPS).log())).sum(dim=1))
    overlap = torch.sqrt(gnn * llm).sum(dim=1)
    difference = gnn - llm
    gnn_entropy, _, gnn_top1, _ = gnn_shape
    llm_entropy, _, llm_top1, _ = llm_shape
    global_features = torch.stack([js, overlap, difference.abs().sum(dim=1), difference.square().sum(dim=1).sqrt(), torch.minimum(gnn_top1, llm_top1), torch.maximum(gnn_top1, llm_top1), (gnn_top1 - llm_top1).abs(), gnn_entropy + llm_entropy, (gnn_entropy - llm_entropy).abs()], dim=1)
    return (g, l, global_features, list(PROBABILITY_CANDIDATE_NAMES), list(PROBABILITY_GLOBAL_NAMES))

def _undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    edge = edge_index.detach().cpu().long()
    if edge.ndim != 2 or edge.size(0) != 2:
        raise ValueError('edge_index must have shape [2, E]')
    src = torch.cat([edge[0], edge[1]])
    dst = torch.cat([edge[1], edge[0]])
    if src.numel():
        if int(src.min()) < 0 or int(dst.min()) < 0 or int(src.max()) >= num_nodes or (int(dst.max()) >= num_nodes):
            raise ValueError('edge_index contains node IDs outside graph')
        keys = torch.unique(src * int(num_nodes) + dst)
        src = torch.div(keys, int(num_nodes), rounding_mode='floor')
        dst = keys.remainder(int(num_nodes))
    return (src, dst)

def structural_evidence(*, edge_index: torch.Tensor, full_gnn_pred: torch.Tensor, row_ids: torch.Tensor, candidate_g: torch.Tensor, candidate_l: torch.Tensor, num_nodes: int, num_classes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    prediction = full_gnn_pred.detach().cpu().view(-1).long()
    row_ids = row_ids.detach().cpu().view(-1).long()
    candidate_g = candidate_g.detach().cpu().view(-1).long()
    candidate_l = candidate_l.detach().cpu().view(-1).long()
    if prediction.numel() != int(num_nodes):
        raise ValueError('full_gnn_pred must cover the full graph')
    if not row_ids.shape == candidate_g.shape == candidate_l.shape:
        raise ValueError('structural row and candidate shapes differ')
    if prediction.numel() and (int(prediction.min()) < 0 or int(prediction.max()) >= int(num_classes)):
        raise ValueError('full_gnn_pred contains class outside configured class space')
    src, dst = _undirected_edges(edge_index, int(num_nodes))
    degree = torch.bincount(src, minlength=int(num_nodes)).float()
    counts = torch.zeros((int(num_nodes), int(num_classes)), dtype=torch.float32)
    if src.numel():
        counts.index_put_((src, prediction[dst]), torch.ones(src.numel()), accumulate=True)
    distribution = counts / degree.clamp_min(1.0)[:, None]
    entropy = -(distribution * distribution.clamp_min(EPS).log()).sum(dim=1)
    entropy = entropy / math.log(max(int(num_classes), 2))
    entropy = torch.where(degree > 0, entropy, torch.ones_like(entropy))

    def candidate_features(candidate: torch.Tensor) -> torch.Tensor:
        vote = distribution[row_ids, candidate]
        return torch.stack([vote, torch.zeros_like(vote), torch.ones_like(vote)], dim=1)
    g = candidate_features(candidate_g)
    l = candidate_features(candidate_l)
    row_degree = degree[row_ids]
    global_features = torch.stack([entropy[row_ids], torch.log1p(row_degree), (row_degree == 0).float()], dim=1)
    return (g, l, global_features, list(STRUCTURAL_CANDIDATE_NAMES), list(STRUCTURAL_GLOBAL_NAMES))
