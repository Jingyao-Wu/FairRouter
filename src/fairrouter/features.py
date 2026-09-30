import torch
from eargtc.router_v8.evidence import probability_evidence, structural_evidence
from eargtc.router_v8.prototypes import prototype_evidence


def build70(pg, pl, ids, edge, full_pred, embedding, fitted, n, c):
    g, llm_candidate = (pg.argmax(1), pl.argmax(1))
    p = probability_evidence(pg, pl, g, llm_candidate)
    s = structural_evidence(
        edge_index=edge,
        full_gnn_pred=full_pred,
        row_ids=ids,
        candidate_g=g,
        candidate_l=llm_candidate,
        num_nodes=n,
        num_classes=c,
    )
    r = prototype_evidence(
        embeddings=embedding, row_ids=ids, candidate_g=g, candidate_l=llm_candidate, fitted=fitted
    )
    return torch.cat([torch.cat((p[i], s[i], r[i]), 1) for i in range(3)], 1)
