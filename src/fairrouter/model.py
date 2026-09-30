"""Multi-modal residual classifier and adaptive fusion (paper Section 3.3)."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class UnifiedRepair(nn.Module):
    def __init__(self, gnn_dim, llm_dim, classes, dim, width=64):
        super().__init__()
        if min(gnn_dim, llm_dim, classes, dim, width) < 1:
            raise ValueError("All dimensions must be positive")
        self.g_norm = nn.LayerNorm(gnn_dim)
        self.l_norm = nn.LayerNorm(llm_dim)
        self.g_project = nn.Linear(gnn_dim, dim)
        self.l_project = nn.Linear(llm_dim, dim)
        self.hidden = nn.Linear(2 * dim, width)
        self.output = nn.Linear(width, classes)
        # Default bounded random initialization supplies projection gradients immediately.
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.raw_beta = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def coefficients(self):
        return F.softplus(self.raw_alpha), F.softplus(self.raw_beta)

    def weights(self):
        a, b = self.coefficients()
        return torch.stack((a, b, a.new_ones(()))) / (a + b + 1)

    def forward(self, g, llm_hidden, logg, logl, detach_gate=False):
        g = g.detach().float()
        llm_hidden = llm_hidden.detach().float()
        logg = logg.detach().float()
        logl = logl.detach().float()
        ug = F.gelu(self.g_project(self.g_norm(g)))
        ul = F.gelu(self.l_project(self.l_norm(llm_hidden)))
        r = F.log_softmax(self.output(F.gelu(self.hidden(torch.cat((ug, ul), -1)))), dim=-1)
        a, b = self.coefficients()
        if detach_gate:
            a, b = a.detach(), b.detach()
        terms = torch.stack((logg + a.clamp_min(1e-30).log(), logl + b.clamp_min(1e-30).log(), r))
        logp = torch.logsumexp(terms, dim=0) - torch.log1p(a + b)
        return logp, r
