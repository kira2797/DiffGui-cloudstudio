import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from models.common import GaussianSmearing, MLP


class FrequencySurrogate(nn.Module):
    """Differentiable surrogate from 3D ligand geometry to an xTB-like spectrum."""

    def __init__(
        self,
        atom_feat_dim,
        spectrum_dim,
        hidden_dim=128,
        distance_dim=48,
        distance_cutoff=12.0,
    ):
        super().__init__()
        self.atom_feat_dim = atom_feat_dim
        self.spectrum_dim = spectrum_dim
        self.hidden_dim = hidden_dim
        self.distance_dim = distance_dim
        self.distance_cutoff = distance_cutoff

        self.atom_emb = nn.Linear(atom_feat_dim, hidden_dim)
        self.distance_expansion = GaussianSmearing(
            start=0.0,
            stop=distance_cutoff,
            num_gaussians=distance_dim,
            type_="linear",
        )
        self.pair_mlp = MLP(
            hidden_dim * 3 + distance_dim,
            hidden_dim,
            hidden_dim,
            num_layer=3,
            act_fn="silu",
        )
        self.out_mlp = MLP(
            hidden_dim,
            spectrum_dim,
            hidden_dim,
            num_layer=3,
            act_fn="silu",
            norm=False,
        )

    def _pair_index(self, batch, num_graphs):
        left, right, pair_batch = [], [], []
        for graph_idx in range(num_graphs):
            node_idx = (batch == graph_idx).nonzero().view(-1)
            if node_idx.numel() < 2:
                continue
            local_left, local_right = torch.triu_indices(
                node_idx.numel(),
                node_idx.numel(),
                offset=1,
                device=batch.device,
            )
            left.append(node_idx[local_left])
            right.append(node_idx[local_right])
            pair_batch.append(torch.full_like(local_left, graph_idx))
        if not left:
            empty = torch.empty(0, dtype=torch.long, device=batch.device)
            return empty, empty, empty
        return torch.cat(left), torch.cat(right), torch.cat(pair_batch)

    def forward(self, node_h, pos, batch, num_graphs=None):
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if node_h.dtype in (torch.long, torch.int64):
            node_h = F.one_hot(node_h, num_classes=self.atom_feat_dim).float()
        else:
            node_h = node_h.float()

        node_emb = self.atom_emb(node_h)
        left, right, pair_batch = self._pair_index(batch, num_graphs)
        if left.numel() == 0:
            graph_h = torch.zeros(
                num_graphs,
                self.hidden_dim,
                dtype=node_emb.dtype,
                device=node_emb.device,
            )
        else:
            dist = torch.norm(pos[left] - pos[right], dim=-1)
            dist_emb = self.distance_expansion(dist)
            pair_feat = torch.cat(
                [
                    node_emb[left],
                    node_emb[right],
                    node_emb[left] * node_emb[right],
                    dist_emb,
                ],
                dim=-1,
            )
            pair_h = self.pair_mlp(pair_feat)
            graph_h = scatter_sum(pair_h, pair_batch, dim=0, dim_size=num_graphs)

        spectrum = F.softplus(self.out_mlp(graph_h))
        return F.normalize(spectrum, p=2, dim=-1)


def load_target_spectrum(path, device):
    with open(path, "r") as f:
        data = json.load(f)
    if "spectrum" in data:
        values = data["spectrum"]
    elif "target_spectrum" in data:
        values = data["target_spectrum"]
    else:
        raise ValueError("Target spectrum JSON must contain `spectrum`.")
    spectrum = torch.tensor(values, dtype=torch.float32, device=device)
    return F.normalize(spectrum, p=2, dim=0)


class FrequencyGuidance(object):
    """Position guidance that increases surrogate-predicted spectrum similarity."""

    def __init__(
        self,
        surrogate,
        target_spectrum,
        scale=1e-4,
        start_step=0,
        end_step=None,
        max_delta_norm=0.05,
    ):
        self.surrogate = surrogate
        self.target_spectrum = target_spectrum
        self.scale = scale
        self.start_step = start_step
        self.end_step = end_step
        self.max_delta_norm = max_delta_norm

    def active(self, step):
        if step < self.start_step:
            return False
        if self.end_step is not None and step > self.end_step:
            return False
        return self.scale > 0

    def __call__(self, node_h, pos, batch, step, num_graphs):
        if not self.active(step):
            return torch.zeros_like(pos)

        pos_in = pos.detach().requires_grad_(True)
        node_in = node_h.detach()
        pred = self.surrogate(node_in, pos_in, batch, num_graphs=num_graphs)
        target = self.target_spectrum.view(1, -1).expand_as(pred)
        score = F.cosine_similarity(pred, target, dim=-1).sum()
        grad = torch.autograd.grad(score, pos_in, retain_graph=False)[0]

        # Remove graph-wise translation so the guidance changes shape, not center.
        grad = grad - scatter_mean(grad, batch, dim=0, dim_size=num_graphs)[batch]
        delta = grad * self.scale
        if self.max_delta_norm is not None and self.max_delta_norm > 0:
            norm = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-12)
            scale = (self.max_delta_norm / norm).clamp_max(1.0)
            delta = delta * scale
        return delta
