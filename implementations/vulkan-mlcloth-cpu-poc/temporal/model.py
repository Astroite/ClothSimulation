"""Four-observation shared encoder; body/cloth node count is not a model parameter."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from real_scene.tinyhood import TinyHood, load_tinyhood
from real_scene.fine15 import aggregate_sum

WINDOW, CHANNELS, HZ = 4, 7, 30

@dataclass
class GnnHistoryState:
    samples: torch.Tensor | None = None
    ticks: int = 0

    def reset(self):
        self.samples, self.ticks = None, 0

    def push(self, displacement, normal):
        if displacement.shape != normal.shape or displacement.ndim != 2 or displacement.shape[1] != 3:
            raise ValueError("history needs [N,3] displacement and normal")
        latest = torch.cat((displacement, normal, torch.ones_like(displacement[:, :1])), -1)
        if self.samples is None:
            missing = torch.cat((latest[:, :6], torch.zeros_like(latest[:, 6:])), -1)
            self.samples = torch.stack((missing, missing, missing, latest), 1)
        else:
            if self.samples.shape != (len(latest), WINDOW, CHANNELS):
                raise ValueError("topology changed without resetting history")
            self.samples = torch.cat((self.samples[:, 1:], latest[:, None]), 1)
        self.ticks += 1
        return self.samples

    def detach(self):
        if self.samples is not None:
            self.samples = self.samples.detach()

    def clone(self):
        return GnnHistoryState(None if self.samples is None else self.samples.clone(), self.ticks)

def normalize_history(history, mean, std, current_only=False):
    if history.ndim != 3 or history.shape[1:] != (4, 7):
        raise ValueError("history contract is [nodes,4,7]")
    mean,std=mean.reshape(-1),std.reshape(-1)
    indices = torch.tensor([0, 1, 2, 12, 13, 14], device=history.device)
    normalized = (history[..., :6] - mean[indices]) / std[indices]
    result = torch.cat((normalized, history[..., 6:]), -1)
    if current_only:
        result = result[:, -1:].expand(-1, 4, -1)
    return result.reshape(-1, 28)

class TemporalTinyHood(TinyHood):
    def __init__(self, current_only=False):
        super().__init__(latent=32, blocks=12)
        self.temporal_encoder = nn.Sequential(nn.Linear(28,32), nn.ReLU(), nn.Linear(32,32))
        nn.init.zeros_(self.temporal_encoder[2].weight)
        nn.init.zeros_(self.temporal_encoder[2].bias)
        self.current_only = bool(current_only)
        self.checkpoint_blocks = False

    @classmethod
    def from_legacy(cls, path, current_only=False):
        old = load_tinyhood(path)
        if old.latent != 32 or len(old.processor_steps) != 12:
            raise ValueError("v4 requires a 32x12 backbone")
        model = cls(current_only)
        model.load_state_dict(old.state_dict(), strict=False)
        return model

    def forward(self, graph, history):
        n, m = len(graph.cloth_nodes), len(graph.obstacle_nodes)
        if history.shape != (n + m, 28):
            raise ValueError("normalized history must cover cloth then body nodes")
        if self.current_only:
            history = history.reshape(-1, 4, 7)[:, -1:].expand(-1,4,-1).reshape(-1,28)
        h = self.node_encoder(torch.cat((graph.cloth_nodes, graph.obstacle_nodes))) + self.temporal_encoder(history)
        c, b = h[:n], h[n:]
        e = self.mesh_encoder(graph.mesh_edges)
        d, r = self.world_encoder(graph.direct_world), self.world_encoder(graph.inverse_world)
        for block in self.processor_steps:
            def step(c,b,e,d,r, block=block):
                em = block.mesh_edge_processor(torch.cat((c[graph.mesh_receivers],c[graph.mesh_senders],e),-1))
                dm = block.world_edge_processor(torch.cat((b[graph.world_obstacle],c[graph.world_cloth],d),-1))
                rm = block.world_edge_processor(torch.cat((c[graph.world_cloth],b[graph.world_obstacle],r),-1))
                ca = aggregate_sum(em,graph.mesh_receivers,n)
                cw = aggregate_sum(rm,graph.world_cloth,n)
                bw = aggregate_sum(dm,graph.world_obstacle,m)
                return (c+block.node_processor(torch.cat((ca,cw,c),-1)),
                        b+block.node_processor(torch.cat((torch.zeros_like(bw),bw,b),-1)), e+em,d+dm,r+rm)
            if self.checkpoint_blocks and self.training:
                c,b,e,d,r = checkpoint(step,c,b,e,d,r,use_reentrant=False)
            else:
                c,b,e,d,r = step(c,b,e,d,r)
        return self.decoder(c)
