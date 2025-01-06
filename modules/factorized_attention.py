import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Union, Tuple, List, Optional
from modules.siren_module import SirenNet, SirenWrapper
from modules.embedding import RotaryEmbedding, apply_rotary_pos_emb

class LowRankKernel(nn.Module):
    # low rank kernel, ideally operates only on one dimension
    def __init__(self,
                 dim,
                 dim_head,
                 heads,  # only used for calculating positional embedding
                 use_rotary_emb=False,
                 dropout=0,
                 scaling=1,
                 qk_norm=False,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dim_head = dim_head
        self.heads = heads
        if dropout > 1e-6:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.to_qk = nn.Linear(dim, dim_head*heads*2, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim_head, elementwise_affine=False)
            self.k_norm = nn.LayerNorm(dim_head, elementwise_affine=False)

        self.use_rotary_emb = use_rotary_emb
        if use_rotary_emb:
            self.pos_emb = RotaryEmbedding(dim_head)

        self.scaling = scaling

    def forward(self, x):
        # u_x: b n c
        # pos: b n d

        n = x.shape[1]
        pos = torch.linspace(0, 1, n, device=x.device).view(1, n)   # simpler way to encode position

        qk = self.to_qk(x)
        q, k = qk.split(qk.shape[-1]//2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rotary_emb:
            freqs = self.pos_emb.forward(pos, q.device)
            freqs = repeat(freqs, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

        K = torch.einsum('bhid,bhjd->bhij', q, k) * self.scaling
        K = self.dropout(K)
        return K


class PoolingReducer(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super().__init__()
        self.to_in = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim*2, out_dim, bias=True),
        )

    def forward(self, x):
        # note that the dimension to be pooled will be the last dimension
        # x: b c nx ...
        x = self.to_in(rearrange(x, 'b c ... -> b ... c'))
        # pool all spatial dimension but the first one
        ndim = len(x.shape)
        x = x.mean(dim=tuple(range(2, ndim-1)))
        x = self.out_ffn(x)
        return x  # b nx c


class FABlock2D(nn.Module):
    # contains factorization and attention on each axis
    def __init__(self,
                 dim,
                 dim_head,
                 latent_dim,
                 heads,
                 dim_out,
                 use_rope=True,
                 kernel_multiplier=2,
                 qk_norm=False):
        super().__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.in_norm = nn.GroupNorm(1, dim)
        self.in_proj = nn.Conv2d(self.dim, heads * dim_head, 1, 1, 0, bias=False)

        self.to_in = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=False),
        )

        self.to_x = nn.Sequential(
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )
        self.to_y = nn.Sequential(
            Rearrange('b c nx ny -> b c ny nx'),
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )

        self.low_rank_kernel_x = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier,
                                               heads,
                                               use_rotary_emb=use_rope,
                                               qk_norm=qk_norm)

        self.low_rank_kernel_y = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier,
                                               heads,
                                               use_rotary_emb=use_rope,
                                               qk_norm=qk_norm)

        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(dim_head * heads),
            nn.Conv2d(dim_head * heads, dim_out, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, 1, 1, 0, bias=False))

    def forward(self, u,):
        # x: b c h w
        u_skip = u
        u = self.in_norm(u)
        u_phi = self.in_proj(u)
        u = self.to_in(u)
        u_x = self.to_x(u)
        u_y = self.to_y(u)

        k_x = self.low_rank_kernel_x(u_x)
        k_y = self.low_rank_kernel_y(u_y)

        u_phi = rearrange(u_phi, 'b (h c) i l -> b h c i l', h=self.heads)
        u_phi = torch.einsum('bhij,bhcjm->bhcim', k_x, u_phi)
        u_phi = torch.einsum('bhlm,bhcim->bhcil', k_y, u_phi)
        u_phi = rearrange(u_phi, 'b h c i l -> b (h c) i l', h=self.heads)
        return self.to_out(u_phi) + u_skip