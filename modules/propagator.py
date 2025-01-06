import torch
import torch.nn as nn
from modules.basics import ResidualBlock, SABlock, CABlock, \
    DownSampleBlock, UpSampleBlock, GroupNorm, Swish, FourierBasicBlock
from einops import rearrange
from einops.layers.torch import Rearrange


class SimpleResNet(nn.Module):
    def __init__(self, args):
        super(SimpleResNet, self).__init__()
        is_periodic = args.is_periodic
        if is_periodic:
            padding_mode = 'circular'
        else:
            padding_mode = 'zeros'
        self.net = nn.Sequential(
            nn.Conv2d(args.latent_dim, args.propagator_dim, 1, 1, 0),
            Swish(),
            nn.Conv2d(args.propagator_dim, args.propagator_dim, 3, 1, 1, padding_mode=padding_mode),
            GroupNorm(args.propagator_dim),
            ResidualBlock(args.propagator_dim, args.propagator_dim, padding_mode=padding_mode),
            ResidualBlock(args.propagator_dim, args.propagator_dim, padding_mode=padding_mode),
            ResidualBlock(args.propagator_dim, args.propagator_dim, padding_mode=padding_mode),
            GroupNorm(args.propagator_dim),
            Swish(),
            nn.Conv2d(args.propagator_dim, args.latent_dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x)


class SimpleMLP(nn.Module):
    def __init__(self, args):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(args.latent_dim*args.latent_resolution**2, args.propagator_dim),
            Swish(),
            nn.Linear(args.propagator_dim, args.propagator_dim),
            Swish(),
            nn.Linear(args.propagator_dim, args.latent_dim*args.latent_resolution**2)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b 1 (h w c)')
        dx = self.net(x)
        x = x + dx
        x = rearrange(x, 'b 1 (h w c) -> b c h w', h=h, w=w)
        return x

class ConditionalResNet(nn.Module):
    def __init__(self, args):
        super(ConditionalResNet, self).__init__()
        is_periodic = args.is_periodic
        if is_periodic:
            padding_mode = 'circular'
        else:
            padding_mode = 'zeros'

        self.num_blocks = args.propagator_num_blocks
        self.dim = args.propagator_dim
        self.context_emb_dim = args.gpt_n_embd
        self.propagator_ca_heads = args.propagator_ca_heads
        self.propagator_ca_dim_head = args.propagator_ca_dim_head

        self.use_sa = args.propagator_use_sa
        if self.use_sa:
            self.propagator_sa_heads = args.propagator_sa_heads
            self.propagator_sa_dim_head = args.propagator_sa_dim_head

        self.to_in = nn.Sequential(
            nn.Conv2d(args.latent_dim, self.dim, 3, 1, 1, padding_mode=padding_mode),
            GroupNorm(self.dim),
        )

        layers = []
        for nb in range(self.num_blocks):
            # cross attention + resnet
            layer = nn.ModuleList([])
            if self.use_sa:
                layer.append(SABlock(self.dim, self.propagator_sa_heads, self.propagator_sa_dim_head))
            layer.append(CABlock(self.dim,
                                 self.context_emb_dim,
                                 self.propagator_ca_heads, self.propagator_ca_dim_head))
            layer.append(ResidualBlock(self.dim, self.dim, padding_mode=padding_mode))

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        self.to_out = nn.Sequential(
            GroupNorm(self.dim),
            Swish(),
            nn.Conv2d(self.dim, args.latent_dim, 3, 1, 1, padding_mode=padding_mode))

    def forward(self, x, c):
        b, _, h, w = x.shape
        x = self.to_in(x)

        for nb in range(self.num_blocks):
            # cross attention + resnet
            if not self.use_sa:
                x = self.layers[nb][0](x, c)
                x = self.layers[nb][1](x)
            else:
                x = self.layers[nb][0](x)
                x = self.layers[nb][1](x, c)
                x = self.layers[nb][2](x)
        x = self.to_out(x)
        return x