import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.basics import ResidualBlock, SABlock, CABlock, \
    DownSampleBlock, UpSampleBlock, GroupNorm, Swish, FourierBasicBlock
from modules.factorized_attention import FABlock2D
from modules.propagator import SimpleResNet, SimpleMLP
from modules.cond_utils import CondResidualBlock, fourier_embedding
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.linalg import matrix_norm


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = args.encoder_channels
        fourier_resolutions = args.fourier_resolutions
        resolutions = args.resolutions
        latent_resolution = args.latent_resolution

        num_res_blocks = args.encoder_res_blocks
        resolution_height, resolution_width = resolutions[0], resolutions[1]
        hw_ratio = args.hw_ratio
        assert (len(channels) - 2) == int(math.log2(resolution_height//latent_resolution))

        is_periodic = args.is_periodic
        if is_periodic:
            padding_mode = 'circular'
        else:
            padding_mode = 'zeros'

        layers = [nn.Conv2d(args.in_channels, channels[0], 1, 1, 0),
                  Swish(),
                  nn.Conv2d(channels[0], channels[0], 3, 1, 1, padding_mode=padding_mode)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels, num_dimensions=2,
                                            padding_mode=padding_mode))
                in_channels = out_channels
                if resolution_height in fourier_resolutions:
                    # can filter out high-frequency components
                    if resolution_height <= 32:
                        layers.append(FourierBasicBlock(in_channels, out_channels,
                                                        modes=[6, int(6*hw_ratio)]))
                    else:
                        layers.append(FourierBasicBlock(in_channels, out_channels,
                                                        modes=[10, int(10*hw_ratio)]))
            if i != len(channels)-2:
                layers.append(DownSampleBlock(channels[i+1],
                                              num_dimensions=2, padding_mode=padding_mode))
                resolution_height //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1],
                                    num_dimensions=2, padding_mode=padding_mode))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 1, 1, 0, padding_mode=padding_mode))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # input has a shape of [b c h w]
        out = self.model(x)
        return out


class CondEncoder(nn.Module):
    def __init__(self, args):
        super(CondEncoder, self).__init__()
        channels = args.encoder_channels
        fourier_resolutions = args.fourier_resolutions
        resolutions = args.resolutions
        latent_resolution = args.latent_resolution

        num_res_blocks = args.encoder_res_blocks
        resolution_height, resolution_width = resolutions[0], resolutions[1]
        hw_ratio = args.hw_ratio
        assert (len(channels) - 2) == int(math.log2(resolution_height//latent_resolution))

        is_periodic = args.is_periodic
        if is_periodic:
            padding_mode = 'circular'
        else:
            padding_mode = 'zeros'

        self.cond_emb_channels = args.cond_emb_channels

        self.to_in = nn.Sequential(nn.Conv2d(args.in_channels, channels[0], 1, 1, 0),
                  Swish(),
                  nn.Conv2d(channels[0], channels[0], 3, 1, 1, padding_mode=padding_mode))

        self.embed = nn.Sequential(nn.Linear(args.cond_emb_channels, channels[0]),
                                    Swish(),
                                    nn.Linear(channels[0], args.cond_emb_channels))

        self.layers = nn.ModuleList()
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            layer = nn.ModuleList()
            res_layer = nn.ModuleList()
            for j in range(num_res_blocks):
                res_layer.append(CondResidualBlock(in_channels, out_channels,
                                                cond_channels=args.cond_emb_channels,
                                                norm=True,
                                                padding_mode=padding_mode))
                in_channels = out_channels
            layer.append(res_layer)
            if i != len(channels)-2:
                layer.append(DownSampleBlock(channels[i+1],
                                              num_dimensions=2, padding_mode=padding_mode))
                resolution_height //= 2
            self.layers.append(layer)

        self.to_out_conv = CondResidualBlock(channels[-1], channels[-1],
                                        cond_channels=args.cond_emb_channels,
                                        norm=True,
                                        padding_mode=padding_mode)

        self.to_out = nn.Sequential(GroupNorm(channels[-1]),
                                    Swish(),
                                    nn.Conv2d(channels[-1], args.latent_dim, 1, 1, 0, padding_mode=padding_mode))

    def forward(self, x, param):
        # input has a shape of [b c h w]
        cond_emb = self.embed(fourier_embedding(param, self.cond_emb_channels))
        out = self.to_in(x)
        for l, layer in enumerate(self.layers):
            if l < len(self.layers)-1:
                res_layer, downsample_layer = layer[0], layer[1]
                for res in res_layer:
                    out = res(out, cond_emb)
                out = downsample_layer(out)
            else:
                res_layer = layer[0]
                for res in res_layer:
                    out = res(out, cond_emb)
        out = self.to_out_conv(out, cond_emb)
        out = self.to_out(out)

        return out


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = args.decoder_channels
        attn_resolutions = args.attn_resolutions
        latent_resolution = args.latent_resolution
        is_periodic = args.is_periodic
        final_smoothing = args.final_smoothing

        if is_periodic:
            padding_mode = 'circular'
        else:
            padding_mode = 'zeros'
        num_res_blocks = args.decoder_res_blocks
        attn_heads = args.decoder_attn_heads
        attn_dim = args.decoder_attn_dim
        resolutions = args.resolutions

        in_channels = channels[0]
        resolution_height = latent_resolution
        hw_ratio = resolutions[1] / resolutions[0]

        disable_coarse_attn = True if args.disable_coarse_attn is not None and args.disable_coarse_attn \
            else False
        if not disable_coarse_attn:
            layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1, padding_mode=padding_mode),
                      ResidualBlock(in_channels, in_channels,
                                    num_dimensions=2,
                                    padding_mode=padding_mode),
                      SABlock(in_channels, attn_heads, attn_dim, use_pe=True,
                              block_size=resolution_height*int(resolution_height*(hw_ratio+0.5))),
                      ResidualBlock(in_channels, in_channels,
                                    num_dimensions=2,
                                    padding_mode=padding_mode)]
        else:
            layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1, padding_mode=padding_mode),
                      ResidualBlock(in_channels, in_channels,
                                    num_dimensions=2,
                                    padding_mode=padding_mode),
                      ResidualBlock(in_channels, in_channels,
                                    num_dimensions=2,
                                    padding_mode=padding_mode)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels,
                                            num_dimensions=2,
                                            padding_mode=padding_mode))
                in_channels = out_channels
                if resolution_height in attn_resolutions:
                    if args.use_fa:
                        layers.append(FABlock2D(in_channels,
                                                attn_dim,
                                                attn_dim,
                                                attn_heads,
                                                in_channels,
                                                use_rope=True,
                                                kernel_multiplier=2))
                    else:
                        layers.append(SABlock(in_channels, attn_heads, attn_dim,
                                              use_pe=True,
                                              block_size=
                                              resolution_height*int(resolution_height*(hw_ratio+0.5))))

            if i != 0 and i != len(channels) - 1:
                layers.append(UpSampleBlock(in_channels,
                                            num_dimensions=2,
                                            padding_mode=padding_mode))
                resolution_height *= 2
        layers.append(nn.Upsample(size=(args.Ly, args.Lx), mode='nearest'))
        resolution_height = args.Ly
        layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1, padding_mode=padding_mode))  # resample with conv
        if final_smoothing:
            layers.append(FourierBasicBlock(in_channels, in_channels,
                            modes=[16, int(16*hw_ratio)]))    # this assumes at least 64*64 output
        else:
            if resolution_height in attn_resolutions:
                if args.use_fa:
                    layers.append(FABlock2D(in_channels,
                                            attn_dim,
                                            attn_dim,
                                            attn_heads,
                                            in_channels,
                                            use_rope=True,
                                            kernel_multiplier=2))
                else:
                    layers.append(SABlock(in_channels, attn_heads, attn_dim,
                                          use_pe=True,
                                          block_size=
                                          resolution_height * int(resolution_height * (hw_ratio + 0.5))))
            layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1, padding_mode=padding_mode))
        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.in_channels, 1, 1, 0))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class SimpleAutoencoder(nn.Module):
    def __init__(self, args):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def encode(self, x):
        encoded_x = self.encoder(x)
        quant_conv_encoded_x = self.quant_conv(encoded_x)
        return quant_conv_encoded_x

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def load_checkpoint(self, path, device=None):
        ckpt = torch.load(path, map_location=device)
        self.load_state_dict(ckpt, strict=True)


class ConditionalSimpleAutoencoder(nn.Module):
    def __init__(self, args):
        super(ConditionalSimpleAutoencoder, self).__init__()
        self.encoder = CondEncoder(args)
        self.decoder = Decoder(args)

        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)

    def forward(self, x, param):
        z = self.encode(x, param)
        x_hat = self.decode(z)
        return x_hat

    def encode(self, x, param):
        encoded_x = self.encoder(x, param)
        quant_conv_encoded_x = self.quant_conv(encoded_x)
        return quant_conv_encoded_x

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def load_checkpoint(self, path):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt, strict=True)
