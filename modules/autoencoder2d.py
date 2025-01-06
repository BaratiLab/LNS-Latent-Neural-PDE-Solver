import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.basics import ResidualBlock, SABlock, CABlock, \
    DownSampleBlock, UpSampleBlock, GroupNorm, Swish, FourierBasicBlock
from modules.propagator import SimpleResNet, SimpleMLP
from modules.factorized_attention import FABlock2D
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.linalg import matrix_norm


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = args.encoder_channels
        fourier_resolutions = args.fourier_resolutions
        resolution = args.resolution
        latent_resolution = args.latent_resolution

        attn_resolutions = args.attn_resolutions

        assert (len(channels) - 2) == int(math.log2(resolution//latent_resolution))
        num_res_blocks = args.encoder_res_blocks


        layers = [nn.Conv2d(args.in_channels, channels[0], 1, 1, 0),
                  Swish(),
                  nn.Conv2d(channels[0], channels[0], 3, 1, 1, padding_mode=padding_mode)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels,
                                            num_dimensions=2,
                                            padding_mode=padding_mode))
                in_channels = out_channels

            if resolution in attn_resolutions and args.use_attn_enc:
                if not args.use_fa:
                    layers.append(SABlock(in_channels, args.attn_heads, args.attn_dim,
                                              use_pe=True, block_size=resolution ** 2))
                else:
                    layers.append(FABlock2D(in_channels, args.attn_dim, args.attn_dim,
                                                args.attn_heads, in_channels))

            if resolution in fourier_resolutions:
                # can filter out high-frequency components
                if resolution <= 32:
                    layers.append(FourierBasicBlock(in_channels, out_channels,
                                                    modes=[6, 6]))
                else:
                    layers.append(FourierBasicBlock(in_channels, out_channels,
                                                    modes=[10, 10]))
            if i != len(channels)-2:
                layers.append(DownSampleBlock(channels[i+1],
                                              num_dimensions=2,
                                              padding_mode=padding_mode))
                resolution //= 2
        layers.append(nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, padding_mode=padding_mode))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 1, 1, 0))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # input has a shape of [b c h w]
        out = self.model(x)
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
        attn_heads = args.attn_heads
        attn_dim = args.attn_dim
        disable_coarse_attn = True if args.disable_coarse_attn is not None and args.disable_coarse_attn\
            else False

        in_channels = channels[0]
        resolution = latent_resolution
        if not disable_coarse_attn:
            layers = [nn.Conv2d(args.latent_dim, in_channels, 1, 1, 0),
                      ResidualBlock(in_channels, in_channels,
                                    num_dimensions=2,
                                    padding_mode=padding_mode),
                      SABlock(in_channels, attn_heads, attn_dim, use_pe=True, block_size=resolution ** 2),
                      ResidualBlock(in_channels, in_channels,
                                    num_dimensions=2,
                                    padding_mode=padding_mode)]
        else:
            layers = [nn.Conv2d(args.latent_dim, in_channels, 1, 1, 0),
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
            if resolution in attn_resolutions:
                if not args.use_fa:
                    layers.append(SABlock(in_channels, args.attn_heads, args.attn_dim,
                                              use_pe=True, block_size=resolution ** 2))
                else:
                    layers.append(FABlock2D(in_channels, args.attn_dim, args.attn_dim,
                                                args.attn_heads, in_channels))

            if i != 0 and i != len(channels) - 1:
                layers.append(UpSampleBlock(in_channels,
                                            num_dimensions=2,
                                            padding_mode=padding_mode))
                resolution *= 2
        layers.append(nn.Upsample(size=(args.Ly, args.Lx), mode='nearest'))
        resolution = args.Ly
        layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1, padding_mode=padding_mode))  # resample with conv
        if final_smoothing:
            layers.append(FourierBasicBlock(in_channels, in_channels,
                                            modes=[16, 16]))    # this assumes at least 64*64 output
        else:
            if resolution in attn_resolutions:
                if not args.use_fa:
                    layers.append(SABlock(in_channels, args.attn_heads, args.attn_dim,
                                          use_pe=True, block_size=resolution ** 2))
                else:
                    layers.append(FABlock2D(in_channels, args.attn_dim, args.attn_dim,
                                            args.attn_heads, in_channels))
            layers.append(nn.Conv2d(in_channels, in_channels, 1, 1, 0, padding_mode=padding_mode))
        layers.append(nn.GroupNorm(8, in_channels))
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

    def load_checkpoint(self, path):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt, strict=True)

