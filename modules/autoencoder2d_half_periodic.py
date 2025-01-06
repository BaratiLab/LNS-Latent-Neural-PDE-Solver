import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.basics import ResidualBlock, GroupNorm, Swish, FourierBasicBlock, SABlock
from modules.factorized_attention import FABlock2D
from modules.propagator import SimpleResNet, SimpleMLP
from modules.cond_utils import CondResidualBlock, fourier_embedding


class NormSwish(nn.Module):
    def __init__(self, in_channels):
        super(NormSwish, self).__init__()
        self.in_channels = in_channels
        self.norm_act = nn.Sequential(
            GroupNorm(in_channels),
            Swish()
        )

    def forward(self, x):
        return self.norm_act(x)


class HalfPeriodicConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, periodic_direction='x'):
        super(HalfPeriodicConv2d, self).__init__(in_channels, out_channels,
                                                  kernel_size, stride,
                                                  0, dilation,
                                                  groups, bias)
        self.padding_ = padding
        self.periodic_direction = periodic_direction

    def pad(self, x):
        # pad x according to the periodic direction
        if self.periodic_direction == 'x':
            x = F.pad(x, (self.padding_, self.padding_, 0, 0), mode='circular')
            x = F.pad(x, (0, 0, self.padding_, self.padding_), mode='constant', value=0)
        elif self.periodic_direction == 'y':
            x = F.pad(x, (0, 0, self.padding_, self.padding_), mode='circular')
            x = F.pad(x, (self.padding_, self.padding_, 0, 0), mode='constant', value=0)
        else:
            raise ValueError('periodic_direction must be x or y')
        return x

    def forward(self, x):
        # pad x according to the periodic direction
        x = self.pad(x)
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class UpSampleBlock2D(nn.Module):
    def __init__(self, channels,
                 periodic_direction='x'):
        super(UpSampleBlock2D, self).__init__()
        self.conv_layer = HalfPeriodicConv2d(channels, channels, 3, 1, 1, periodic_direction=periodic_direction)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        x = self.conv_layer(x)

        return x


class DownSampleBlock2d(nn.Module):
    def __init__(self, channels, periodic_direction='x'):
        super(DownSampleBlock2d, self).__init__()
        self.conv_layer = HalfPeriodicConv2d(channels, channels, 3, 2, 1, periodic_direction=periodic_direction)

    def forward(self, x):
        return self.conv_layer(x)


class HalfPeriodicResBlock2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 periodic_direction='x'):
        super(HalfPeriodicResBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_act1 = NormSwish(in_channels)
        self.norm_act2 = NormSwish(out_channels)
        self.periodic_direction = periodic_direction

        self.conv1 = HalfPeriodicConv2d(in_channels, out_channels, 3, 1, 1, periodic_direction=periodic_direction)
        self.conv2 = HalfPeriodicConv2d(out_channels, out_channels, 3, 1, 1, periodic_direction=periodic_direction)

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        # x: [b c h w]
        x_skip = self.channel_up(x) if hasattr(self, 'channel_up') else x
        x = self.norm_act1(x)
        x = self.conv1(x)
        x = self.norm_act2(x)
        x = self.conv2(x)

        return x + x_skip


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = args.encoder_channels
        resolutions = args.resolutions
        latent_resolution = args.latent_resolution

        num_res_blocks = args.encoder_res_blocks
        resolution_height, resolution_width = resolutions[0], resolutions[1]
        hw_ratio = args.hw_ratio
        assert (len(channels) - 2) == int(math.log2(resolution_height//latent_resolution))

        periodic_direction = args.periodic_direction

        layers = [nn.Conv2d(args.in_channels, channels[0], 1, 1, 0),
                  Swish(),
                  HalfPeriodicResBlock2d(channels[0], channels[0], periodic_direction=periodic_direction)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(HalfPeriodicResBlock2d(in_channels, out_channels,
                                                     periodic_direction=periodic_direction))
                in_channels = out_channels

            if i != len(channels)-2:
                layers.append(DownSampleBlock2d(channels[i+1], periodic_direction=periodic_direction))
                resolution_height //= 2
        layers.append(HalfPeriodicResBlock2d(channels[-1], channels[-1],
                                             periodic_direction=periodic_direction))
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
        final_smoothing = args.final_smoothing

        periodic_direction = args.periodic_direction
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
            layers = [HalfPeriodicConv2d(args.latent_dim, in_channels, 3, 1, 1, periodic_direction=periodic_direction),
                      SABlock(in_channels, attn_heads, attn_dim, use_pe=False,
                              block_size=resolution_height*int(resolution_height*(hw_ratio+0.5))),
                      HalfPeriodicResBlock2d(in_channels, in_channels, periodic_direction=periodic_direction)]
        else:
            layers = [HalfPeriodicConv2d(args.latent_dim, in_channels, 3, 1, 1, periodic_direction=periodic_direction),
                      HalfPeriodicResBlock2d(in_channels, in_channels, periodic_direction=periodic_direction),
                      HalfPeriodicResBlock2d(in_channels, in_channels, periodic_direction=periodic_direction)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(HalfPeriodicResBlock2d(in_channels, out_channels, periodic_direction=periodic_direction))
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
                                              use_pe=False,
                                              block_size=
                                              resolution_height*int(resolution_height*(hw_ratio+0.5))))

            if i != 0 and i != len(channels) - 1:
                layers.append(UpSampleBlock2D(in_channels, periodic_direction=periodic_direction))
                resolution_height *= 2
        layers.append(nn.Upsample(size=(args.Ly, args.Lx), mode='nearest'))
        resolution_height = args.Ly
        # resample with conv
        layers.append(HalfPeriodicConv2d(in_channels, in_channels, 3, 1, 1, periodic_direction=periodic_direction))
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
                                          use_pe=False,
                                          block_size=
                                          resolution_height * int(resolution_height * (hw_ratio + 0.5))))
            layers.append(HalfPeriodicConv2d(in_channels, in_channels, 3, 1, 1, periodic_direction=periodic_direction))
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
