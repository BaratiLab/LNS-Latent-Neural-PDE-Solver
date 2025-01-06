# This code snippet is modified from: https://github.com/microsoft/pdearena/blob/main/pdearena/modules/conditioned/fourier_cond.py

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
from torch import nn
from modules.cond_utils import ConditionedBlock


def batchmul2d(input, weights, emb):
    temp = input * emb.unsqueeze(1)
    out = torch.einsum("bixy,ioxy->boxy", temp, weights)
    return out


class FreqLinear(nn.Module):
    def __init__(self, in_channel, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channel + 4 * modes1 * modes2)
        self.weights = nn.Parameter(scale * torch.randn(in_channel, 4 * modes1 * modes2, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, 4 * modes1 * modes2, dtype=torch.float32))

    def forward(self, x):
        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, self.modes2, 2, 2)
        return torch.view_as_complex(h)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, modes1, modes2):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        @author: Zongyi Li
        [paper](https://arxiv.org/pdf/2010.08895.pdf)
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )
        self.cond_emb = FreqLinear(cond_channels, self.modes1, self.modes2)

    def forward(self, x, emb):
        emb12 = self.cond_emb(emb)
        emb1 = emb12[..., 0]
        emb2 = emb12[..., 1]
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = batchmul2d(
            x_ft[:, :, : self.modes1, : self.modes2], torch.view_as_complex(self.weights1), emb1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = batchmul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], torch.view_as_complex(self.weights2), emb2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class CondFourierBasicBlock(ConditionedBlock):
    # only support 2D for now

    expansion: int = 1

    def __init__(
        self,
        in_planes,
        planes,
        modes,
        residual=True,
    ):
        super().__init__()
        self.modes = modes
        self.num_dimensions = len(modes)
        assert self.num_dimensions == 2
        self.residual = residual

        self.fourier = SpectralConv2d(in_planes, planes, in_planes, modes[0], modes[1])
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0)
        self.cond_emb = nn.Linear(in_planes, planes)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        x_skip = x

        x1 = self.fourier(x, cond_emb)
        x2 = self.conv(x)
        emb_out = self.cond_emb(cond_emb)
        while len(emb_out.shape) < len(x2.shape):
            emb_out = emb_out[..., None]
        out = nn.GELU()(x1 + x2 + emb_out)    # hard coded for now

        if self.residual:
            out = x_skip + out
        return out