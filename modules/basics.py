import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Union, Tuple, List, Optional
from modules.siren_module import SirenNet, SirenWrapper
from modules.embedding import RotaryEmbedding, apply_rotary_pos_emb
from modules.cond_utils import zero_module, ConditionedBlock

ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Complex multiplication 1d
def batchmul1d(input, weights):
    # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", input, weights)


# Complex multiplication 2d
def batchmul2d(input, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", input, weights)


# Complex multiplication 3d
def batchmul3d(input, weights):
    # (batch, in_channel, x,y,z ), (in_channel, out_channel, x,y,z) -> (batch, out_channel, x,y,z)
    return torch.einsum("bixyz,ioxyz->boxyz", input, weights)


################################################################
# fourier layer
################################################################


class SpectralConv1d(nn.Module):
    """1D Fourier layer. Does FFT, linear transform, and Inverse FFT.
    Implemented in a way to allow multi-gpu training.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes (int): Number of Fourier modes
    [paper](https://arxiv.org/abs/2010.08895)
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, 2, dtype=torch.float32)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes] = batchmul1d(
            x_ft[:, :, : self.modes], torch.view_as_complex(self.weights)
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=(x.size(-1),))
        return x


class SpectralConv2d(nn.Module):
    """2D Fourier layer. Does FFT, linear transform, and Inverse FFT.
    Implemented in a way to allow multi-gpu training.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes1 (int): Number of Fourier modes to keep in the first spatial direction
        modes2 (int): Number of Fourier modes to keep in the second spatial direction
    [paper](https://arxiv.org/abs/2010.08895)
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()

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

    def forward(self, x, x_dim=None, y_dim=None):
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
            x_ft[:, :, : self.modes1, : self.modes2], torch.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1:, : self.modes2] = batchmul2d(
            x_ft[:, :, -self.modes1:, : self.modes2], torch.view_as_complex(self.weights2)
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv3d(nn.Module):
    """3D Fourier layer. Does FFT, linear transform, and Inverse FFT.
    Implemented in a way to allow multi-gpu training.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes1 (int): Number of Fourier modes to keep in the first spatial direction
        modes2 (int): Number of Fourier modes to keep in the second spatial direction
        modes3 (int): Number of Fourier modes to keep in the third spatial direction
    [paper](https://arxiv.org/abs/2010.08895)
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2,
                                    dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2,
                                    dtype=torch.float32)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2,
                                    dtype=torch.float32)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2,
                                    dtype=torch.float32)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = batchmul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], torch.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1:, : self.modes2, : self.modes3] = batchmul3d(
            x_ft[:, :, -self.modes1:, : self.modes2, : self.modes3], torch.view_as_complex(self.weights2)
        )
        out_ft[:, :, : self.modes1, -self.modes2:, : self.modes3] = batchmul3d(
            x_ft[:, :, : self.modes1, -self.modes2:, : self.modes3], torch.view_as_complex(self.weights3)
        )
        out_ft[:, :, -self.modes1:, -self.modes2:, : self.modes3] = batchmul3d(
            x_ft[:, :, -self.modes1:, -self.modes2:, : self.modes3], torch.view_as_complex(self.weights4)
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

# general n-d convolution
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 num_dimensions,
                 padding_mode='zeros'):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dimensions = num_dimensions
        # set conv layer according to the number of dimensions
        if self.num_dimensions == 1:
            self.block = nn.Sequential(
                GroupNorm(in_channels),
                Swish(),
                nn.Conv1d(in_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
                GroupNorm(out_channels),
                Swish(),
                nn.Conv1d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode)
            )

            if in_channels != out_channels:
                self.channel_up = nn.Conv1d(in_channels, out_channels, 1, 1, 0)
        elif self.num_dimensions == 2:
            self.block = nn.Sequential(
                GroupNorm(in_channels),
                Swish(),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
                GroupNorm(out_channels),
                Swish(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode)
            )

            if in_channels != out_channels:
                self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        elif self.num_dimensions == 3:
            self.block = nn.Sequential(
                GroupNorm(in_channels),
                Swish(),
                nn.Conv3d(in_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
                GroupNorm(out_channels),
                Swish(),
                nn.Conv3d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode))

            if in_channels != out_channels:
                self.channel_up = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

        else:
            raise ValueError('num_dimensions must be 1, 2, or 3')

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels, num_dimensions,
                 padding_mode='zeros'):
        super(UpSampleBlock, self).__init__()
        self.num_dimensions = num_dimensions
        # set conv layer according to the number of dimensions
        if self.num_dimensions == 1:
            self.conv_layer = nn.Conv1d(channels, channels, 3, 1, 1, padding_mode=padding_mode)
        elif self.num_dimensions == 2:
            self.conv_layer = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode=padding_mode)
        elif self.num_dimensions == 3:
            self.conv_layer = nn.Conv3d(channels, channels, 3, 1, 1, padding_mode=padding_mode)
        else:
            raise ValueError('num_dimensions must be 1, 2, or 3')


    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        x = self.conv_layer(x)

        return x


class DownSampleBlock(nn.Module):
    def __init__(self, channels, num_dimensions,
                 padding_mode='zeros'):
        super(DownSampleBlock, self).__init__()
        self.num_dimensions = num_dimensions
        # set conv layer according to the number of dimensions
        if self.num_dimensions == 1:
            self.conv_layer = nn.Conv1d(channels, channels, 3, 2, 0)
        elif self.num_dimensions == 2:
            self.conv_layer = nn.Conv2d(channels, channels, 3, 2, 0)
        elif self.num_dimensions == 3:
            self.conv_layer = nn.Conv3d(channels, channels, 3, 2, 0)
        else:
            raise ValueError('num_dimensions must be 1, 2, or 3')

        self.padding_mode = padding_mode if padding_mode != 'zeros' else 'constant'
        self.pad = []
        if self.padding_mode == 'circular':
            for dim in range(self.num_dimensions):
                self.pad.extend((1, 1))
        else:
            for dim in range(self.num_dimensions):
                self.pad.extend((0, 1))

    def forward(self, x):
        x = F.pad(x, self.pad, mode=self.padding_mode)
        return self.conv_layer(x)


class SABlock(nn.Module):
    def __init__(self,
                 dim,
                 heads,
                 dim_head,
                 use_pe=False,     # simple learnable positional encoding
                 block_size=512,    # maximum length
                 ):
        super(SABlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        self.ln = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, heads*dim_head, bias=False)
        self.to_k = nn.Linear(dim, heads*dim_head, bias=False)
        self.to_v = nn.Linear(dim, heads*dim_head)
        self.proj_out = nn.Linear(heads*dim_head, dim)

        if use_pe:
            # simple patch embedding
            self.pe = nn.Parameter(torch.randn(1, block_size, dim) * 0.02, requires_grad=True)
        else:
            self.pe = None

        self.init_params()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_params(self):
        for m in self.modules():
            self._init_weights(m)

    def add_pe(self, x):
        if self.pe is None:
            return x
        else:
            return x + self.pe[:, :x.shape[1]]

    def forward(self, x, channel_last=False):
        if not channel_last:
            b, c = x.shape[:2]
            ls = x.shape[2:]
            x = x.view(b, c, -1).transpose(1, 2)
            x = x.contiguous()
        # now x is b n c
        x_in = x
        x = self.ln(x)  # pre-norm
        x = self.add_pe(x)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        attn = torch.einsum('bhid,bhjd->bhij', q, k)
        attn = attn * (int(v.shape[-1])**(-0.5))
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = self.proj_out(rearrange(out, 'b h n d -> b n (h d)'))
        out = x_in + out

        if not channel_last:
            out = out.transpose(1, 2).view(b, c, *ls)
        return out


class LABlock(nn.Module):
    def __init__(self,
                 dim,
                 heads,
                 dim_head,
                 use_pe=False,     # simple learnable positional encoding
                 block_size=512,    # maximum length
                 ):
        super(LABlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        self.ln = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, heads*dim_head, bias=False)
        self.to_k = nn.Linear(dim, heads*dim_head, bias=False)
        self.to_v = nn.Linear(dim, heads*dim_head)
        self.proj_out = nn.Linear(heads*dim_head, dim)

        if use_pe:
            # simple patch embedding
            self.pe = nn.Parameter(torch.randn(1, block_size, dim) * 0.02, requires_grad=True)
        else:
            self.pe = None

        self.init_params()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_params(self):
        for m in self.modules():
            self._init_weights(m)

    def add_pe(self, x):
        if self.pe is None:
            return x
        else:
            return x + self.pe[:, :x.shape[1]]

    def forward(self, x, channel_last=False):
        if not channel_last:
            b, c = x.shape[:2]
            ls = x.shape[2:]
            x = x.view(b, c, -1).transpose(1, 2)
            x = x.contiguous()
        # now x is b n c
        x_in = self.ln(x)  # pre-norm
        x_in = self.add_pe(x_in)
        q = self.to_q(x_in)
        k = self.to_k(x_in)
        v = self.to_v(x_in)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        attn = torch.einsum('bhid,bhjd->bhij', q, k)
        attn = attn * (int(v.shape[-1])**(-0.5))
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = self.proj_out(rearrange(out, 'b h n d -> b n (h d)'))
        out = out + x

        if not channel_last:
            out = out.transpose(1, 2).view(b, c, *ls)
        return out


class CABlock(nn.Module):
    def __init__(self,
                 dim,
                 context_dim,
                 heads,
                 dim_head):
        super(CABlock, self).__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.heads = heads
        self.dim_head = dim_head

        self.ln_x = nn.LayerNorm(dim)
        self.ln_y = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(context_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(context_dim, heads * dim_head)
        self.proj_out = nn.Linear(heads * dim_head, dim)

    def forward(self, x, y, channel_last=False):
        if not channel_last:
            b, c = x.shape[:2]
            ls = x.shape[2:]
            x = x.view(b, c, -1).transpose(1, 2)
            x = x.contiguous()
        # now x is b n c
        x = self.ln_x(x)  # pre-norm
        y = self.ln_y(y)  # pre-norm

        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        attn = torch.einsum('bhid,bhjd->bhij', q, k)
        attn = attn * (int(v.shape[-1]) ** (-0.5))
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = self.proj_out(rearrange(out, 'b h n d -> b n (h d)'))
        out = x + out

        if channel_last:
            out = out.transpose(1, 2).view(b, c, *ls)
        return out


class FourierBasicBlock(nn.Module):
    """
    Generic ND Basic block for Fourier Neural Operators

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        stride (int, optional): stride of the convolution. Defaults to 1.
        modes_num (list, optional): number of modes for each spatial dimension. Defaults to 16.
        activation (str, optional): activation function. Defaults to "relu".
        norm (bool, optional): whether to use group normalization. Defaults to False.

    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        modes: List[int],
        activation: str = "gelu",
        residual: bool = True,
    ):
        super().__init__()
        self.modes = modes
        self.num_dimensions = len(modes)
        self.residual = residual

        if self.num_dimensions == 1:
            self.fourier = SpectralConv1d(in_planes, planes, modes[0])
            self.conv = nn.Conv1d(in_planes, planes, kernel_size=1, stride=1, padding=0)
        elif self.num_dimensions == 2:
            self.fourier = SpectralConv2d(in_planes, planes, modes[0], modes[1])
            self.conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0)
        elif self.num_dimensions == 3:
            self.fourier = SpectralConv3d(in_planes, planes, modes[0], modes[1], modes[2])
            self.conv = nn.Conv3d(in_planes, planes, kernel_size=1, stride=1, padding=0)

        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x

        x1 = self.fourier(x)
        x2 = self.conv(x)
        out = self.activation(x1 + x2)

        if self.residual:
            out = x_skip + out
        return out


# general n-d convolution
class ResFNOMixerBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 modes,
                 norm='in'        # ['in', 'ln', 'none']
                 ):
        super(ResFNOMixerBlock, self).__init__()
        self.modes = modes
        self.num_dimensions = len(modes)
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert norm in ['in', 'ln', 'none']
        # set conv layer according to the number of dimensions
        if self.num_dimensions == 1:
            if norm == 'in':
                self.norm_fn = nn.InstanceNorm1d(in_channels)
            elif norm == 'ln':
                self.norm_fn = nn.GroupNorm(1, out_channels),
            else:
                self.norm_fn = nn.Identity()
            self.token_mixer = SpectralConv1d(in_channels, out_channels, modes[0])
            self.channel_mixer = nn.Sequential(
                nn.GroupNorm(1, out_channels),      # hard coded for now
                nn.Conv1d(out_channels, out_channels, 1),
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, 1)
            )

            if in_channels != out_channels:
                self.channel_up = nn.Conv1d(in_channels, out_channels, 1, 1, 0)
        elif self.num_dimensions == 2:
            if norm == 'in':
                self.norm_fn = nn.InstanceNorm2d(in_channels)
            elif norm == 'ln':
                self.norm_fn = nn.GroupNorm(1, out_channels),
            else:
                self.norm_fn = nn.Identity()
            self.token_mixer = SpectralConv2d(in_channels, out_channels, modes[0], modes[1])
            self.channel_mixer = nn.Sequential(
                nn.GroupNorm(1, out_channels),      # hard coded for now
                nn.Conv2d(out_channels, out_channels, 1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 1)
            )

            if in_channels != out_channels:
                self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        elif self.num_dimensions == 3:
            if norm == 'in':
                self.norm_fn = nn.InstanceNorm2d(in_channels)
            elif norm == 'ln':
                self.norm_fn = nn.GroupNorm(1, in_channels)
            else:
                self.norm_fn = nn.Identity()
            self.token_mixer = SpectralConv3d(in_channels, out_channels, modes[0], modes[1], modes[2])
            self.channel_mixer = nn.Sequential(
                nn.GroupNorm(1, out_channels),     # hard coded for now
                nn.Conv3d(out_channels, out_channels, 1),
                nn.GELU(),
                nn.Conv3d(out_channels, out_channels, 1)
            )

            if in_channels != out_channels:
                self.channel_up = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

        else:
            raise ValueError('num_dimensions must be 1, 2, or 3')

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.channel_mixer(self.token_mixer(self.norm_fn(x)))
        else:
            return x + self.channel_mixer(self.token_mixer(self.norm_fn(x)))


class CondResFNOMixerBlock(ConditionedBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 modes,
                 norm='in'        # ['in', 'ln', 'none']
                 ):
        super(CondResFNOMixerBlock, self).__init__()
        self.modes = modes
        self.num_dimensions = len(modes)
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert norm in ['in', 'ln', 'none']
        assert self.num_dimensions == 2  # only try 2d for now

        if norm == 'in':
            self.norm_fn = nn.InstanceNorm2d(in_channels)
        elif norm == 'ln':
            self.norm_fn = nn.GroupNorm(1, out_channels),
        else:
            self.norm_fn = nn.Identity()
        self.token_mixer = SpectralConv2d(in_channels, out_channels, modes[0], modes[1])
        self.channel_mixer = nn.Sequential(
            nn.GroupNorm(1, out_channels),      # hard coded for now
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        self.cond_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 1),
            nn.GELU(),
            zero_module(nn.Conv2d(self.in_channels, self.in_channels, 1)),
        )

    def forward(self, x, cond_emb):
        while len(cond_emb.shape) < len(x.shape):
            cond_emb = cond_emb[..., None]
        if self.in_channels != self.out_channels:
            x_skip = self.channel_up(x)
            x = self.token_mixer(self.norm_fn(x))
            x = self.channel_mixer(x * (1. + self.cond_conv(cond_emb)))
            return x + x_skip
        else:
            x_skip = x
            x = self.token_mixer(self.norm_fn(x))
            x = self.channel_mixer(x * (1. + self.cond_conv(cond_emb)))
            return x + x_skip