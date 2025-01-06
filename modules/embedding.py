import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# helpers
def exists(val):
    return val is not None


def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


# sin activation
class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 w0=1.,
                 c=6.,
                 is_first=False,
                 use_bias=True,
                 activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (np.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


# siren network
class SirenNet(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_hidden, dim_out, num_layers,
                 w0=1.,
                 w0_initial=30.,
                 use_bias=True, final_activation=None,
                 normalize_input=True):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.normalize_input = normalize_input

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first,
            ))

        # final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = nn.Linear(dim_hidden, dim_out)
        self.reset_params()
        # self.last_layer = Siren(dim_in=dim_hidden,
        #                         dim_out=dim_out,
        #                         w0=w0,
        #                         use_bias=use_bias,
        #                         activation=final_activation)

    def in_norm(self, x):
        return (2 * x - torch.min(x, dim=1, keepdim=True)[0] - torch.max(x, dim=1, keepdim=True)[0]) /\
            (torch.max(x, dim=1, keepdim=True)[0] - torch.min(x, dim=1, keepdim=True)[0])

    def reset_params(self):
        # reset parameters of last layer, weight to normal 0.02, bias to 0
        nn.init.normal_(self.last_layer.weight, mean=0, std=0.02)
        nn.init.constant_(self.last_layer.bias, 0)

    def forward(self, x, mods=None):
        if self.normalize_input:
            x = self.in_norm(x)
        else:
            if torch.max(x) > 1. or torch.min(x) < -1.:
                raise Exception('Input x to SirenNet is not normalized')
        # x = (x - 0.5) * 2

        for layer in self.layers:
            x = layer(x)
        if mods is not None:
            x *= mods
        x = self.last_layer(x)
        # x = self.final_activation(x)
        return x


class EmbeddingWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding_key = []
        for k, v in zip(args.context_embedding.keys, args.context_embedding.settings):
            assert k[-3:] == 'emb', 'context embedding key must end with emb'
            # build siren layers with key
            if v['encoder'] == 'siren':
                net = SirenNet(dim_in=v["in_channels"], dim_hidden=v["hidden_channels"], dim_out=v["out_channels"],
                               num_layers=v["num_layers"], normalize_input=False)  # assume context is already normalized
                self.add_module(k, net)
            elif v['encoder'] == 'embedding':
                # use torch.nn.Embedding
                assert v['in_channels'] == 1, 'embedding only support 1 channel'
                self.add_module(k,
                                 nn.Embedding(v['num_embeddings'], v['out_channels']))
            elif v['encoder'] == 'linear':
                self.add_module(k,
                                 nn.Linear(v['in_channels'], v['out_channels']))

            self.embedding_key.append(k)

    def forward(self, context_dict):
        context_embedding = []
        for k, (param_name, param) in zip(self.embedding_key, context_dict.items()):
            assert param_name == k[:-4], 'context embedding key does not match'
            context_embedding.append(self._modules[k](param))

            if len(context_embedding[-1].shape) == 3:
                context_embedding[-1] = context_embedding[-1].squeeze(1)  # [b, emb_dim]
        if len(context_embedding) == 1:
            return context_embedding[0].unsqueeze(1)  # [b, 1, emb_dim]
        else:
            return torch.stack(context_embedding, dim=1)        # [b, n_context, emb_dim]


# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)

def apply_3d_rotary_pos_emb(t, freqs_x, freqs_y, freqs_z):
    # split t into three parts
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y, t_z = t[..., :d//3], t[..., d//3:2*d//3], t[..., 2*d//3:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y),
                      apply_rotary_pos_emb(t_z, freqs_z)), dim=-1)


# https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)