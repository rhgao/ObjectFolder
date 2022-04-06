import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from AudioNet_utils import *

class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = 'relu', *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        #5 is for x, y, z, f, t
        return nn.Identity(), 5

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class AudioNeRF(nn.Module):
    def __init__(self, D=8, input_ch=5, output_ch=2):
        super(AudioNeRF, self).__init__()
        self.model_x = NeRF(D = D, input_ch = input_ch, output_ch = output_ch)
        self.model_y = NeRF(D = D, input_ch = input_ch, output_ch = output_ch)
        self.model_z = NeRF(D = D, input_ch = input_ch, output_ch = output_ch)

    def forward(self, embedded_x, embedded_y, embedded_z):
        results_x = self.model_x(embedded_x)
        results_y = self.model_y(embedded_y)
        results_z = self.model_z(embedded_z)
        return results_x, results_y, results_z


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=5, input_ch_views=0, output_ch=2, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation='relu')] + [DenseLayer(W, W, activation='relu') if i not in self.skips else DenseLayer(W + input_ch, W, activation='relu') for i in range(D-1)])

        self.views_linears = nn.ModuleList([DenseLayer(input_ch_views + W, W//2, activation='relu')])

        if use_viewdirs:
            self.feature_linear = DenseLayer(W, W, activation='sigmoid')
            #self.alpha_linear = DenseLayer(W, 1, activation='linear')
            self.rgb_linear = DenseLayer(W//2, output_ch, activation='sigmoid')
        else:
            self.output_linear = DenseLayer(W, output_ch, activation='sigmoid')


    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            outputs = self.rgb_linear(h)
        else:
            outputs = self.output_linear(h)

        return outputs
