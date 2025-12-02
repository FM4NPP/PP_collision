import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# class Embedder(nn.Module):
#     def __init__(self, embed_dim, init_scale = 0.02, dropout = 0.2):
#         super(Embedder, self).__init__()
        
#         # assert embed_dim % 6 == 0, "embed_dim should be divisible by 6"
#         in_dim = 1  # E, θ, φ, ψ instead of just E
#         E_emb_dim = embed_dim // 4
#         pos_emb_dim = embed_dim - E_emb_dim
#         self.proj = nn.Parameter(torch.randn(in_dim, E_emb_dim) * init_scale, requires_grad=True)
#         self.proj_pos = nn.Parameter(torch.randn(3, pos_emb_dim) * init_scale, requires_grad=True)
#         self.norm = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, neighborhood):
#         B, P, _ = neighborhood.shape
#         E = neighborhood[..., 0:1]  # First coordinate as Energy (or time-dependent variable)
#         E_embed = torch.matmul(E, self.proj)
        
#         pos_embed = torch.matmul(neighborhood[..., 1:4], self.proj_pos)
        
#         out = self.dropout(torch.cat([E_embed, pos_embed], dim=-1))       
#         return self.norm(out)
    
class EmbedderAdd(nn.Module):
    def __init__(self, pe_method, embed_dim, learnable_projection = False):
        super(EmbedderAdd, self).__init__()
        assert pe_method in ['none', 'ff', 'nerf', 'cpe']

        in_dim = 1  # E, θ, φ, ψ instead of just E
        self.proj = nn.Parameter(torch.randn(in_dim, embed_dim), requires_grad=learnable_projection)
        self.embed = CoordinateEmbedder(method = pe_method, n_continuous_dim = 3, target_dim = embed_dim, learnable_projection = learnable_projection)

    def forward(self, neighborhood):
        B, P, _ = neighborhood.shape
        E = neighborhood[..., 0:1]  # First coordinate as Energy (or time-dependent variable)
        E_embed = torch.matmul(E, self.proj)
        pos_embed = self.embed(neighborhood[..., 1:4])        
        out = E_embed + pos_embed
        return out, pos_embed
    
class EmbedderConcat(nn.Module):
    def __init__(self, pe_method, embed_dim, learnable_projection = False):
        super(EmbedderConcat, self).__init__()
        assert pe_method in ['none', 'ff', 'nerf', 'cpe']

        in_dim = 1  # E, θ, φ, ψ instead of just E
        E_emb_dim = embed_dim // 4
        self.proj = nn.Parameter(torch.randn(in_dim, E_emb_dim), requires_grad=learnable_projection)
        self.embed = CoordinateEmbedder(method = pe_method, n_continuous_dim = 3, target_dim = embed_dim, learnable_projection = learnable_projection)
        self.proj2 = nn.Parameter(torch.randn(E_emb_dim + embed_dim, embed_dim), requires_grad=learnable_projection)

    def forward(self, neighborhood):
        B, P, _ = neighborhood.shape
        E = neighborhood[..., 0:1]  # First coordinate as Energy (or time-dependent variable)
        E_embed = torch.matmul(E, self.proj)
        pos_embed = self.embed(neighborhood[..., 1:4])        
        out = torch.cat([E_embed, pos_embed], dim=-1)    
        out = torch.matmul(out, self.proj2)
        return out, pos_embed
    
class EmbedderPosOnly(nn.Module):
    def __init__(self, pe_method, embed_dim, learnable_projection = False):
        super(EmbedderPosOnly, self).__init__()
        assert pe_method in ['none', 'ff', 'nerf', 'cpe']
        self.embed = CoordinateEmbedder(method = pe_method, n_continuous_dim = 3, target_dim = embed_dim, learnable_projection = learnable_projection)

    def forward(self, neighborhood):
        out = self.embed(neighborhood[..., 1:4])       
        return out
        
class CoordinateEmbedder(nn.Module):
    """
    Three different continuous coordinate embedding methods are merged.
    1. Fourier features
    2. Nerf
    3. Continuous PE
    """
    
    def __init__(self, method = 'cpe', n_continuous_dim = 3, target_dim = 256, learnable_projection = False):
        super(CoordinateEmbedder, self).__init__()
        
        pseudo_input = torch.randn(1, 2, n_continuous_dim)
        
        if method == 'ff':
            self.get_ff()
            out_dim = self.pec.forward(pseudo_input).shape[-1]
            print('orig output_dim: {}'.format(out_dim))
            self.projection = nn.Parameter(torch.randn(out_dim, target_dim), requires_grad = learnable_projection)
            
        elif method == 'nerf':
            multires = 10
            self.get_nerf(multires, n_continuous_dim)
            out_dim = self.pec.forward(pseudo_input).shape[-1]
            print('orig output_dim: {}'.format(out_dim))
            self.projection = nn.Parameter(torch.randn(out_dim, target_dim), requires_grad = learnable_projection)
                        
        elif method == 'cpe':
            self.get_cpe(n_continuous_dim, target_dim)
            self.projection = None     
            
        elif method == 'none':
            self.pec = nn.Identity()
            self.projection = nn.Parameter(torch.randn(3, target_dim), requires_grad = learnable_projection)
            
    def apply_projection(self, tensor):
        return torch.matmul(tensor, self.projection)
    
    def get_ff(self,):
        pos2fourier_position_encoding_kwargs = dict(
        num_bands = [12, 12, 12],
        max_resolution = [20, 20, 20],
        )
        self.pec = FourierPositionEncoding(**pos2fourier_position_encoding_kwargs)

    def get_cpe(self, n_continuous_dim, target_dim):
        self.pec = PositionEmbeddingCoordsSine(n_dim = n_continuous_dim, d_model = target_dim)
    
    def get_nerf(self, multires, n_continuous_dim):
        embed_kwargs = {
                'include_input': True,
                'n_continuous_dim': n_continuous_dim,
                'max_freq_log2': multires-1,
                'num_freqs': multires,
                'log_sampling': True,
                'periodic_fns': [torch.sin, torch.cos],
            }
        self.pec = NerfEmbedder(**embed_kwargs)

    def forward(self, tensor):
        """
        tensor: b x N_seq x self.n_continuous_dim
        out: b x N_seq x self.target_dim
        """
        out = self.pec.forward(tensor)
        if self.projection is not None:
            out = self.apply_projection(out)
        return out

    
class FourierPositionEncoding():
    """ Fourier (Sinusoidal) position encoding. """

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    def output_size(self):
        """ Returns size of positional encodings last dimension. """
        encoding_size = sum(self.num_bands)
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += len(self.max_resolution)
        return encoding_size

    def forward(self, pos=None):
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only)
        return fourier_pos_enc


def generate_fourier_features(pos, num_bands, max_resolution=(2 ** 10), concat_pos=True, sine_only=False):
    """
    Generate a Fourier feature position encoding with linear spacing.

    Args:
        pos: The Tensor containing the position of n points in d dimensional space.
        num_bands: The number of frequency bands (K) to use.
        max_resolution: The maximum resolution (i.e., the number of pixels per dim). A tuple representing resoltuion for each dimension.
        concat_pos: Whether to concatenate the input position encoding to the Fourier features.
        sine_only: Whether to use a single phase (sin) or two (sin/cos) for each frequency band.
    """
    batch_size = pos.shape[0]
    min_freq = 1.0 
    stacked = []
    
    for i, (res, num_band) in enumerate(zip(max_resolution, num_bands)):       
        stacked.append(pos[:, :, i, None] * torch.linspace(start=min_freq, end=res / 2, steps=num_band)[None, :].to(device = pos.device))

    per_pos_features = torch.cat(stacked, dim=-1)  
    per_pos_features = torch.cat([torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1)
    per_pos_features = torch.cat([pos, per_pos_features], dim=-1)
    return per_pos_features


class NerfEmbedder:

    def __init__(self, n_continuous_dim, include_input, max_freq_log2, num_freqs, log_sampling, periodic_fns):
        
        self.n_continuous_dim = n_continuous_dim
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.n_continuous_dim 
        out_dim = 0
        
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
    
class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
       arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super(PositionEmbeddingCoordsSine, self).__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb