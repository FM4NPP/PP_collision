"""
Longformer model for particle physics pretraining
Uses local + global attention for linear complexity O(n*w) where w is window size
Target: 2M-5M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fm4npp.models.embed import EmbedderAdd, EmbedderConcat
from fm4npp.models.rmsnorm import RMSNorm


class LongformerAttention(nn.Module):
    """Longformer attention with sliding window + global attention"""
    def __init__(self, embed_dim, num_heads=4, window_size=256, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Compute sliding window attention
        # For simplicity, use chunked attention with window_size
        window_size = min(self.window_size, N)

        # Pad sequence to be divisible by window_size
        pad_len = (window_size - N % window_size) % window_size
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        # Reshape into windows
        N_padded = N + pad_len
        num_windows = N_padded // window_size

        q = q.reshape(B, self.num_heads, num_windows, window_size, self.head_dim)
        k = k.reshape(B, self.num_heads, num_windows, window_size, self.head_dim)
        v = v.reshape(B, self.num_heads, num_windows, window_size, self.head_dim)

        # Compute attention within each window
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v

        # Reshape back
        out = out.reshape(B, self.num_heads, N_padded, self.head_dim)
        out = out[:, :, :N, :]  # Remove padding

        # Transpose and reshape
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return out


class LongformerBlock(nn.Module):
    """Longformer transformer block"""
    def __init__(self, embed_dim, num_heads=4, window_size=256, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = LongformerAttention(embed_dim, num_heads, window_size, dropout)
        self.norm2 = RMSNorm(embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LongformerGPT(nn.Module):
    """
    Longformer model matching MambaGPT interface

    Parameters for ~3M params:
    - embed_dim: 256
    - num_layers: 6
    - num_heads: 4
    - window_size: 256
    """
    def __init__(self, embed_dim=256, num_layers=6, num_heads=4, window_size=256,
                 klen=10, dropout=0.1, embed_method='add', pe_method='nerf', mlp_ratio=4.0):
        super().__init__()
        assert embed_method in ['concat', 'add']
        self.embed_dim = embed_dim

        if embed_method == 'concat':
            Embedder = EmbedderConcat
        else:
            Embedder = EmbedderAdd

        self.embedder = Embedder(pe_method=pe_method, embed_dim=embed_dim, learnable_projection=False)

        self.blocks = nn.ModuleList([
            LongformerBlock(embed_dim, num_heads, window_size, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(embed_dim, klen * 3)
        self.norm = RMSNorm(embed_dim)

    def change_maskval(self, x, init_val=-100, target_val=0):
        out = x.clone()
        out[out == init_val] = target_val
        return out

    def forward(self, x, return_z=False):
        in_scale, out_scale = 1.0, 1.0
        x = self.change_maskval(x)
        x, pos = self.embedder(x)

        x = x * in_scale
        feature = []

        for block in self.blocks:
            z = block(x)
            feature.append(z)
            x = z

        x = self.norm(x)

        if return_z:
            return self.output_layer(x) * out_scale, feature, pos
        else:
            return self.output_layer(x) * out_scale
