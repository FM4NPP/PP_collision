"""
Linformer model for particle physics pretraining
Uses low-rank projection for linear complexity O(n*k) where k << n
Target: 2M-5M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fm4npp.models.embed import EmbedderAdd, EmbedderConcat
from fm4npp.models.rmsnorm import RMSNorm


class LinformerAttention(nn.Module):
    """Linformer attention with low-rank projection"""
    def __init__(self, embed_dim, num_heads=4, seq_len=512, proj_dim=256, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.proj_dim = proj_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Low-rank projection matrices E and F
        self.E = nn.Parameter(torch.randn(seq_len, proj_dim) / math.sqrt(proj_dim))
        self.F = nn.Parameter(torch.randn(seq_len, proj_dim) / math.sqrt(proj_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Project K and V to lower dimension
        # Handle variable sequence lengths by truncating/padding projection matrices
        if N <= self.seq_len:
            E = self.E[:N, :]
            F_proj = self.F[:N, :]
        else:
            # For sequences longer than expected, use interpolation
            import torch.nn.functional as F_func
            E = F_func.interpolate(self.E.unsqueeze(0).unsqueeze(0), size=(N, self.proj_dim), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            F_proj = F_func.interpolate(self.F.unsqueeze(0).unsqueeze(0), size=(N, self.proj_dim), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        # Project K: (B, num_heads, N, head_dim) -> (B, num_heads, proj_dim, head_dim)
        k_proj = torch.einsum('bhnd,np->bhpd', k, E)

        # Project V: (B, num_heads, N, head_dim) -> (B, num_heads, proj_dim, head_dim)
        v_proj = torch.einsum('bhnd,np->bhpd', v, F_proj)

        # Compute attention with projected K
        scale = self.head_dim ** -0.5
        attn = (q @ k_proj.transpose(-2, -1)) * scale  # (B, num_heads, N, proj_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to projected V
        out = attn @ v_proj  # (B, num_heads, N, head_dim)

        # Reshape
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return out


class LinformerBlock(nn.Module):
    """Linformer transformer block"""
    def __init__(self, embed_dim, num_heads=4, seq_len=512, proj_dim=256, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = LinformerAttention(embed_dim, num_heads, seq_len, proj_dim, dropout)
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


class LinformerGPT(nn.Module):
    """
    Linformer model matching MambaGPT interface

    Parameters for ~3M params:
    - embed_dim: 256
    - num_layers: 6
    - num_heads: 4
    - seq_len: 512
    - proj_dim: 256
    """
    def __init__(self, embed_dim=256, num_layers=6, num_heads=4, seq_len=512, proj_dim=256,
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
            LinformerBlock(embed_dim, num_heads, seq_len, proj_dim, mlp_ratio, dropout)
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
