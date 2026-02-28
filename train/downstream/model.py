import torch
import torch.nn as nn

from fm4npp.models.embed import *
from fm4npp.models.rmsnorm import RMSNorm
from fm4npp.models.mamba2 import Mamba2



class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, prenorm=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.prenorm = prenorm

    def forward(self, x, key_padding_mask=None):
        if self.prenorm:
            x_norm = self.norm(x)
            attn_output, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
            return x + attn_output
        else:
            attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
            return self.norm(x + attn_output)


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, prenorm=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.prenorm = prenorm

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        if key_padding_mask is not None and key_padding_mask.dtype == torch.bool and attn_mask is not None and attn_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.float()
            key_padding_mask = key_padding_mask.masked_fill(key_padding_mask == 1, float('-inf'))
            key_padding_mask = key_padding_mask.masked_fill(key_padding_mask == 0, float(0.0))

        if self.prenorm:
            query_norm = self.norm(query)
            attn_output, _ = self.attn(query_norm, key, value,
                                       key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
            return query + attn_output
        else:
            attn_output, _ = self.attn(query, key, value,
                                       key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
            return self.norm(query + attn_output)


class FFNBlock(nn.Module):
    def __init__(self, embed_dim, ffn_dim, prenorm=True):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x_norm = self.norm(x)
            return x + self.ffn(x_norm)
        else:
            return self.norm(x + self.ffn(x))


class SetTransformerPooling(nn.Module):
    """
    Lightweight Set Transformer block that applies a stack of
    self-attention blocks followed by pooling via multihead attention
    with learnable seed vectors (PMA).
    """

    def __init__(
        self,
        embed_dim,
        num_heads=4,
        num_sab_layers=1,
        num_seeds=4,
        ffn_dim=None,
        dropout=0.1,
    ):
        super().__init__()
        ffn_dim = ffn_dim or 2 * embed_dim
        self.sab_layers = nn.ModuleList([
            nn.ModuleList([
                SelfAttentionBlock(embed_dim, num_heads, dropout=dropout, prenorm=True),
                FFNBlock(embed_dim, ffn_dim, prenorm=True),
            ])
            for _ in range(num_sab_layers)
        ])
        self.num_seeds = num_seeds
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, embed_dim))
        self.pma_attn = CrossAttentionBlock(embed_dim, num_heads, dropout=dropout, prenorm=True)
        self.pma_ffn = FFNBlock(embed_dim, ffn_dim, prenorm=True)

    def forward(self, x, mask):
        """
        Args:
            x: (B, N, D)
            mask: (B, N) boolean mask (True for valid entries)
        Returns:
            pooled: (B, D)
        """
        pad_mask = None
        if mask is not None:
            pad_mask = ~mask

        seq = x.transpose(0, 1)  # (N, B, D)
        for self_attn, ffn in self.sab_layers:
            seq = self_attn(seq, key_padding_mask=pad_mask)
            seq = ffn(seq)

        seeds = self.seed_vectors.unsqueeze(1).expand(-1, x.size(0), -1)  # (K, B, D)
        pooled_seq = self.pma_attn(seeds, seq, seq, key_padding_mask=pad_mask)
        pooled_seq = self.pma_ffn(pooled_seq)  # (K, B, D)
        pooled = pooled_seq.transpose(0, 1).mean(dim=1)  # (B, D)
        return pooled

class MLPHead(nn.Module):
    def __init__(self, embed_dim, output_dim, dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class RefinementLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(embed_dim, num_heads)
        self.self_attn = SelfAttentionBlock(embed_dim, num_heads)
        self.ffn = FFNBlock(embed_dim, ffn_dim)
    
    def forward(self, refined_protos, context, pos_emb, padding_mask=None, attn_mask=None):
        # Cross-attention with transformed points
        q_cross = refined_protos + pos_emb


        refined_protos = self.cross_attn(
            q_cross, context, context,
            key_padding_mask=~padding_mask if padding_mask is not None else None,
            attn_mask=attn_mask if attn_mask is not None else None
        )
        #self-attention
        refined_protos = self.self_attn(refined_protos)
        # FFN processing
        return self.ffn(refined_protos)

class MambaAttentionHead(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_layers=3, d_state=64, d_conv=4, expand=2, 
                 num_feature_layers=15, num_output_dim=256, num_prototypes=10, num_heads=4, ffn_dim=512,
                 num_pid_classes=5, num_track_features= 4, num_jet_preposoals=5, num_jet_features=4,
                 num_self_attn_layers=2, softmax_mask=False, do_masked_attn=True, do_holistic=True, return_embedding=False):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_prototypes = num_prototypes
        self.softmax_mask = softmax_mask
        self.do_masked_attn = do_masked_attn
        self.num_heads = num_heads
        self.do_holistic = do_holistic
        self.return_embedding = return_embedding

        # Input processing
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim)
        )

        # Prototype embeddings
        self.prototype_embed = nn.Embedding(num_prototypes, embed_dim)
        self.pos_emb_embed = nn.Embedding(num_prototypes, embed_dim)

        # Mamba backbone
        self.mamba_layers = nn.ModuleList([
            nn.Sequential(
                RMSNorm(embed_dim),
                Mamba2(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

        # Output transformation (should use MLP)
        self.output_layer = MLPHead(embed_dim, num_output_dim)

        # Prototype refinement
        self.refinement_layers = nn.ModuleList([
            RefinementLayer(embed_dim, num_heads, ffn_dim)
            for _ in range(num_self_attn_layers)
        ])

        # Prediction heads
        self.class_mlp = MLPHead(embed_dim, 2)
        self.mask_mlp = MLPHead(embed_dim, num_output_dim)
        self.pid_mlp = MLPHead(embed_dim, num_pid_classes + 1)  # +1 class that are not in the PID target
        self.track_mlp = MLPHead(embed_dim, num_track_features) # (q/(pT + 1 ), eta, sin(phi), cos(phi))
        # Noise prediction head go from point embedding
        self.noise_mlp = MLPHead(embed_dim, 2)

        self.embedder = Embedder(embed_dim=input_dim)
        self.weighted_avg_weights = nn.Parameter(torch.ones(num_feature_layers))

    def make_predictions(self, refined_protos, point_features):
        # Prepare prototypes
        refined_protos = refined_protos.transpose(0, 1)  # -> (B, C, D)
        class_logits = self.class_mlp(refined_protos)  # -> (B, C, 2)
        class_probs = torch.softmax(class_logits, dim=-1)  # -> (B, C, 2)
        mask_prototypes = self.mask_mlp(refined_protos)   # -> (B, C, D)

    
        # Similarity: (B, N, D) @ (B, C, D)^T -> (B, N, C)
        similarity = torch.einsum('bnd, bcd -> bnc', point_features, mask_prototypes)
    
        # Compute mask probabilities
        if not self.softmax_mask:
            mask_probs = torch.sigmoid(similarity)
        else:
            mask_probs = torch.softmax(similarity, dim=-1)
        return class_probs, mask_probs

    def cal_attn_mask(self, mask_probs, padding_mask=None, mode='soft', eps=1e-6):
        """
        Compute attention mask for MHA.

        Args:
            mask_probs: Float tensor of shape (B, N, C), where N is the number of points
            padding_mask: Bool tensor of shape (B, N), True for non-padded positions
            mode: 'hard' for boolean mask, 'soft' for additive mask
            eps: small epsilon for numerical stability in log
    
        Returns:
            attn_mask:
              if mode=='hard': BoolTensor of shape (B*H, C, N)
              if mode=='soft': FloatTensor of shape (B*H, C, N) with additive biases
        """
    
        # Expand over heads: (B, N, C) -> (B, H, C, N)
        B, N, C = mask_probs.shape
        H = self.num_heads
        mask_probs_heads = mask_probs.unsqueeze(1).expand(-1, H, -1, -1)
        mask_probs_heads = mask_probs_heads.permute(0, 1, 3, 2).reshape(B * H, C, N)
    
        # Handle padding mask
        if padding_mask is not None:
            # Invert so True=mask, False=keep
            pad_inv = ~padding_mask  # (B, N)
            pad_heads = pad_inv.unsqueeze(1).expand(-1, H, -1).reshape(B * H, 1, N)
        
        if mode == 'hard':
            # Hard boolean mask
            attn_mask = (mask_probs_heads < 0.5)
            if padding_mask is not None:
                attn_mask = attn_mask | pad_heads.bool()
    
            # Ensure no-all-mask
            all_masked = attn_mask.all(dim=-1)
            if all_masked.any():
                attn_mask = attn_mask.clone()
                b_idx, c_idx = torch.nonzero(all_masked, as_tuple=True)
                attn_mask[b_idx, c_idx, 0] = False
    
        elif mode == 'soft':
            # Soft additive mask: log-prob biases
            bias = torch.log(mask_probs_heads + eps)  # negative bias for prob<1
            if padding_mask is not None:
                # assign large negative bias to padded keys
                bias = bias.masked_fill(pad_heads.bool(), float('-1e9'))
            attn_mask = bias
    
        else:
            raise ValueError(f"Unknown mode {mode}, choose 'hard' or 'soft'.")
    
        return attn_mask

    def forward(self, x, feature=None, padding_mask=None, pretrain=False):
        # Input processing
        if pretrain:
            x = feature.permute(1, 2, 0, 3)
            weights = torch.softmax(self.weighted_avg_weights, dim=0)
            # Ensure dtype consistency for einsum operation
            weights = weights.to(x.dtype)
            x = torch.einsum('bsnd,n->bsd', x, weights)
        else:
            x = self.embedder(x)
            if isinstance(x, tuple):
                x = x[0]

        embedding_pre_proj = None
        embedding_post_proj = None
        if self.return_embedding:
            embedding_pre_proj = x

        x = self.input_proj(x)
        if self.return_embedding:
            embedding_post_proj = x

        # Process through Mamba layers
        for layer in self.mamba_layers:
            x = layer(x) + x
        x = self.norm(x)

        # Generate transformed points using FFN
        transformed_points = self.output_layer(x)
        #noise prediction
        noise_logits = self.noise_mlp(x) # (B, N, 2)
        context = transformed_points.transpose(0, 1)  # (N, B, D)
        
        # Initialize prototypes
        indices = torch.arange(self.num_prototypes, device=x.device)
        prototypes = self.prototype_embed(indices).unsqueeze(1).expand(-1, x.size(0), -1) # (C, B, D)
        pos_emb = self.pos_emb_embed(indices).unsqueeze(1).expand(-1, x.size(0), -1) # (C, B, D)
        # for aux loss
        aux_class_probs = []
        aux_mask_probs = []
        class_probs, mask_probs = self.make_predictions(prototypes, transformed_points)
        aux_class_probs.append(class_probs)
        aux_mask_probs.append(mask_probs)
        attn_mask = None
        if self.do_masked_attn:
            # Compute attention mask
            attn_mask = self.cal_attn_mask(mask_probs, padding_mask)
        refined_protos = prototypes

        # Prototype refinement using transformed points
        for layer in self.refinement_layers:
            refined_protos = layer(refined_protos, context, pos_emb, padding_mask, attn_mask = attn_mask)
            class_probs, mask_probs = self.make_predictions(refined_protos, transformed_points)
            aux_class_probs.append(class_probs)
            aux_mask_probs.append(mask_probs)
            if self.do_masked_attn:
                attn_mask = self.cal_attn_mask(mask_probs, padding_mask)


        # Final predictions
        class_probs, mask_probs = self.make_predictions(refined_protos, transformed_points)

        refined_protos = refined_protos.transpose(0, 1)

        if self.do_holistic:
            # Holistic predictions
            track_reg_result = self.track_mlp(refined_protos)  # (B, C, num_track_features)
            pid_logits = self.pid_mlp(refined_protos)  # (B, C, num_pid_classes + 1)
        else:
            #per point
            track_reg_result = self.track_mlp(x)  # (B, N, num_track_features)
            pid_logits = self.pid_mlp(x) # (B, N, num_pid_classes + 1)

        aux_list = [
            {"class_probs": c, "mask_probs": m}
            for c, m in zip(aux_class_probs[:-1], aux_mask_probs[:-1])
        ]

        return {
            'class_probs': class_probs,
            'mask_probs': mask_probs,
            'track_reg_result': track_reg_result, # (B, C, num_track_features) # no holistic then (B, N, num_track_features)
            'pid_logits': pid_logits, # (B, C, num_pid_classes + 1) # no holistic then (B, N, num_pid_classes + 1)
            'noise_logits': noise_logits, # (B, N, 2)
            'aux_list': aux_list,
            'embedding_pre_proj': embedding_pre_proj,
            'embedding_post_proj': embedding_post_proj,
        }
    
# very simple adapter head with few mamba layer for refinement and MLP head
class MambaHead(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_layers=3, d_state=64, d_conv=4, expand=2, 
                 num_embedder_layers=1, d_state_embedder=64, d_conv_embedder=4, expand_embedder=2,
                 num_feature_layers=15, num_output_dim=5, return_embedding=False, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.return_embedding = return_embedding

        # Input processing
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim)
        )


        # Mamba feature extractor
        self.mamba_layers = nn.ModuleList([
            nn.Sequential(
                RMSNorm(embed_dim),
                Mamba2(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
            ) for _ in range(num_layers)
        ])
        # if not using the pretrained mamba2, we can use the embedder layers
        self.mamba_embedder_layers = nn.ModuleList([
            nn.Sequential(
                RMSNorm(input_dim),
                Mamba2(
                    d_model=input_dim,
                    d_state=d_state_embedder,
                    d_conv=d_conv_embedder,
                    expand=expand_embedder
                )
            ) for _ in range(num_embedder_layers)
        ])
        self.embedder_norm = RMSNorm(input_dim)
        self.norm = RMSNorm(embed_dim)

        # Noise prediction head go from point embedding
        self.out_mlp = MLPHead(embed_dim, num_output_dim, dropout=dropout)

        self.embedder = Embedder(embed_dim=input_dim)
        self.weighted_avg_weights = nn.Parameter(torch.ones(num_feature_layers))


    def forward(self, x, feature=None, padding_mask=None, pretrain=False):
        # Input processing
        if pretrain:
            x = feature.permute(1, 2, 0, 3)
            weights = torch.softmax(self.weighted_avg_weights, dim=0)
            # Ensure dtype consistency for einsum operation
            weights = weights.to(x.dtype)
            x = torch.einsum('bsnd,n->bsd', x, weights)
        else:
            x = self.embedder(x)
            if isinstance(x, tuple):
                x = x[0]
            for layer in self.mamba_embedder_layers:
                x = layer(x) + x
            x = self.embedder_norm(x)
        
        embedding_pre_projection = None
        embedding_post_projection = None
        if self.return_embedding:
            embedding_pre_projection = x

        x = self.input_proj(x)

        if self.return_embedding:
            embedding_post_projection = x
        
        # Process through Mamba layers
        for layer in self.mamba_layers:
            x = layer(x) + x
        x = self.norm(x)

        #noise prediction
        out_logits = self.out_mlp(x) # (B, N, num_output_dim)
        
        return {
            'pred_logits': out_logits,  # (B, N, num_output_dim)
            'embedding_pre_projection': embedding_pre_projection,
            'embedding_post_projection': embedding_post_projection
        }

# very simple adapter head with few SA for refinement and MLP head
class AttentionHead(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_layers=3, num_heads = 4, 
                 num_embedder_layers=1, d_state_embedder=64, d_conv_embedder=4, expand_embedder=2,
                 num_feature_layers=15, num_output_dim=5, return_embedding=False, embed_method='add', pe_method = 'nerf', dropout=0.0, ffn_dim=512, adapter_FFN = True):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.return_embedding = return_embedding
        self.FFN = adapter_FFN
        if embed_method == 'concat':
            Embedder = EmbedderConcat
        else:
            Embedder = EmbedderAdd

        # Input processing
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim)
        )


        # SA feature extractor
        self.SA_layers = nn.ModuleList([
            SelfAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])
        #ffn layers
        if self.FFN:
            self.ffn_layers = nn.ModuleList([
                FFNBlock(
                    embed_dim=embed_dim,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_layers)
            ])
        else:
            self.ffn_layers = []
        # if not using the pretrained mamba2, we can use the embedder layers
        self.mamba_embedder_layers = nn.ModuleList([
            nn.Sequential(
                RMSNorm(input_dim),
                Mamba2(
                    d_model=input_dim,
                    d_state=d_state_embedder,
                    d_conv=d_conv_embedder,
                    expand=expand_embedder
                )
            ) for _ in range(num_embedder_layers)
        ])
        self.embedder_norm = RMSNorm(input_dim)
        self.norm = RMSNorm(embed_dim)

        # Noise prediction head go from point embedding
        self.out_mlp = MLPHead(embed_dim, num_output_dim, dropout=dropout)

        self.embedder = Embedder(pe_method = 'nerf', embed_dim=input_dim)
        self.weighted_avg_weights = nn.Parameter(torch.ones(num_feature_layers))


    def forward(self, x, feature=None, padding_mask=None, pretrain=False):
        # Input processing
        if pretrain:
            x = feature.permute(1, 2, 0, 3)
            weights = torch.softmax(self.weighted_avg_weights, dim=0)
            # Ensure dtype consistency for einsum operation
            weights = weights.to(x.dtype)
            x = torch.einsum('bsnd,n->bsd', x, weights)
        else:
            x = self.embedder(x)
            if isinstance(x, tuple):
                x = x[0]
            for layer in self.mamba_embedder_layers:
                x = layer(x) + x
            x = self.embedder_norm(x)
        
        embedding_pre_projection = None
        embedding_post_projection = None
        if self.return_embedding:
            embedding_pre_projection = x

        x = self.input_proj(x) #(B,N,D)

        

        if self.return_embedding:
            embedding_post_projection = x
        
        x = x.transpose(0, 1) #(N,B,D)
        # Process through SA + FFN layers
        if self.FFN:
            for sa_layer, ffn_layer in zip(self.SA_layers, self.ffn_layers):
                x = sa_layer(x, key_padding_mask=~padding_mask)
                x = ffn_layer(x)
        else:
            for sa_layer in self.SA_layers:
                x = sa_layer(x, key_padding_mask=~padding_mask)
                x = x
        x = self.norm(x)
        x = x.transpose(0, 1) #(B,N,D)
        #noise prediction
        out_logits = self.out_mlp(x) # (B, N, num_output_dim)

        return {
            'pred_logits': out_logits,  # (B, N, num_output_dim)
            'embedding_pre_projection': embedding_pre_projection,
            'embedding_post_projection': embedding_post_projection
        }


class EventClassificationHead(nn.Module):
    """
    Event-level classification head for binary weak decay detection.
    Uses transformer self-attention + FFN to aggregate point-level features
    into a single event-level prediction.
    """
    def __init__(
        self,
        input_dim=256,
        embed_dim=256,
        num_layers=1,
        num_heads=4,
        ffn_dim=512,
        num_feature_layers=12,
        num_output_classes=2,
        pooling='set_transformer',  # 'mean', 'max', 'both', 'learnable', 'set_transformer'
        dropout=0.1,
        return_embedding=False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        allowed_poolings = {'mean', 'max', 'both', 'learnable', 'set_transformer'}
        if pooling not in allowed_poolings:
            raise ValueError(f"Unsupported pooling '{pooling}'. Choose from {allowed_poolings}.")
        self.pooling = pooling
        self.return_embedding = return_embedding

        # Weighted aggregation of pretrained backbone layers
        self.weighted_avg_weights = nn.Parameter(torch.ones(num_feature_layers))

        # Input projection
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim)
        )

        # Transformer layers (self-attention + FFN)
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': SelfAttentionBlock(embed_dim, num_heads, dropout=dropout, prenorm=True),
                'ffn': FFNBlock(embed_dim, ffn_dim, prenorm=True)
            })
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Set Transformer pooling setup (keeps output dim = embed_dim)
        if self.pooling == 'set_transformer':
            self.set_pool = SetTransformerPooling(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_sab_layers=1,
                num_seeds=4,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )

        # Classification head
        # If pooling='both', input is 2*embed_dim (mean + max concatenated)
        pool_dim = 2 * embed_dim if pooling == 'both' else embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_output_classes)
        )

        if self.pooling == 'learnable':
            hidden_dim = max(embed_dim // 2, 1)
            self.pool_attn = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )

    def global_pool(self, x, mask):
        """
        Global pooling with padding mask handling.

        Args:
            x: (B, N, D) point features
            mask: (B, N) boolean mask (True for valid, False for padded)

        Returns:
            pooled: (B, D) or (B, 2*D) depending on self.pooling
        """
        if self.pooling == 'set_transformer':
            return self.set_pool(x, mask)
        if self.pooling == 'learnable':
            attn_logits = self.pool_attn(x).squeeze(-1)  # (B, N)
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
            attn_weights = torch.softmax(attn_logits, dim=1)
            attn_weights = attn_weights.masked_fill(~mask, 0.0)

            no_valid = (~mask).all(dim=1)
            if no_valid.any():
                attn_weights = attn_weights.clone()
                attn_weights[no_valid] = 1.0 / attn_weights.size(1)

            pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=1)
            return pooled

        # Expand mask for broadcasting
        mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)

        pooled_features = []

        if self.pooling in ['mean', 'both']:
            # Mean pooling: masked sum / count
            masked_sum = (x * mask_expanded).sum(dim=1)  # (B, D)
            counts = mask_expanded.sum(dim=1).clamp(min=1)  # Avoid division by zero
            mean_pool = masked_sum / counts
            pooled_features.append(mean_pool)

        if self.pooling in ['max', 'both']:
            # Max pooling: set padded positions to -inf before max
            x_masked = x.clone()
            x_masked[~mask.unsqueeze(-1).expand_as(x)] = float('-inf')
            max_pool = x_masked.max(dim=1)[0]  # (B, D)
            # Handle edge case: all-padded (shouldn't happen, but be safe)
            max_pool[max_pool == float('-inf')] = 0
            pooled_features.append(max_pool)

        # Concatenate mean and max if using both
        return torch.cat(pooled_features, dim=-1)  # (B, D) or (B, 2*D)

    def forward(self, points, feature=None, pretrain=True, padding_mask=None):
        """
        Forward pass for event classification.

        Args:
            points: (B, N, 4) raw point features (not used directly, for API consistency)
            feature: (B, S, N, D) pretrained backbone features from multiple layers
            pretrain: bool, whether using pretrained features
            padding_mask: (B, N) boolean mask (True for valid, False for padded)

        Returns:
            dict with:
                'event_logits': (B, num_output_classes)
                'event_probs': (B, num_output_classes)
                'pooled_features': (B, pool_dim)
                'embedding_pre_projection': optional
                'embedding_post_projection': optional
        """
        if padding_mask is None:
            b, n = points.shape[:2]
            padding_mask = torch.ones(b, n, dtype=torch.bool, device=points.device)

        # Aggregate pretrained features using learnable weights
        if pretrain and feature is not None:
            # feature: (S,B, N, D) where S is num_feature_layers
            x = feature.permute(1, 2, 0, 3) #(B, N, S, D)
            weights = torch.softmax(self.weighted_avg_weights, dim=0)  # (S,)
            x = torch.einsum('bnsd,s->bnd', x, weights)  # (B, N, D)
        else:
            # For non-pretrain mode (shouldn't be used for this model, but for safety)
            x = points  # (B, N, D)

        # Store embeddings if requested
        embedding_pre_projection = x if self.return_embedding else None

        # Input projection
        x = self.input_proj(x)  # (B, N, embed_dim)

        embedding_post_projection = x if self.return_embedding else None

        # Transpose for PyTorch MultiheadAttention: (N, B, D)
        x = x.transpose(0, 1)

        # Process through transformer layers
        for layer_dict in self.transformer_layers:
            # Self-attention with padding mask
            x = layer_dict['self_attn'](x, key_padding_mask=~padding_mask)
            # FFN
            x = layer_dict['ffn'](x)

        # Final normalization
        x = self.norm(x)

        # Transpose back: (B, N, D)
        x = x.transpose(0, 1)

        # Global pooling to get event-level features
        pooled = self.global_pool(x, padding_mask)  # (B, pool_dim)

        # Classification
        event_logits = self.classifier(pooled)  # (B, num_output_classes)
        event_probs = torch.softmax(event_logits, dim=-1)

        return {
            'event_logits': event_logits,
            'event_probs': event_probs,
            'pooled_features': pooled,
            'embedding_pre_projection': embedding_pre_projection,
            'embedding_post_projection': embedding_post_projection
        }


class MultiTaskAttentionHead(nn.Module):
    """
    Multi-task attention head for joint PID + NID classification.

    Architecture:
    - Shared input projection (same for both tasks)
    - Shared self-attention + FFN layers
    - Separate task-specific output heads

    This maximizes parameter sharing while maintaining task-specific
    capacity only for final predictions.
    """

    def __init__(
        self,
        input_dim=256,
        embed_dim=256,
        num_shared_sa_layers=1,
        num_heads=4,
        ffn_dim=512,
        num_feature_layers=12,
        num_pid_classes=5,
        num_nid_classes=2,
        dropout=0.1,
        adapter_FFN=True,
        embed_method='add',
        pe_method='nerf',
        return_embedding=False
    ):
        """
        Args:
            input_dim: Dimension of backbone features
            embed_dim: Embedding dimension for SA layers
            num_shared_sa_layers: Number of shared SA+FFN blocks
            num_heads: Number of attention heads
            ffn_dim: Hidden dimension in FFN
            num_feature_layers: Number of backbone layers to aggregate
            num_pid_classes: Number of PID classes (5)
            num_nid_classes: Number of NID classes (2)
            dropout: Dropout probability
            adapter_FFN: Whether to include FFN after SA
            embed_method: Embedding method (for compatibility)
            pe_method: Position encoding method (for compatibility)
            return_embedding: Whether to return intermediate embeddings
        """
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_shared_sa_layers = num_shared_sa_layers
        self.return_embedding = return_embedding
        self.adapter_FFN = adapter_FFN

        # Weighted feature aggregation from backbone layers
        self.weighted_avg_weights = nn.Parameter(torch.ones(num_feature_layers))

        # Shared input projection (used by both tasks)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim)
        )

        # Shared self-attention layers
        self.shared_sa_layers = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads, dropout=dropout, prenorm=True)
            for _ in range(num_shared_sa_layers)
        ])

        # Shared FFN layers (if enabled)
        if adapter_FFN:
            self.shared_ffn_layers = nn.ModuleList([
                FFNBlock(embed_dim, ffn_dim, prenorm=True)
                for _ in range(num_shared_sa_layers)
            ])

        # Final normalization
        self.shared_norm = RMSNorm(embed_dim)

        # Task-specific classification heads
        self.pid_head = MLPHead(embed_dim, num_pid_classes, dropout=dropout)
        self.nid_head = MLPHead(embed_dim, num_nid_classes, dropout=dropout)

        # Embedding layers (reuse upstream embedder implementations)
        if embed_method == 'concat':
            embedder_cls = EmbedderConcat
        else:
            embedder_cls = EmbedderAdd
        self.embedder = embedder_cls(
            pe_method=pe_method,
            embed_dim=input_dim,
            learnable_projection=False
        )
        self.embedder_norm = RMSNorm(input_dim)
        # No additional Mamba embedder layers (num_embedder_layers=0)
        self.mamba_embedder_layers = nn.ModuleList([])

    def forward(self, x, feature=None, padding_mask=None, pretrain=True, task='both'):
        """
        Forward pass for multi-task learning.

        Args:
            x: Input features (B, N, 4) if not pretrain, not used if pretrain
            feature: Backbone features (S, B, N, D) if pretrain
            padding_mask: (B, N) boolean mask (True for valid, False for padded)
            pretrain: Whether using pretrained backbone
            task: 'pid', 'nid', or 'both' - which task(s) to compute

        Returns:
            dict with 'pid_logits' and/or 'nid_logits' depending on task
        """
        # Feature aggregation from backbone
        if pretrain and feature is not None:
            # feature: (S, B, N, D) where S = num_layers_backbone
            x = feature.permute(1, 2, 0, 3)  # (B, N, S, D)

            # Weighted average across backbone layers
            weights = torch.softmax(self.weighted_avg_weights, dim=0)
            weights = weights.to(x.dtype)
            x = torch.einsum('bnsd,s->bnd', x, weights)  # (B, N, D)
        else:
            # Not using pretrained features - encode from raw input
            x = self.embedder(x)  # (B, N, D)
            if isinstance(x, tuple):
                x = x[0]
            for layer in self.mamba_embedder_layers:
                x = layer(x) + x
            x = self.embedder_norm(x)

        # Shared input projection
        x = self.input_proj(x)  # (B, N, embed_dim)

        # Apply shared SA + FFN layers
        x = self._apply_shared_layers(x, padding_mask)

        # Task-specific heads
        outputs = {}

        if task in ['pid', 'both']:
            outputs['pid_logits'] = self.pid_head(x)  # (B, N, num_pid_classes)

        if task in ['nid', 'both']:
            outputs['nid_logits'] = self.nid_head(x)  # (B, N, num_nid_classes)

        return outputs

    def _apply_shared_layers(self, x, padding_mask):
        """
        Apply shared SA + FFN layers.

        Args:
            x: Input features (B, N, embed_dim)
            padding_mask: (B, N) boolean mask

        Returns:
            Processed features (B, N, embed_dim)
        """
        # Transpose for PyTorch MultiheadAttention (expects N, B, D)
        x = x.transpose(0, 1)  # (N, B, D)

        # Apply shared layers
        for i, sa_layer in enumerate(self.shared_sa_layers):
            # Self-attention with padding mask
            x = sa_layer(x, key_padding_mask=~padding_mask)

            # FFN (if enabled)
            if self.adapter_FFN:
                x = self.shared_ffn_layers[i](x)

        # Final normalization
        x = self.shared_norm(x)

        # Transpose back to (B, N, D)
        x = x.transpose(0, 1)

        return x