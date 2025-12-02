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

        self.embedder = EmbedderAdd(pe_method='nerf', embed_dim=input_dim, learnable_projection=False)
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
            x = torch.einsum('bsnd,n->bsd', x, weights)
        else:
            x = self.embedder(x)

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
                 num_feature_layers=15, num_output_dim=5, return_embedding=False):
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
        self.out_mlp = MLPHead(embed_dim, num_output_dim)

        self.embedder = EmbedderAdd(pe_method='nerf', embed_dim=input_dim, learnable_projection=False)
        self.weighted_avg_weights = nn.Parameter(torch.ones(num_feature_layers))


    def forward(self, x, feature=None, padding_mask=None, pretrain=False):
        # Input processing
        if pretrain:
            x = feature.permute(1, 2, 0, 3)
            weights = torch.softmax(self.weighted_avg_weights, dim=0)
            x = torch.einsum('bsnd,n->bsd', x, weights)
        else:
            x = self.embedder(x)
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
                 num_feature_layers=15, num_output_dim=5, return_embedding=False):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.return_embedding = return_embedding

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
        self.out_mlp = MLPHead(embed_dim, num_output_dim)

        self.embedder = EmbedderAdd(pe_method='nerf', embed_dim=input_dim, learnable_projection=False)
        self.weighted_avg_weights = nn.Parameter(torch.ones(num_feature_layers))


    def forward(self, x, feature=None, padding_mask=None, pretrain=False):
        # Input processing
        if pretrain:
            x = feature.permute(1, 2, 0, 3)
            weights = torch.softmax(self.weighted_avg_weights, dim=0)
            x = torch.einsum('bsnd,n->bsd', x, weights)
        else:
            x = self.embedder(x)
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
        # Process through SA layers
        for layer in self.SA_layers:
            x = layer(x, key_padding_mask=~padding_mask)
        x = self.norm(x)
        x = x.transpose(0, 1) #(B,N,D)
        #noise prediction
        out_logits = self.out_mlp(x) # (B, N, num_output_dim)
        
        return {
            'pred_logits': out_logits,  # (B, N, num_output_dim)
            'embedding_pre_projection': embedding_pre_projection,
            'embedding_post_projection': embedding_post_projection
        }