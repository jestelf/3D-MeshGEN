import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# PartCrafter architecture
class PartCrafterModel(nn.Module):
    def __init__(self, max_parts=8, tokens_per_part=64, d_model=256, n_heads=8, num_blocks=6, freeze_backbone: bool = True):
        """
        PartCrafter model: generates part-wise shape tokens with local-global attention.
        - max_parts: maximum number of parts
        - tokens_per_part: number of latent tokens per part
        - d_model: dimension of token embeddings and transformer features
        - n_heads: number of attention heads for multi-head attention
        - num_blocks: number of transformer blocks (alternate local and global attention)
        - freeze_backbone: freeze pretrained image backbone weights if True
        """
        super(PartCrafterModel, self).__init__()
        self.max_parts = max_parts
        self.tokens_per_part = tokens_per_part
        self.d_model = d_model
        # Image encoder: use ResNet18 conv layers to get feature map (512 channels, 7x7 spatial for 224x224 input)
        resnet = models.resnet18(pretrained=True)
        # Use feature extractor (everything except avgpool & fc)
        self.image_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        for p in self.image_conv.parameters():
            # Optionally freeze pretrained backbone
            p.requires_grad = not freeze_backbone
        self.img_proj = nn.Linear(512, d_model)  # project image feature dimension to match tokens
        # Learnable initial tokens for each part (max_parts * tokens_per_part, d_model)
        self.init_part_tokens = nn.Parameter(torch.randn(max_parts * tokens_per_part, d_model))
        # Learnable part ID embeddings (to give each part a distinct identity)
        self.part_id_embeddings = nn.Parameter(torch.randn(max_parts, d_model))
        # Transformer components
        self.global_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.local_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # Layer normalization for stability
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.num_blocks = num_blocks
        # Final linear to decode each token to a 3D point coordinate
        self.token_to_coord = nn.Linear(d_model, 3)
    
    def forward(self, images):
        """
        Forward pass: images -> (max_parts, points_per_part, 3) predicted point cloud for each part.
        """
        batch_size = images.size(0)
        # Image encoding
        feat_map = self.image_conv(images)             # [B, 512, 7, 7] feature map
        feat_map = feat_map.view(batch_size, 512, -1)  # [B, 512, 49]
        feat_map = feat_map.permute(0, 2, 1)           # [B, 49, 512], each of 49 tokens is a 512-dim vector
        img_tokens = self.img_proj(feat_map)           # [B, 49, d_model], project to transformer dimension
        # Initialize shape tokens for all parts
        # Start from the learned base tokens (broadcast to batch)
        shape_tokens = self.init_part_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [B, max_parts*tokens_per_part, d_model]
        # Add part identity embeddings to corresponding token groups
        if self.part_id_embeddings is not None:
            part_ids = self.part_id_embeddings.unsqueeze(1).expand(self.max_parts, self.tokens_per_part, self.d_model)
            part_ids = part_ids.reshape(self.max_parts * self.tokens_per_part, self.d_model)  # [max_parts*tokens_per_part, d_model]
            shape_tokens += part_ids.unsqueeze(0)  # broadcast to batch
        # Transformers: alternate global and local attention blocks
        # shape_tokens shape is [B, N_tokens, d_model] where N_tokens = max_parts * tokens_per_part
        for i in range(self.num_blocks):
            if i % 2 == 0:
                # Global attention block: tokens attend to all other tokens (across parts)
                # Self-attention (global)
                attn_out, _ = self.global_attn(query=shape_tokens, key=shape_tokens, value=shape_tokens)
                shape_tokens = self.norm1[i](shape_tokens + attn_out)  # add & norm
            else:
                # Local attention block: each part's tokens attend only within that part
                # We'll do this by splitting tokens for each part and applying attention separately
                B, N, D = shape_tokens.shape
                attn_out_tokens = torch.zeros_like(shape_tokens)
                # Reshape to [B, max_parts, tokens_per_part, D] for convenience
                part_tokens = shape_tokens.reshape(B, self.max_parts, self.tokens_per_part, D)
                for pj in range(self.max_parts):
                    # extract tokens for part j
                    tokens_pj = part_tokens[:, pj, :, :]  # [B, tokens_per_part, D]
                    # Apply multi-head self-attention for this part
                    out_pj, _ = self.local_attn(query=tokens_pj, key=tokens_pj, value=tokens_pj)
                    attn_out_tokens[:, pj*self.tokens_per_part:(pj+1)*self.tokens_per_part, :] = out_pj
                shape_tokens = self.norm1[i](shape_tokens + attn_out_tokens)
            # Cross-attention: shape tokens attend to image tokens (image features as keys/values)
            attn_cross, _ = self.cross_attn(query=shape_tokens, key=img_tokens, value=img_tokens)
            shape_tokens = self.norm2[i](shape_tokens + attn_cross)
        # Decode each token to a 3D point coordinate
        coords = self.token_to_coord(shape_tokens)  # [B, max_parts*tokens_per_part, 3]
        # Reshape to separate parts: [B, max_parts, tokens_per_part, 3]
        coords = coords.view(batch_size, self.max_parts, self.tokens_per_part, 3)
        return coords

# Baseline: ShapeAsPoints++
class ShapeAsPointsPlusPlusModel(nn.Module):
    def __init__(self, points_per_shape=512, freeze_backbone: bool = True):
        """
        ShapeAsPoints++: generates a whole shape point cloud from image (no part structure).
        - points_per_shape: number of points to output for the whole shape.
        - freeze_backbone: freeze pretrained image backbone weights if True
        """
        super(ShapeAsPointsPlusPlusModel, self).__init__()
        self.points_per_shape = points_per_shape
        # Image encoder: we use ResNet18 and take the final average pooled features as a global image descriptor
        resnet = models.resnet18(pretrained=True)
        # We will use the entire model up to avgpool to get a 512-dim feature
        self.feature_extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool  # global average pool -> [B, 512, 1, 1]
        )
        for p in self.feature_extractor.parameters():
            # Optionally freeze pretrained backbone
            p.requires_grad = not freeze_backbone
        self.feature_dim = 512
        # Fully connected layers to map image feature -> flattened point cloud coordinates
        hidden_dim = 1024
        self.fc1 = nn.Linear(self.feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, points_per_shape * 3)  # output 3 coords per point
    
    def forward(self, images):
        B = images.size(0)
        # Extract global image feature
        feat = self.feature_extractor(images)       # [B, 512, 1, 1]
        feat = feat.view(B, -1)                     # [B, 512]
        x = F.relu(self.fc1(feat))
        coords = self.fc2(x)                        # [B, points_per_shape*3]
        coords = coords.view(B, self.points_per_shape, 3)  # [B, points_per_shape, 3]
        return coords

# Baseline: PointCRAFT++
class PointCraftPlusPlusModel(nn.Module):
    def __init__(self, max_parts=8, tokens_per_part=64, d_model=256, n_heads=8, num_blocks=6, freeze_backbone: bool = True):
        """
        PointCRAFT++: generates each part independently (no global part-part attention).
        - Similar parameters as PartCrafterModel, but uses only local attention blocks.
        - freeze_backbone: freeze pretrained image backbone weights if True
        """
        super(PointCraftPlusPlusModel, self).__init__()
        self.max_parts = max_parts
        self.tokens_per_part = tokens_per_part
        self.d_model = d_model
        # Image encoder (ResNet18 conv features)
        resnet = models.resnet18(pretrained=True)
        self.image_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        for p in self.image_conv.parameters():
            # Optionally freeze pretrained backbone
            p.requires_grad = not freeze_backbone
        self.img_proj = nn.Linear(512, d_model)
        # Initial tokens and part ID embeddings (similar to PartCrafter)
        self.init_part_tokens = nn.Parameter(torch.randn(max_parts * tokens_per_part, d_model))
        self.part_id_embeddings = nn.Parameter(torch.randn(max_parts, d_model))
        # Only local self-attention layers
        self.local_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.num_blocks = num_blocks
        # Output decoder
        self.token_to_coord = nn.Linear(d_model, 3)
    
    def forward(self, images):
        B = images.size(0)
        # Image feature tokens
        feat_map = self.image_conv(images)          # [B, 512, 7, 7]
        feat_map = feat_map.view(B, 512, -1).permute(0, 2, 1)  # [B, 49, 512]
        img_tokens = self.img_proj(feat_map)        # [B, 49, d_model]
        # Initialize part tokens (with identity embeddings)
        shape_tokens = self.init_part_tokens.unsqueeze(0).expand(B, -1, -1).clone()  # [B, N_tokens, d_model]
        if self.part_id_embeddings is not None:
            part_ids = self.part_id_embeddings.unsqueeze(1).expand(self.max_parts, self.tokens_per_part, self.d_model)
            part_ids = part_ids.reshape(self.max_parts * self.tokens_per_part, self.d_model)
            shape_tokens += part_ids.unsqueeze(0)
        # Only local-attention blocks (no global mixing)
        for i in range(self.num_blocks):
            # Local self-attention for each part
            B_, N, D = shape_tokens.shape
            attn_out_tokens = torch.zeros_like(shape_tokens)
            part_tokens = shape_tokens.reshape(B_, self.max_parts, self.tokens_per_part, D)
            for pj in range(self.max_parts):
                tokens_pj = part_tokens[:, pj, :, :]
                out_pj, _ = self.local_attn(query=tokens_pj, key=tokens_pj, value=tokens_pj)
                attn_out_tokens[:, pj*self.tokens_per_part:(pj+1)*self.tokens_per_part, :] = out_pj
            shape_tokens = self.norm1[i](shape_tokens + attn_out_tokens)
            # Cross-attention with image for all tokens combined
            attn_cross, _ = self.cross_attn(query=shape_tokens, key=img_tokens, value=img_tokens)
            shape_tokens = self.norm2[i](shape_tokens + attn_cross)
        coords = self.token_to_coord(shape_tokens)                   # [B, max_parts*tokens_per_part, 3]
        coords = coords.view(B, self.max_parts, self.tokens_per_part, 3)  # [B, max_parts, tokens_per_part, 3]
        return coords
