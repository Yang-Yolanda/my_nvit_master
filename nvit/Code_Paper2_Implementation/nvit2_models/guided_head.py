
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from hmr2.models.components.pose_transformer import TransformerDecoder, TransformerCrossAttn, DropTokenDropout, ZeroTokenDropout, PreNorm, FeedForward, Attention, CrossAttention
from hmr2.utils.geometry import rot6d_to_rotmat

class GuidedTransformerDecoder(nn.Module):
    """
    Modified TransformerDecoder that accepts dynamic queries.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: int = None,
    ):
        super().__init__()
        # We don't need token_embedding or pos_embedding here because 
        # queries are constructed externally and passed in.
        
        self.dropout = nn.Dropout(dropout)
        
        self.transformer = TransformerCrossAttn(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
        )

    def forward(self, queries, context):
        """
        queries: (B, N, D) - N=24 joints
        context: (B, L, D) - L=196 patches
        """
        x = queries
        # x = self.dropout(x) # Optional dropout on queries
        x = self.transformer(x, context=context)
        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        Sine/Cosine Positional Encoding for 2D coordinates.
        channels: total output dimension (must be divisible by 4)
        """
        super().__init__()
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels // 2, 2).float() / (channels // 2)))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coords):
        """
        coords: (B, N, 2) in [-1, 1]
        """
        # Map [-1, 1] to a suitable range for sine/cosine? 
        # Actually standard PE expects indices, but for continuous coords, we just multiply.
        x, y = coords[:, :, 0:1], coords[:, :, 1:2] # (B, N, 1)
        
        # x_sin: (B, N, channels // 4)
        sin_x = torch.sin(x * self.inv_freq)
        cos_x = torch.cos(x * self.inv_freq)
        sin_y = torch.sin(y * self.inv_freq)
        cos_y = torch.cos(y * self.inv_freq)
        
        # Concatenate: (B, N, channels)
        pos_embed = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)
        
        if not torch.isfinite(pos_embed).all():
             print(f"DEBUG: Found non-finite values in PositionalEncoding2D output. Coords max: {coords.max()}, min: {coords.min()}")
        
        return pos_embed

class GuidedSMPLHead(nn.Module):
    """
    DETR-style SMPL Head with 24 Joint Queries + Coordinate Guidance.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Dimensions
        self.dim = 1024 # Standard HMR2 dim
        self.num_joints = 24 
        

        # 1. Learnable Static Queries
        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints, self.dim))
        
        # [NEW] Learnable Guidance Scale
        self.guidance_scale = nn.Parameter(torch.ones(1) * 0.1) 
        
        # [Ablation Flags]
        self.heatmap_only = cfg.MODEL.get('heatmap_only', False)
        self.indexing_only = cfg.MODEL.get('indexing_only', False)

        # 2. Coordinate Encoder (Fourier PE + Projection)
        self.pe_2d = PositionalEncoding2D(self.dim // 4) # 256 dims of sine/cosine
        self.coord_encoder = nn.Sequential(
            nn.Linear(self.dim // 4, self.dim // 2),
            nn.GELU(),
            nn.Linear(self.dim // 2, self.dim)
        )
        
        # 3. Transformer Decoder
        # Depth=2? 3? HMR2 uses 6? Let's check config. 
        # Default to 3 for speed in debug, or match HMR2.
        decoder_depth = cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER.depth
        decoder_heads = cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER.heads
        decoder_mlp_dim = cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER.mlp_dim
        
        self.decoder = GuidedTransformerDecoder(
            dim=self.dim,
            depth=decoder_depth,
            heads=decoder_heads,
            mlp_dim=decoder_mlp_dim,
            context_dim=self.dim # Patches are also 1024? Check embed_dim.
        )
        # Note: If backbone embed_dim != 1024, we need projection.
        # HMR2 (ViT-H) embed_dim is 1280. self.dim is 1024.
        self.context_proj = nn.Linear(1280, self.dim)
        
        # 4. Readout Heads (Per Joint)
        # Each query predicts [Rot6D (6)] + [Shape (10)] + [Cam (3)]?
        # Typically Shape and Cam are Global.
        # Strategy: 
        #   - Joints 1-24 predict Rot6D.
        #   - Joint 0 (or new token) predicts Shape/Cam?
        #   - OR: Average Pool queries for Shape/Cam.
        
        self.decpose = nn.Linear(self.dim, 6) # Per joint rotation
        self.decshape = nn.Linear(self.dim, 10)
        self.deccam = nn.Linear(self.dim, 3)

        self._init_weights()

    def _init_weights(self):
        # [Fix] Initialize to Identity/Mean Pose to prevent Gradient Explosion
        # 1. Pose: Bias = Identity (6D), Weights = Tiny
        nn.init.normal_(self.decpose.weight, std=0.001)
        # Identity 6D is [1, 0, 0, 0, 1, 0]
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        nn.init.constant_(self.decpose.bias, 0)
        self.decpose.bias.data[:] = identity_6d

        # 2. Shape: Bias = 0, Weights = Tiny
        nn.init.normal_(self.decshape.weight, std=0.001)
        nn.init.constant_(self.decshape.bias, 0)

        # 3. Cam: Bias = 0, Weights = Tiny
        nn.init.normal_(self.deccam.weight, std=0.001)
        nn.init.constant_(self.deccam.bias, 0)
        self.deccam.bias.data[0] = 1.0  # [Fix] Rule 4: Camera Scale initialized to 1.0
        with torch.no_grad():
            self.deccam.bias[0] = 1.0 # Set scale s=1.0
        
    def forward(self, x_patches, coords):
        """
        x_patches: (B, 196, 1280)
        coords: (B, 24, 2) in [-1, 1]
        """
        B = x_patches.shape[0]
        
        # 1. Prepare Context
        context = self.context_proj(x_patches) # (B, 196, 1024)
        


        # 2. Prepare Queries
        # Dynamic Query = Static Identity + Positional Guidance
        queries = self.joint_queries.expand(B, -1, -1) # (B, 24, 1024)
        dynamic_queries = queries
        if not self.heatmap_only:
             pe = self.pe_2d(coords) # (B, 24, self.dim // 4)
             pos_embed = self.coord_encoder(pe) # (B, 24, 1024)
             dynamic_queries = dynamic_queries + self.guidance_scale * pos_embed
        
        # 3. Decode
        # (B, 24, 1024)
        out = self.decoder(dynamic_queries, context)
        
        # 4. Regress Parameters
        # Pose: (B, 24, 6)
        pred_pose_6d = self.decpose(out) 
        
        # Shape/Cam: Pool all queries? Or use efficient one?
        # Using Mean of all queries to predict global params
        global_feat = out.mean(dim=1) # (B, 1024)
        pred_shape = self.decshape(global_feat)
        pred_cam = self.deccam(global_feat)
        
        # 5. Format Output
        # Convert Rot6D -> RotMat
        pred_pose_rotmat = rot6d_to_rotmat(pred_pose_6d.reshape(-1, 6)).reshape(B, 24, 3, 3)
        
        # HMR2 convention:
        # global_orient: (B, 1, 3, 3) -> Joint 0 (Pelvis)
        # body_pose: (B, 23, 3, 3) -> Joints 1-23
        return {
            'global_orient': pred_pose_rotmat[:, 0:1],
            'body_pose': pred_pose_rotmat[:, 1:],
            'betas': pred_shape
        }, pred_cam, None
