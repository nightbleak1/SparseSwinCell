from einops import rearrange
from models.encoders.VIT.vits_histo import VisionTransformer
from models.encoders.VIT.SAM.image_encoder import ImageEncoderViT
from models.encoders.VIT.sparse_vit import SparseVisionTransformer

import torch
import torch.nn as nn
from typing import Callable, Tuple, Type, List

class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)

class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)

class SparseViTCellViT(SparseVisionTransformer):
    """使用稀疏自注意力的CellViT编码器"""
    def __init__(
        self,
        extract_layers: List[int],
        img_size: List[int] = [224],
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        qkv_bias: bool = False,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        norm_layer: Callable = nn.LayerNorm,
        attention_type: str = "mixed",  # "local", "content", "mixed"
        window_size: int = 16,
        top_k_ratio: float = 0.2,
        local_head_ratio: float = 0.6,
        **kwargs
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            attention_type=attention_type,
            window_size=window_size,
            top_k_ratio=top_k_ratio,
            local_head_ratio=local_head_ratio,
            **kwargs
        )
        self.extract_layers = extract_layers
        
        # 输入嵌入层优化：添加局部稀疏注意力预处理
        self.local_preprocess = nn.Sequential(
            Conv2DBlock(embed_dim, embed_dim, kernel_size=1, dropout=drop_rate),
            nn.LayerNorm(embed_dim)
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播，返回中间层输出用于跳跃连接"""
        extracted_layers = []
        x = self.prepare_tokens(x)
        
        # 输入嵌入层优化：局部稀疏注意力预处理
        cls_token = x[:, 0:1, :]
        patch_tokens = x[:, 1:, :]
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        
        # 重塑为空间维度
        patch_tokens_spatial = patch_tokens.view(B, H, W, C)
        # 应用局部预处理
        patch_tokens = self.local_preprocess(patch_tokens_spatial.permute(0, 3, 1, 2))
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(B, N, C)
        
        # 重新组合分类标记和补丁标记
        x = torch.cat([cls_token, patch_tokens], dim=1)
        
        # 通过Transformer块
        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)

        x = self.norm(x)
        output = self.head(x[:, 0])

        return output, x[:, 0], extracted_layers

class MultiScaleSparseAttention(nn.Module):
    """多尺度特征融合前的稀疏注意力增强模块"""
    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.window_size = window_size
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 应用局部稀疏注意力
        attn_map = self.attention(x)
        x = x * attn_map
        x = self.dropout(x)
        
        return x