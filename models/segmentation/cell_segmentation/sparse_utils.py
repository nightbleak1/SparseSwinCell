from einops import rearrange
from models.encoders.VIT.vits_histo import VisionTransformer
from models.encoders.VIT.SAM.image_encoder import ImageEncoderViT
from models.encoders.VIT.sparse_vit import SparseVisionTransformer

import torch
import torch.nn as nn
from typing import Callable, Tuple, Type, List


class ChannelAttention(nn.Module):
    """通道注意力模块：增强特征通道的注意力机制"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class BoundaryAwareConv(nn.Module):
    """增强的边界感知卷积模块：多尺度边界特征提取
    使用互补膨胀率避免网格效应
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # 标准卷积支路
        self.norm_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn_norm = nn.BatchNorm2d(out_channels)
        
        # 多尺度边界检测：使用互补膨胀率的卷积，避免网格效应
        # 膨胀率1：小边界（padding=dilation保持特征图尺寸不变）
        self.boundary_conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                          padding=1 * padding, dilation=1)
        self.bn_boundary_1 = nn.BatchNorm2d(out_channels)
        
        # 膨胀率3：中等边界
        self.boundary_conv_3 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                          padding=3 * padding, dilation=3)
        self.bn_boundary_3 = nn.BatchNorm2d(out_channels)
        
        # 膨胀率5：大范围边界
        self.boundary_conv_5 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                          padding=5 * padding, dilation=5)
        self.bn_boundary_5 = nn.BatchNorm2d(out_channels)
        
        # 边界特征融合
        self.boundary_fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        
        # 通道注意力机制
        self.ca = ChannelAttention(out_channels)
        
        # 空间注意力机制：检测边界区域
        self.sa_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 残差连接
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.residual(x)
        
        # 标准特征
        x_norm = self.relu(self.bn_norm(self.norm_conv(x)))
        
        # 多尺度边界特征（使用互补膨胀率）
        x_boundary_1 = self.relu(self.bn_boundary_1(self.boundary_conv_1(x)))
        x_boundary_3 = self.relu(self.bn_boundary_3(self.boundary_conv_3(x)))
        x_boundary_5 = self.relu(self.bn_boundary_5(self.boundary_conv_5(x)))
        
        # 融合多尺度边界特征
        x_boundary = self.boundary_fusion(torch.cat([x_boundary_1, x_boundary_3, x_boundary_5], dim=1))
        
        # 通道注意力加权
        x_boundary = x_boundary * self.ca(x_boundary)
        
        # 空间注意力：检测边界区域并增强
        sa_weight = self.sa_conv(x_boundary)
        x_boundary = x_boundary * sa_weight
        
        # 融合标准特征和边界特征
        x_fused = self.fusion(torch.cat([x_norm, x_boundary], dim=1))
        
        # 残差连接
        return x_fused + residual


class AttentionFusion(nn.Module):
    """注意力融合模块：融合不同尺度的特征"""
    def __init__(self, channels: int):
        super().__init__()
        # 通道注意力
        self.channel_attention = ChannelAttention(channels)
        
        # 空间注意力
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 通道注意力加权
        ca_weight = self.channel_attention(x)
        x_ca = x * ca_weight
        
        # 空间注意力
        sa_weight = self.sigmoid(self.conv(x))
        x_sa = x * sa_weight
        
        # 融合
        return x_ca + x_sa


class FeatureFusion(nn.Module):
    """改进的特征融合模块：融合跳跃连接特征"""
    def __init__(self, channels: int):
        super().__init__()
        # 特征对齐卷积
        self.align_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 边界感知卷积
        self.boundary_conv = BoundaryAwareConv(channels, channels)
        
        # 注意力融合
        self.attention_fusion = AttentionFusion(channels)
        
        # 最终融合
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        # 对齐特征
        x_aligned = self.align_conv(x)
        
        # 边界感知处理
        x_boundary = self.boundary_conv(x_aligned)
        
        # 注意力融合
        x_fused = self.attention_fusion(x_boundary)
        
        # 最终融合
        return self.fusion(x_fused)

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
        sparsity_level: float = 0.2,  # 注意力稀疏性控制（0.0-1.0）
        use_gap: bool = True,  # 使用全局平均池化
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
            sparsity_level=sparsity_level,
            **kwargs
        )
        self.extract_layers = extract_layers
        self.use_gap = use_gap
        
        # 输入嵌入层优化：添加局部稀疏注意力预处理
        self.local_preprocess = nn.Sequential(
            Conv2DBlock(embed_dim, embed_dim, kernel_size=1, dropout=drop_rate),
            nn.BatchNorm2d(embed_dim)
        )
        
        # 增强的组织类型分类头：包含BatchNorm、GELU激活和dropout层
        if num_classes > 0:
            # 输入维度为embed_dim（仅CLS）或2*embed_dim（CLS+GAP融合）
            classifier_input_dim = embed_dim if not use_gap else embed_dim * 2
            self.tissue_classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, classifier_input_dim // 2),
                nn.BatchNorm1d(classifier_input_dim // 2),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(classifier_input_dim // 2, num_classes)
            )
        else:
            self.tissue_classifier = None

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
        
        # 组织类型预测：使用CLS token + 全局平均池化
        if self.tissue_classifier is not None:
            if self.use_gap:
                # 全局平均池化：捕获全局上下文信息
                patch_features = x[:, 1:, :]  # 移除CLS token
                gap_features = patch_features.mean(dim=1)  # [B, C]
                cls_features = x[:, 0, :]  # [B, C]
                # 融合CLS和GAP特征
                fused_features = torch.cat([cls_features, gap_features], dim=-1)
                output = self.tissue_classifier(fused_features)
            else:
                output = self.tissue_classifier(x[:, 0])
        else:
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
        
        # 增强的注意力机制：添加边界感知的空间注意力
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 添加边界增强模块：使用更精细的卷积捕捉边界
        self.boundary_enhance = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 特征融合：注意力 + 边界增强
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 应用局部稀疏注意力
        attn_map = self.attention(x)
        x_attn = x * attn_map
        
        # 边界增强
        boundary_map = self.boundary_enhance(x)
        x_boundary = x * boundary_map
        
        # 融合注意力特征和边界特征
        x_fused = torch.cat([x_attn, x_boundary], dim=1)
        x = self.fusion(x_fused)
        
        x = self.dropout(x)
        
        return x