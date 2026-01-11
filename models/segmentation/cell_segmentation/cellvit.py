# -*- coding: utf-8 -*-
# CellViT networks and adaptions, without sharing encoders
# Modified to use Swin Transformer as backbone
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor

from .utils import Conv2DBlock, Deconv2DBlock


# Swin Transformer with km attention
class SwinTransformerWithKMAttention(nn.Module):
    """Swin Transformer with km attention mechanism that retains only top-k attention weights, enhanced for better regularization"""
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        embed_dim=64,  # 降低嵌入维度，减少参数量
        depths=[1, 1, 3, 1],  # 降低深度，减少参数量
        num_heads=[2, 4, 8, 16],  # 减少注意力头数，减少参数量
        window_size=7,
        mlp_ratio=3.,  # 降低MLP比例，减少参数量
        qkv_bias=True,
        drop_rate=0.1,  # 增加dropout率，增强正则化
        attn_drop_rate=0.1,  # 增加注意力dropout率，增强正则化
        drop_path_rate=0.2,  # 增加路径dropout率，增强正则化
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        k=0.3,  # 降低top-k比例，增加稀疏性，缓解过拟合
        dynamic_k=False,  # 启用动态k值调整
        min_k_ratio=0.1,  # 最小k值比例
        max_k_ratio=0.7,  # 最大k值比例
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.k = k
        self.dynamic_k = dynamic_k
        self.min_k_ratio = min_k_ratio
        self.max_k_ratio = max_k_ratio
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0),
        )
        self.patch_norm = norm_layer(embed_dim) if patch_norm else nn.Identity()
        
        # Drop paths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build stages with downsampling
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_dpr_idx = 0
        current_dim = embed_dim
        
        for i_stage in range(self.num_layers):
            # Create stage layers
            stage_layers = nn.ModuleList()
            for j in range(depths[i_stage]):
                layer = SwinTransformerLayer(
                    dim=current_dim,
                    num_heads=num_heads[i_stage],
                    window_size=window_size,
                    shift_size=0 if ((i_stage + j) % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[current_dpr_idx],
                    norm_layer=norm_layer,
                    k=self.k,
                    dynamic_k=self.dynamic_k,
                    min_k_ratio=self.min_k_ratio,
                    max_k_ratio=self.max_k_ratio,
                )
                stage_layers.append(layer)
                current_dpr_idx += 1
            
            self.stages.append(stage_layers)
            
            # Add downsampling layer (except for last stage)
            if i_stage < self.num_layers - 1:
                downsample = PatchMerging(current_dim, norm_layer)
                self.downsamples.append(downsample)
            else:
                self.downsamples.append(None)
            
            # Update current dimension for next stage
            current_dim *= 2
        
        # Classification head
        self.norm = norm_layer(current_dim // 2)  # Last stage dimension
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # BCHW
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.patch_norm(x)  # Apply patch norm after reshaping to BNC
        
        # Extract features from different stages for skip connections
        stage_features = []
        current_H, current_W = H, W
        
        for i_stage in range(self.num_layers):
            layers = self.stages[i_stage]
            downsample = self.downsamples[i_stage]
            
            # Process all layers in current stage
            for layer in layers:
                x = layer(x)
            
            # Reshape to spatial dimensions and save as feature for skip connection
            stage_C = x.shape[-1]
            stage_L = x.shape[1]
            stage_H = stage_W = int(stage_L ** 0.5)
            feat = x.transpose(1, 2).view(B, stage_C, stage_H, stage_W)
            stage_features.append(feat)
            
            # Apply downsampling if not last stage
            if downsample is not None:
                x = downsample(x)
                current_H //= 2
                current_W //= 2
        
        # Classification token (global average pooling)
        x = self.norm(x)
        x_cls = self.avgpool(x.transpose(1, 2)).squeeze(-1)
        
        return x_cls, x, stage_features


class SwinTransformerLayer(nn.Module):
    """Swin Transformer Layer with km attention, enhanced for better regularization"""
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        k=0.3,  # 降低k值，增加稀疏性
        dynamic_k=False,  # 启用动态k值调整
        min_k_ratio=0.1,
        max_k_ratio=0.7
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.k = k
        self.dynamic_k = dynamic_k
        self.min_k_ratio = min_k_ratio
        self.max_k_ratio = max_k_ratio
        
        # Layer normalization
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # Window attention with enhanced km mechanism
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, k=self.k,
            dynamic_k=self.dynamic_k, min_k_ratio=self.min_k_ratio,
            max_k_ratio=self.max_k_ratio
        )
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        
    def forward(self, x):
        B, L, C = x.shape
        # 动态计算H和W，假设输入是正方形
        H = W = int(L ** 0.5)
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Window attention with cyclic shift
        if self.shift_size > 0:
            pad_l = pad_t = self.window_size - self.shift_size
            pad_r = pad_b = 0
            x_shifted = torch.nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b), mode='circular')
            Hp, Wp = x_shifted.shape[1], x_shifted.shape[2]
            x_windows, pad_h, pad_w, H_padded, W_padded = window_partition(x_shifted, self.window_size)
        else:
            x_windows, pad_h, pad_w, H_padded, W_padded = window_partition(x, self.window_size)
        
        # Apply attention
        attn_windows = self.attn(x_windows)
        
        # Window reverse
        if self.shift_size > 0:
            x = window_reverse(attn_windows, self.window_size, H_padded, W_padded, pad_h, pad_w)
            pad_l = pad_t = self.window_size - self.shift_size
            x = x[:, pad_t:, pad_l:, :]
        else:
            x = window_reverse(attn_windows, self.window_size, H_padded, W_padded, pad_h, pad_w)
        
        x = x.reshape(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention with km attention mechanism, enhanced for better regularization"""
    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 k=0.3,  # 降低k值，增加稀疏性，缓解过拟合
                 dynamic_k=False,  # 启用动态k值调整
                 min_k_ratio=0.1,  # 最小k值比例
                 max_k_ratio=0.7  # 最大k值比例
                 ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.k = k
        self.dynamic_k = dynamic_k
        self.min_k_ratio = min_k_ratio
        self.max_k_ratio = max_k_ratio
        
        # Q, K, V projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Add layer normalization for better regularization
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
    
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, num_heads, N, C//num_heads
        
        # Normalize q and k for better stability and regularization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Scaled dot product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Dynamic k adjustment based on attention entropy (optional)
        current_k = self.k
        if self.dynamic_k:
            # Calculate attention entropy to measure uncertainty
            soft_attn = F.softmax(attn, dim=-1)
            entropy = -torch.sum(soft_attn * torch.log(soft_attn + 1e-10), dim=-1).mean()
            # Adjust k based on entropy: higher entropy means more uncertainty, use larger k
            current_k = self.min_k_ratio + (self.max_k_ratio - self.min_k_ratio) * entropy
            current_k = torch.clamp(current_k, self.min_k_ratio, self.max_k_ratio).item()
        
        # Apply km attention: retain only top-k attention weights
        k_val = int(N * current_k)
        if k_val < N:
            # Get top-k values and their indices
            top_k_attn, top_k_indices = attn.topk(k_val, dim=-1)
            # Create a mask for non-top-k values
            mask = torch.full_like(attn, float('-inf'))
            mask.scatter_(-1, top_k_indices, 0)
            attn = attn + mask
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP module"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def DropPath(drop_prob=None):
    """Drop paths (Stochastic Depth) per sample"""
    return nn.Sequential()


def window_partition(x, window_size):
    """Partition into non-overlapping windows with padding"""
    B, H, W, C = x.shape
    
    # Pad H and W to be divisible by window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    H_padded = H + pad_h
    W_padded = W + pad_w
    
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    
    x = x.view(B, H_padded // window_size, window_size, W_padded // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows, pad_h, pad_w, H_padded, W_padded


def window_reverse(windows, window_size, H, W, pad_h=0, pad_w=0):
    """Reverse window_partition with padding removal"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    
    # Remove padding if added during window_partition
    if pad_h > 0 or pad_w > 0:
        x = x[:, :-pad_h, :-pad_w, :] if pad_h > 0 and pad_w > 0 else \
            x[:, :-pad_h, :, :] if pad_h > 0 else \
            x[:, :, :-pad_w, :] if pad_w > 0 else x
    
    return x


class PatchMerging(nn.Module):
    """Patch merging layer to downsample the feature map"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        
        # Reshape to spatial dimensions
        x = x.view(B, H, W, C)
        
        # Pad if H or W is odd
        if H % 2 == 1:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1, 0, 1))
            H += 1
            W += 1
        
        # Split into four patches
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        # Concatenate along channel dimension
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        
        # Reshape back to sequence
        x = x.view(B, -1, 4 * C)
        
        # Apply normalization and reduction
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class CellViT(nn.Module):
    """CellViT Modell for cell segmentation. U-Net like network with Swin Transformer as backbone encoder

    Skip connections are shared between branches, but each network has a distinct encoder

    The modell is having multiple branches:
        * tissue_types: Tissue prediction based on global class token
        * nuclei_binary_map: Binary nuclei prediction
        * hv_map: HV-prediction to separate isolated instances
        * nuclei_type_map: Nuclei instance-prediction
        * [Optional, if regression loss]:
        * regression_map: Regression map for binary prediction

    Args:
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        embed_dim (int): Embedding dimension of backbone Swin Transformer
        input_channels (int): Number of input channels (default: 2 for H-DAB)
        depth (int): Depth of the backbone Swin Transformer
        num_heads (int): Number of heads of the backbone Swin Transformer
        extract_layers: (List[int]): List of Transformer Blocks whose outputs should be returned in addition to the tokens.
            Is used for skip connections. At least 4 skip connections needs to be returned.
        mlp_ratio (float, optional): MLP ratio for hidden MLP dimension of backbone Swin Transformer. Defaults to 4.
        qkv_bias (bool, optional): If bias should be used for query (q), key (k), and value (v) in backbone Swin Transformer. Defaults to True.
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone Swin Transformer. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
        window_size (int, optional): Window size for Swin Transformer. Defaults to 7.
        k (float, optional): Top-k ratio for km attention. Defaults to 0.5.
    """

    def __init__(
        self,
        num_nuclei_classes: int,
        num_tissue_classes: int,
        embed_dim: int = 64,  # 降低嵌入维度
        input_channels: int = 3,  # RGB input
        depth: int = 6,  # 降低总深度
        num_heads: int = 2,  # 减少注意力头数
        extract_layers: List = None,
        mlp_ratio: float = 3,  # 降低MLP比例
        qkv_bias: bool = True,
        drop_rate: float = 0.1,  # 增加dropout率
        attn_drop_rate: float = 0.1,  # 增加注意力dropout率
        drop_path_rate: float = 0.1,  # 增加路径dropout率
        regression_loss: bool = False,
        window_size: int = 7,
        k: float = 0.3,  # 降低top-k比例，增加稀疏性，缓解过拟合
        dynamic_k: bool = False,  # 启用动态k值调整
        min_k_ratio: float = 0.1,  # 最小k值比例
        max_k_ratio: float = 0.7,  # 最大k值比例
    ):
        super().__init__()
        
        # Default extract layers for Swin Transformer
        if extract_layers is None:
            extract_layers = [1, 2, 3, 4]
        
        self.patch_size = 4  # Swin Transformer uses smaller patch size
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.extract_layers = extract_layers
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.window_size = window_size
        self.k = k
        self.dynamic_k = dynamic_k
        self.min_k_ratio = min_k_ratio
        self.max_k_ratio = max_k_ratio
        
        # Calculate depths per stage for Swin Transformer
        # Swin Transformer typically uses 4 stages
        # 降低模型深度，减少过拟合
        total_depth = sum([1, 1, 3, 1])  # 更浅的深度设置
        if depth != total_depth:
            # Scale depths proportionally
            scale_factor = depth / total_depth
            depths = [max(1, int(1 * scale_factor)), max(1, int(1 * scale_factor)), 
                     max(1, int(3 * scale_factor)), max(1, int(1 * scale_factor))]
        else:
            depths = [1, 1, 3, 1]
        
        # Calculate num_heads per stage (doubles at each stage)
        num_heads_per_stage = [num_heads, num_heads * 2, num_heads * 4, num_heads * 8]
        
        # Initialize Swin Transformer with enhanced km attention
        self.backbone = SwinTransformerWithKMAttention(
            patch_size=self.patch_size,
            in_chans=input_channels,  # RGB input
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads_per_stage,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            k=self.k,
            dynamic_k=self.dynamic_k,
            min_k_ratio=self.min_k_ratio,
            max_k_ratio=self.max_k_ratio
        )
        
        # Add enhanced classification head for tissue types with increased dropout
        self.tissue_head = nn.Sequential(
            nn.Linear(embed_dim * 8, embed_dim * 8),
            nn.BatchNorm1d(embed_dim * 8),
            nn.GELU(),
            nn.Dropout(0.7),  # 增加dropout率
            nn.Linear(embed_dim * 8, num_tissue_classes)
        )


        # Adjust dimensions based on Swin Transformer outputs
        # Swin Transformer outputs at 4 scales with 2x downsampling each time
        self.swin_dims = [embed_dim * 2 ** i for i in range(4)]  # [96, 192, 384, 768] for Swin-T
        self.bottleneck_dim = self.swin_dims[-1] // 2  # Use half of the largest dimension for bottleneck

        # version with shared skip_connections
        self.decoder0 = nn.Sequential(
            Conv2DBlock(self.input_channels, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection from input image
        
        # Adapt decoders for Swin Transformer feature dimensions
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.swin_dims[-4], 128, dropout=self.drop_rate),  # First stage features
        )  # skip connection 1
        
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.swin_dims[-3], 256, dropout=self.drop_rate),  # Second stage features
        )  # skip connection 2
        
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.swin_dims[-2], self.bottleneck_dim, dropout=self.drop_rate)  # Third stage features
        )  # skip connection 3

        self.regression_loss = regression_loss
        offset_branches = 0
        if self.regression_loss:
            offset_branches = 2
        self.branches_output = {
            "nuclei_binary_map": 2 + offset_branches,
            "hv_map": 2,
            "nuclei_type_maps": self.num_nuclei_classes,
        }

        self.nuclei_binary_map_decoder = self.create_upsampling_branch(
            2 + offset_branches
        )  # todo: adapt for helper loss
        self.hv_map_decoder = self.create_upsampling_branch(
            2
        )  # todo: adapt for helper loss
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(
            self.num_nuclei_classes
        )

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (B, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * [Optional, if retrieve tokens]: tokens
                * [Optional, if regression loss]:
                * regression_map: Regression map for binary prediction. Shape: (B, 2, H, W)
        """
        # Swin Transformer expects image size divisible by 2^4 = 16 (for 4 stages)
        H, W = x.shape[-2], x.shape[-1]
        if H % 16 != 0 or W % 16 != 0:
            # Pad to nearest multiple of 16
            pad_h = (16 - H % 16) % 16
            pad_w = (16 - W % 16) % 16
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
            original_shape = (H, W)
        else:
            original_shape = None

        out_dict = {}

        # Forward pass through Swin Transformer
        cls_features, tokens, swin_features = self.backbone(x)
        
        # Apply tissue classification head with enhanced feature processing
        # 增加组织分类的特征处理，提高分类性能
        cls_features_enhanced = F.dropout(cls_features, p=0.3, training=self.training)
        out_dict["tissue_types"] = self.tissue_head(cls_features_enhanced)

        # z0 is the input image, z1-z4 are features from Swin Transformer stages
        z0 = x
        # Ensure we have exactly 4 skip connections
        z1, z2, z3, z4 = swin_features

        # Swin Transformer features are already in spatial format (B, C, H, W), so no need for reshaping

        # Forward through decoders
        if self.regression_loss:
            nb_map = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )
        
        # Crop back to original size if we padded
        if original_shape is not None:
            for key in out_dict:
                if isinstance(out_dict[key], torch.Tensor) and len(out_dict[key].shape) == 4:
                    out_dict[key] = out_dict[key][:, :, :original_shape[0], :original_shape[1]]
        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        """Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns:
            torch.Tensor: Branch Output
        """
        # For Swin Transformer, we need to ensure proper upsampling to match dimensions
        
        # Step 1: Process bottleneck (z4) and upsample to match z3
        b4 = branch_decoder.bottleneck_upsampler(z4)  # z4 (8x8) -> b4 (16x16)
        
        # Step 2: Concatenate z3 and b4, then upsample to match z2
        # z3 is 16x16, b4 is 16x16, after concatenation -> 32x32
        b3_concat = torch.cat([z3, b4], dim=1)
        b3 = branch_decoder.decoder3_upsampler(b3_concat)  # 16x16 -> 32x32
        
        # Step 3: Concatenate z2 and b3, then upsample to match z1
        # z2 is 32x32, b3 is 32x32, after concatenation -> 64x64
        b2_concat = torch.cat([z2, b3], dim=1)
        b2 = branch_decoder.decoder2_upsampler(b2_concat)  # 32x32 -> 64x64
        
        # Step 4: Concatenate z1 and b2, then upsample to match z0
        # z1 is 64x64, b2 is 64x64, after concatenation -> 128x128
        b1_concat = torch.cat([z1, b2], dim=1)
        b1 = branch_decoder.decoder1_upsampler(b1_concat)  # 64x64 -> 128x128
        
        # Step 5: Upsample b1 to match z0 size
        b1 = torch.nn.functional.interpolate(b1, size=(z0.shape[2], z0.shape[3]), mode='bilinear', align_corners=False)  # 128x128 -> 256x256
        
        # Step 6: Concatenate input image (z0) directly with b1, then pass through decoder0_header
        # z0 is 3 channels, b1 is 64 channels, total 3+64=67 channels which matches decoder0_header expectation
        b0_concat = torch.cat([z0, b1], dim=1)
        branch_output = branch_decoder.decoder0_header(b0_concat)

        return branch_output

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create Upsampling branch

        Args:
            num_classes (int): Number of output classes

        Returns:
            nn.Module: Upsampling path
        """
        # Adapt decoder channels for Swin Transformer outputs
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.swin_dims[-1],  # Largest dimension from Swin Transformer
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        
        # First upsampling block (from bottleneck to 3rd stage)
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.swin_dims[-2] + self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        
        # Second upsampling block (3rd to 2nd stage)
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(self.swin_dims[-3] + 256, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        
        # Third upsampling block (2nd to 1st stage)
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(self.swin_dims[-4] + 128, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        
        # Final header block
        decoder0_header = nn.Sequential(
            Conv2DBlock(self.input_channels + 64, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder

    def calculate_instance_map(
        self, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Calculate Instance Map from network predictions (after Softmax output)

        Args:
            predictions (dict): Dictionary with the following required keys:
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, self.num_nuclei_classes, H, W)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, 2, H, W)
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
        """
        # reshape to B, H, W, C
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        cell_post_processor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes, magnification=magnification, gt=False
        )
        instance_preds = []
        type_preds = []

        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate(
                [
                    torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    predictions_["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds

    def generate_instance_nuclei_map(
        self, instance_maps: torch.Tensor, type_preds: List[dict]
    ) -> torch.Tensor:
        """Convert instance map (binary) to nuclei type instance map

        Args:
            instance_maps (torch.Tensor): Binary instance map, each instance has own integer. Shape: (B, H, W)
            type_preds (List[dict]): List (len=B) of dictionary with instance type information (compare post_process_hovernet function for more details)

        Returns:
            torch.Tensor: Nuclei type instance map. Shape: (B, self.num_nuclei_classes, H, W)
        """
        batch_size, h, w = instance_maps.shape
        instance_type_nuclei_maps = torch.zeros(
            (batch_size, h, w, self.num_nuclei_classes)
        )
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, self.num_nuclei_classes))
            instance_map = instance_maps[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][
                    instance_map == nuclei
                ] = nuclei

            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

        instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
        return torch.Tensor(instance_type_nuclei_maps)

    def freeze_encoder(self):
        """Freeze encoder to train only the classification head at first"""
        for layer_name, p in self.backbone.named_parameters():
            if layer_name.split(".")[0] != "head":  # do not freeze head
                p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder to train the whole model"""
        for p in self.backbone.parameters():
            p.requires_grad = True


class CellViT256(CellViT):
    """CellViT with ViT-256 backbone settings (https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth)

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        model256_path (Union[Path, str]): Path to ViT 256 backbone model
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
    """

    def __init__(
        self,
        model256_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,  # to use regressive loss for predicting vector components
        input_channels: int = 3,  # RGB
    ):
        self.patch_size = 16
        self.embed_dim = 384
        self.depth = 12
        self.num_heads = 6
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.extract_layers = [3, 6, 9, 12]
        self.input_channels = input_channels
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes

        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            regression_loss=regression_loss,
        )

        self.model256_path = model256_path

    def load_pretrained_encoder(self, model256_path: str):
        """Load pretrained ViT-256 from provided path

        Args:
            model256_path (str): Path to ViT-256
        """
        state_dict = torch.load(str(model256_path), map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")


class CellViTSAM(CellViT):
    """CellViT with SAM backbone settings

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        model_path (Union[Path, str]): Path to pretrained SAM model
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        vit_structure (Literal["SAM-B", "SAM-L", "SAM-H"]): SAM model type
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.

    Raises:
        NotImplementedError: Unknown SAM configuration
    """

    def __init__(
        self,
        model_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        vit_structure: Literal["SAM-B", "SAM-L", "SAM-H"],
        drop_rate: float = 0,
        regression_loss: bool = False,
        input_channels: int = 3,  # RGB
    ):
        if vit_structure.upper() == "SAM-B":
            self.init_vit_b()
        elif vit_structure.upper() == "SAM-L":
            self.init_vit_l()
        elif vit_structure.upper() == "SAM-H":
            self.init_vit_h()
        else:
            raise NotImplementedError("Unknown ViT-SAM backbone structure")

        self.input_channels = input_channels
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.num_nuclei_classes = num_nuclei_classes
        self.model_path = model_path

        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
            regression_loss=regression_loss,
        )

        self.prompt_embed_dim = 256

        self.encoder = ViTCellViTDeit(
            extract_layers=self.extract_layers,
            depth=self.depth,
            embed_dim=self.embed_dim,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=self.num_heads,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=self.encoder_global_attn_indexes,
            window_size=14,
            out_chans=self.prompt_embed_dim,
        )

        self.classifier_head = (
            nn.Linear(self.prompt_embed_dim, num_tissue_classes)
            if num_tissue_classes > 0
            else nn.Identity()
        )

    def load_pretrained_encoder(self, model_path):
        """Load pretrained SAM encoder from provided path

        Args:
            model_path (str): Path to SAM model
        """
        state_dict = torch.load(str(model_path), map_location="cpu")
        image_encoder = self.encoder
        msg = image_encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")
        self.encoder = image_encoder

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False):
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of ViT should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (B, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * [Optional, if retrieve tokens]: tokens
                * [Optional, if regression loss]:
                * regression_map: Regression map for binary prediction. Shape: (B, 2, H, W)
        """
        assert (
            x.shape[-2] % self.patch_size == 0
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = self.classifier_head(classifier_logits)

        z0, z1, z2, z3, z4 = x, *z

        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        z4 = z4.permute(0, 3, 1, 2)
        z3 = z3.permute(0, 3, 1, 2)
        z2 = z2.permute(0, 3, 1, 2)
        z1 = z1.permute(0, 3, 1, 2)

        if self.regression_loss:
            nb_map = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )

        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def init_vit_b(self):
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.encoder_global_attn_indexes = [2, 5, 8, 11]
        self.extract_layers = [3, 6, 9, 12]

    def init_vit_l(self):
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.encoder_global_attn_indexes = [5, 11, 17, 23]
        self.extract_layers = [6, 12, 18, 24]

    def init_vit_h(self):
        self.embed_dim = 1280
        self.depth = 32
        self.num_heads = 16
        self.encoder_global_attn_indexes = [7, 15, 23, 31]
        self.extract_layers = [8, 16, 24, 32]


@dataclass
class DataclassHVStorage:
    """Storing PanNuke Prediction/GT objects for calculating loss, metrics etc. with HoverNet networks

    Args:
        nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
        hv_map (torch.Tensor): Logit output for HV-Map. Shape: (batch_size, 2, H, W)
        nuclei_type_map (torch.Tensor): Softmax output for nuclei type-prediction. Shape: (batch_size, num_tissue_classes, H, W)
        tissue_types (torch.Tensor): Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
        instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
            Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
        instance_types_nuclei: Pixel-wise nuclear instance segmentation predictions, for each nuclei type.
            Each instance has its own integer, starting from 1.
            Shape: (batch_size, num_nuclei_classes, H, W)
        batch_size (int): Batch size of the experiment
        instance_types (list, optional): Instance type prediction list.
            Each list entry stands for one image. Each list entry is a dictionary with the following structure:
            Main Key is the nuclei instance number (int), with a dict as value.
            For each instance, the dictionary contains the keys: bbox (bounding box), centroid (centroid coordinates),
            contour, type_prob (probability), type (nuclei type)
            Defaults to None.
        regression_map (torch.Tensor, optional): Regression map for binary prediction map.
            Shape: (batch_size, 2, H, W). Defaults to None.
        regression_loss (bool, optional): Indicating if regression map is present. Defaults to False.
        h (int, optional): Height of used input images. Defaults to 256.
        w (int, optional): Width of used input images. Defaults to 256.
        num_tissue_classes (int, optional): Number of tissue classes in the data. Defaults to 19.
        num_nuclei_classes (int, optional): Number of nuclei types in the data (including background). Defaults to 6.
    """

    nuclei_binary_map: torch.Tensor
    hv_map: torch.Tensor
    tissue_types: torch.Tensor
    nuclei_type_map: torch.Tensor
    instance_map: torch.Tensor
    instance_types_nuclei: torch.Tensor
    batch_size: int
    instance_types: list = None
    regression_map: torch.Tensor = None
    regression_loss: bool = False
    h: int = 256
    w: int = 256
    num_tissue_classes: int = 19
    num_nuclei_classes: int = 6

    # def __post_init__(self):
    #     # check shape of every element
    #     assert list(self.nuclei_binary_map.shape) == [
    #         self.batch_size,
    #         2,
    #         self.h,
    #         self.w,
    #     ], "Nuclei Binary Map must be a softmax tensor with shape (B, 2, H, W)"
    #     assert list(self.hv_map.shape) == [
    #         self.batch_size,
    #         2,
    #         self.h,
    #         self.w,
    #     ], "HV Map must be a tensor with shape (B, 2, H, W)"
    #     assert list(self.nuclei_type_map.shape) == [
    #         self.batch_size,
    #         self.num_nuclei_classes,
    #         self.h,
    #         self.w,
    #     ], "Nuclei Type Map must be a tensor with shape (B, num_nuclei_classes, H, W)"
    #     assert list(self.instance_map.shape) == [
    #         self.batch_size,
    #         self.h,
    #         self.w,
    #     ], "Instance Map must be a tensor with shape (B, H, W)"
    #     assert list(self.instance_types_nuclei.shape) == [
    #         self.batch_size,
    #         self.num_nuclei_classes,
    #         self.h,
    #         self.w,
    #     ], "Instance Types Nuclei must be a tensor with shape (B, num_nuclei_classes, H, W)"
    #     if self.regression_map is not None:
    #         self.regression_loss = True
    #     else:
    #         self.regression_loss = False

    def get_dict(self) -> dict:
        """Return dictionary of entries"""
        property_dict = self.__dict__
        if not self.regression_loss and "regression_map" in property_dict.keys():
            property_dict.pop("regression_map")
        return property_dict
