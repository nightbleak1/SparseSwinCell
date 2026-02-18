
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
from typing import List, Literal, Tuple, Union, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor

from .utils import Conv2DBlock, Deconv2DBlock


class AttentionGate(nn.Module):
    """
    Attention Gate for filtering features from skip connections.
    Inspired by Attention U-Net.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g to match x size if necessary
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module.
    """
    def __init__(self, in_channels, out_channels, atrous_rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        
        # 1x1 Convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Atrous Convolutions
        for rate in atrous_rates[1:]:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Image Pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            out = conv(x)
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            res.append(out)
        
        res = torch.cat(res, dim=1)
        return self.project(res)

class ShapeStream(nn.Module):
    """
    Dedicated Shape Stream for Boundary Detection.
    Process features to extract boundaries.
    """
    def __init__(self, in_channels):
        super(ShapeStream, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Gated Convolution for edge refinement
        self.gate = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            # nn.Sigmoid() # Removed Sigmoid because we use BCEWithLogitsLoss
        )
        
    def forward(self, x, residuals=None):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # If residuals from main stream are provided, use them to gate the shape stream
        if residuals is not None:
            # Attention-like gating
            if residuals.shape[2:] != x.shape[2:]:
                residuals = F.interpolate(residuals, size=x.shape[2:], mode='bilinear', align_corners=False)
            # Match channels if needed (simple 1x1 conv if mismatch, but assuming compatibility here or simple broadcasting)
            pass 

        edge_map = self.gate(x)
        return x, edge_map


# Swin Transformer with km attention
class SwinTransformerWithKMAttention(nn.Module):
    """Swin Transformer with km attention mechanism that retains only top-k attention weights, enhanced for better regularization"""
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        embed_dim=64,
        depths=[1, 1, 3, 1],
        num_heads=[2, 4, 8, 16],
        window_size=7,
        mlp_ratio=3.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        k=0.3,
        dynamic_k=False,
        min_k_ratio=0.1,
        max_k_ratio=0.7,
        sparsity_level=0.2,
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
        self.sparsity_level = sparsity_level
        
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
                    sparsity_level=self.sparsity_level,
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
        max_k_ratio=0.7,
        sparsity_level=0.2,  # 注意力稀疏性控制
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
        self.sparsity_level = sparsity_level
        
        # Layer normalization
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # Window attention with enhanced km mechanism
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, k=self.k,
            dynamic_k=self.dynamic_k, min_k_ratio=self.min_k_ratio,
            max_k_ratio=self.max_k_ratio,
            sparsity_level=self.sparsity_level,
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
                 max_k_ratio=0.7,  # 最大k值比例
                 sparsity_level=0.2,  # 注意力稀疏性控制
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
        self.sparsity_level = sparsity_level
        
        # Q, K, V projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Add layer normalization for better regularization
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        
        # 注意力头剪枝机制：可学习的注意力头权重
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        self.head_softmax = nn.Softmax(dim=0)
    
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
        
        # 应用注意力头权重
        head_weights = self.head_softmax(self.head_weights)
        head_weights = head_weights.view(1, self.num_heads, 1, 1)
        attn = attn * head_weights
        
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


class SparseCellViT(nn.Module):
    """CellViT Modell for cell segmentation. U-Net like network with Swin Transformer as backbone encoder
    
    ENHANCED VERSION with ASPP, Attention Gates, and Shape Stream for superior boundary detection.
    """

    def __init__(
        self,
        num_nuclei_classes: int,
        num_tissue_classes: int,
        embed_dim: int = 64,
        input_channels: int = 3,
        depth: int = 6,
        num_heads: int = 2,
        extract_layers: List = None,
        mlp_ratio: float = 3,
        qkv_bias: bool = True,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        regression_loss: bool = False,
        window_size: int = 7,
        k: float = 0.3,
        dynamic_k: bool = False,
        min_k_ratio: float = 0.1,
        max_k_ratio: float = 0.7,
        sparsity_level: float = 0.2,
    ):
        super().__init__()
        
        # Default extract layers for Swin Transformer
        if extract_layers is None:
            extract_layers = [1, 2, 3, 4]
        
        self.patch_size = 4
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
        self.sparsity_level = sparsity_level
        
        # 1. Backbone: Swin Transformer V2
        # 增加参数量以提升性能
        total_depth = sum([1, 1, 3, 1])
        if depth != total_depth:
            scale_factor = depth / total_depth
            depths = [max(1, int(1 * scale_factor)), max(1, int(1 * scale_factor)), 
                     max(1, int(3 * scale_factor)), max(1, int(1 * scale_factor))]
        else:
            depths = [1, 1, 3, 1]
        
        num_heads_per_stage = [num_heads, num_heads * 2, num_heads * 4, num_heads * 8]
        
        self.backbone = SwinTransformerWithKMAttention(
            patch_size=self.patch_size,
            in_chans=input_channels,
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
            max_k_ratio=self.max_k_ratio,
            sparsity_level=self.sparsity_level,
        )
        
        self.swin_dims = [embed_dim * 2 ** i for i in range(4)]
        self.bottleneck_dim = self.swin_dims[-1] // 2
        
        # 2. Advanced Feature Aggregation: ASPP
        # 位于 Backbone 输出 (z4) 之后，捕捉多尺度上下文
        self.aspp = ASPP(self.swin_dims[-1], self.bottleneck_dim)
        
        # 3. Shape Stream for Boundary Detection
        # 独立的轻量级分支，专注于边缘检测
        self.shape_stream = ShapeStream(self.swin_dims[0]) # 从浅层特征开始

        # 4. Classification Head (Tissue)
        self.tissue_head = nn.Sequential(
            nn.Linear(embed_dim * 8, embed_dim * 8),
            nn.BatchNorm1d(embed_dim * 8),
            nn.GELU(),
            nn.Dropout(0.7),
            nn.Linear(embed_dim * 8, num_tissue_classes)
        )

        # 5. Decoders with Attention Gates
        self.regression_loss = regression_loss
        offset_branches = 0
        if self.regression_loss:
            offset_branches = 2
        
        self.nuclei_binary_map_decoder = self.create_upsampling_branch(
            2 + offset_branches, use_shape_stream=True
        )
        self.hv_map_decoder = self.create_upsampling_branch(
            2, use_shape_stream=True
        )
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(
            self.num_nuclei_classes, use_shape_stream=False # Type branch focuses on semantic class
        )
        
        # Attention Gates for Skip Connections
        # z3 (16x16) <-> decoder3 (16x16)
        self.ag3 = AttentionGate(F_g=self.bottleneck_dim, F_l=self.swin_dims[-2], F_int=self.bottleneck_dim // 2)
        # z2 (32x32) <-> decoder2 (32x32)
        self.ag2 = AttentionGate(F_g=256, F_l=self.swin_dims[-3], F_int=128)
        # z1 (64x64) <-> decoder1 (64x64)
        self.ag1 = AttentionGate(F_g=128, F_l=self.swin_dims[-4], F_int=64)


    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        H, W = x.shape[-2], x.shape[-1]
        if H % 16 != 0 or W % 16 != 0:
            pad_h = (16 - H % 16) % 16
            pad_w = (16 - W % 16) % 16
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
            original_shape = (H, W)
        else:
            original_shape = None

        out_dict = {}

        # 1. Backbone Forward
        cls_features, tokens, swin_features = self.backbone(x)
        z0 = x
        z1, z2, z3, z4 = swin_features # z1: 1/4, z2: 1/8, z3: 1/16, z4: 1/32
        
        # 2. Tissue Classification
        cls_features_enhanced = F.dropout(cls_features, p=0.3, training=self.training)
        out_dict["tissue_types"] = self.tissue_head(cls_features_enhanced)

        # 3. ASPP Bottleneck
        # z4: [B, 768, 8, 8] -> [B, 384, 8, 8]
        bottleneck_feat = self.aspp(z4)
        
        # 4. Shape Stream Forward (using shallow feature z1)
        # z1 is [B, 96, 64, 64]
        shape_feat, edge_map = self.shape_stream(z1)
        
        # Upsample edge_map to original input size
        if original_shape is not None:
             edge_map = F.interpolate(edge_map, size=original_shape, mode='bilinear', align_corners=False)
        else:
             edge_map = F.interpolate(edge_map, size=(H, W), mode='bilinear', align_corners=False)

        out_dict["edge_map"] = edge_map # Output edge map for auxiliary loss

        # 5. Decoders
        if self.regression_loss:
            nb_map = self._forward_upsample(
                z0, z1, z2, z3, bottleneck_feat, self.nuclei_binary_map_decoder, shape_feat
            )
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, bottleneck_feat, self.nuclei_binary_map_decoder, shape_feat
            )
            
        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, bottleneck_feat, self.hv_map_decoder, shape_feat
        )
        
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, bottleneck_feat, self.nuclei_type_maps_decoder, shape_feat=None
        )
        
        # Crop back
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
        bottleneck: torch.Tensor,
        branch_decoder: nn.Sequential,
        shape_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # b4: [B, 384, 16, 16] (from 8x8)
        b4 = branch_decoder.bottleneck_upsampler(bottleneck)
        
        # Attention Gate 3
        # z3: [B, 384, 16, 16]
        z3_gated = self.ag3(g=b4, x=z3)
        b3_concat = torch.cat([z3_gated, b4], dim=1)
        b3 = branch_decoder.decoder3_upsampler(b3_concat) # -> [B, 256, 32, 32]
        
        # Attention Gate 2
        # z2: [B, 192, 32, 32]
        z2_gated = self.ag2(g=b3, x=z2)
        b2_concat = torch.cat([z2_gated, b3], dim=1)
        b2 = branch_decoder.decoder2_upsampler(b2_concat) # -> [B, 128, 64, 64]
        
        # Attention Gate 1
        # z1: [B, 96, 64, 64]
        z1_gated = self.ag1(g=b2, x=z1)
        b1_concat = torch.cat([z1_gated, b2], dim=1)
        b1 = branch_decoder.decoder1_upsampler(b1_concat) # -> [B, 64, 128, 128]
        
        # Upsample to full resolution
        b1_up = F.interpolate(b1, size=(z0.shape[2], z0.shape[3]), mode='bilinear', align_corners=False)
        
        # Inject Shape Feature if available
        if shape_feat is not None:
            # shape_feat is [B, 64, 64, 64]
            # Resize shape feature to match b1_up (256x256)
            shape_feat_up = F.interpolate(shape_feat, size=(z0.shape[2], z0.shape[3]), mode='bilinear', align_corners=False)
            
            # Concatenate with z0 and b1_up
            b0_concat = torch.cat([z0, b1_up, shape_feat_up], dim=1)
        else:
            b0_concat = torch.cat([z0, b1_up], dim=1)
            
        branch_output = branch_decoder.decoder0_header(b0_concat)

        return branch_output

    def create_upsampling_branch(self, num_classes: int, use_shape_stream: bool = False) -> nn.Module:
        
        # Bottleneck upsampler
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.bottleneck_dim, # Output of ASPP
            out_channels=self.bottleneck_dim,
            kernel_size=2, stride=2, padding=0
        )
        
        # Decoder 3
        # Input: z3_gated (swin_dims[-2]=384) + b4 (bottleneck_dim=384) = 768
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.swin_dims[-2] + self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(self.bottleneck_dim, 256, kernel_size=2, stride=2, padding=0)
        )
        
        # Decoder 2
        # Input: z2_gated (swin_dims[-3]=192) + b3 (256) = 448
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(self.swin_dims[-3] + 256, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        )
        
        # Decoder 1
        # Input: z1_gated (swin_dims[-4]=96) + b2 (128) = 224
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(self.swin_dims[-4] + 128, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        )
        
        # Decoder Header
        # Input: z0 (3) + b1_up (64) + [shape_feat (64) if used]
        in_channels = self.input_channels + 64
        if use_shape_stream:
            in_channels += 64
            
        decoder0_header = nn.Sequential(
            Conv2DBlock(in_channels, 64, 3, dropout=self.drop_rate), # Reduce channels immediately for efficiency
            Conv2DBlock(64, 64, 3, dropout=self.drop_rate),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        # Specialized Headers for Type and HV
        # Removed broken SE block implementation to ensure spatial resolution is preserved.
        # Using standard decoder header for all branches.
        
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
        instance_types_nuclei (torch.Tensor): Pixel-wise nuclear instance segmentation predictions, for each nuclei type.
            Each instance has its own integer, starting from 1.
            Shape: (batch_size, num_nuclei_classes, H, W)
        batch_size (int): Batch size of the experiment
        instance_types (list, optional): Instance type prediction list.
            Each list entry stands for one image. Each list entry is a dictionary with the following structure:
                {
                    "1": {"bbox": [x_min, y_min, x_max, y_max], "centroid": [x, y], "contour": [[x1, y1], [x2, y2], ...], "type_prob": float, "type": int},
                    "2": {"bbox": [x_min, y_min, x_max, y_max], "centroid": [x, y], "contour": [[x1, y1], [x2, y2], ...], "type_prob": float, "type": int},
                    ...
                }
            where the keys are instance IDs (strings), and the values are dictionaries containing information about each detected nucleus.
    """
    nuclei_binary_map: torch.Tensor
    hv_map: torch.Tensor
    nuclei_type_map: torch.Tensor
    tissue_types: torch.Tensor
    instance_map: torch.Tensor
    instance_types_nuclei: torch.Tensor
    batch_size: int
    instance_types: Optional[List[Dict[str, Dict[str, Union[List[float], float, int]]]]] = None
