import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, List, Tuple, Optional

class LocalSparseAttention(nn.Module):
    """局部稀疏注意力机制
    在固定大小的局部窗口内计算注意力，降低计算复杂度
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 16,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "使用相对位置编码时必须提供输入尺寸"
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size - 1, head_dim))
            if rel_pos_zero_init:
                nn.init.zeros_(self.rel_pos_h)
                nn.init.zeros_(self.rel_pos_w)
            else:
                # 使用 Xavier 初始化，提高模型收敛速度
                nn.init.xavier_uniform_(self.rel_pos_h)
                nn.init.xavier_uniform_(self.rel_pos_w)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        # 将输入重塑为空间维度，便于窗口划分
        x = x.view(B, H, W, C)
        
        # 窗口划分
        x, pad_hw = window_partition(x, self.window_size)
        
        # 计算每个窗口内的注意力
        B_w, H_w, W_w, C = x.shape
        x = x.reshape(B_w, H_w * W_w, C)
        
        # QKV计算
        qkv = self.qkv(x).reshape(B_w, H_w * W_w, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 添加相对位置编码（如果启用）
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, 
                (H_w, W_w), (H_w, W_w)
            )
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_w, H_w * W_w, C)
        
        # 恢复窗口
        x = x.reshape(B_w, H_w, W_w, C)
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        
        # 重塑回原始形状
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn

class ContentBasedSparseAttention(nn.Module):
    """基于内容的稀疏注意力机制
    只对信息熵高的关键区域计算全注意力，支持动态k值调整
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        top_k_ratio: float = 0.2,  # 关键特征向量比例
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dynamic_k: bool = False,
        min_k_ratio: float = 0.1,
        max_k_ratio: float = 0.7,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.top_k_ratio = top_k_ratio
        self.dynamic_k = dynamic_k
        self.min_k_ratio = min_k_ratio
        self.max_k_ratio = max_k_ratio
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 改进的显著性检测模块：增加隐藏层、层归一化和跳跃连接
        self.saliency_detector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        # 跳跃连接
        self.saliency_shortcut = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        
        # 检测关键特征向量（使用改进的显著性检测模块和跳跃连接）
        saliency_scores = self.saliency_detector(x).squeeze(-1)  # [B, N]
        shortcut_scores = self.saliency_shortcut(x).squeeze(-1)  # [B, N]
        saliency_scores = saliency_scores + shortcut_scores  # 添加跳跃连接
        
        # 动态k值调整
        if self.dynamic_k:
            # 计算特征熵来动态调整k值
            saliency_probs = saliency_scores.softmax(dim=-1)
            entropy = -torch.sum(saliency_probs * torch.log(saliency_probs + 1e-8), dim=-1)
            entropy = entropy / torch.log(torch.tensor(N, dtype=torch.float32))
            current_k_ratio = self.min_k_ratio + (self.max_k_ratio - self.min_k_ratio) * entropy
            current_k_ratio = torch.clamp(current_k_ratio, self.min_k_ratio, self.max_k_ratio)
            top_k = (current_k_ratio * N).long().clamp(min=1)
        else:
            top_k = max(1, int(N * self.top_k_ratio))
            current_k_ratio = torch.tensor([self.top_k_ratio] * B, device=x.device)
        
        # 为每个样本选择top-k个关键特征
        top_k = top_k.clamp(max=N)
        _, indices = torch.topk(saliency_scores, top_k.max().item(), dim=1)  # [B, K]
        
        # 构建掩码
        mask = torch.zeros_like(saliency_scores, dtype=torch.bool)
        for b in range(B):
            mask[b, indices[b, :top_k[b]]] = True
        
        # 只对关键特征计算全注意力
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 为每个头部计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 对非关键区域应用稀疏掩码
        mask_reshaped = mask.unsqueeze(1).unsqueeze(2).repeat(1, self.num_heads, N, 1)
        attn = attn.masked_fill(~mask_reshaped, -float('inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn

class MixedSparseAttention(nn.Module):
    """混合稀疏注意力机制
    结合局部稀疏注意力和基于内容的稀疏注意力
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        local_head_ratio: float = 0.6,  # 局部注意力头比例
        window_size: int = 16,
        top_k_ratio: float = 0.2,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        dynamic_k: bool = False,
        min_k_ratio: float = 0.1,
        max_k_ratio: float = 0.7,
        local_attn_weight: float = 0.7,  # 局部注意力融合权重
        content_attn_weight: float = 0.3,  # 内容注意力融合权重
    ):
        super().__init__()
        self.num_heads = num_heads
        # 使用向上取整确保局部注意力头数量正确，并限制在合理范围
        self.num_local_heads = max(1, min(num_heads - 1, int(math.ceil(num_heads * local_head_ratio))))
        self.num_content_heads = max(1, num_heads - self.num_local_heads)
        
        # 注意力头剪枝机制：可学习的注意力头权重
        self.local_head_weights = nn.Parameter(torch.ones(self.num_local_heads))
        self.content_head_weights = nn.Parameter(torch.ones(self.num_content_heads))
        self.head_softmax = nn.Softmax(dim=0)
        
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 分别为局部注意力和内容注意力创建线性投影
        self.local_qkv = nn.Linear(dim, self.num_local_heads * head_dim * 3, bias=qkv_bias)
        self.content_qkv = nn.Linear(dim, self.num_content_heads * head_dim * 3, bias=qkv_bias)
        
        self.local_proj = nn.Linear(self.num_local_heads * head_dim, dim)
        self.content_proj = nn.Linear(self.num_content_heads * head_dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.window_size = window_size
        self.top_k_ratio = top_k_ratio
        
        # 改进的显著性检测模块：增加隐藏层、层归一化和跳跃连接
        self.saliency_detector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        # 跳跃连接
        self.saliency_shortcut = nn.Linear(dim, 1)
        
        # 动态k值参数
        self.dynamic_k = dynamic_k
        self.min_k_ratio = min_k_ratio
        self.max_k_ratio = max_k_ratio
        
        # 可学习的注意力融合权重
        self.fusion_weight = nn.Parameter(torch.tensor([local_attn_weight, content_attn_weight], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        
        # 相对位置编码
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "使用相对位置编码时必须提供输入尺寸"
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size - 1, head_dim))
            # 使用 Xavier 初始化，提高模型收敛速度
            nn.init.xavier_uniform_(self.rel_pos_h)
            nn.init.xavier_uniform_(self.rel_pos_w)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        
        # 检查是否包含CLS token
        has_cls_token = False
        cls_token = None
        if N > 1:
            # 尝试计算H和W，检查是否为整数
            H = int(math.sqrt(N))
            if H * H != N:
                # 包含CLS token，分离出来
                has_cls_token = True
                cls_token = x[:, 0:1, :]
                patch_tokens = x[:, 1:, :]
                B, N_patch, C = patch_tokens.shape
                H = W = int(math.sqrt(N_patch))
                x = patch_tokens
                N = N_patch  # 更新N为补丁标记的数量
            else:
                # 不包含CLS token
                W = H
        else:
            # 只有CLS token，直接返回
            return x, None
        
        head_dim = C // self.num_heads
        
        # 保存原始输入尺寸，用于后续还原
        original_H, original_W = H, W
        
        # 处理非窗口大小倍数的情况：动态padding
        pad_H = (self.window_size - H % self.window_size) % self.window_size
        pad_W = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_H > 0 or pad_W > 0:
            # 对输入进行padding以适配窗口大小
            x = x.view(B, H, W, C)
            x = F.pad(x, (0, 0, 0, pad_W, 0, pad_H), mode='reflect')
            x = x.view(B, (H + pad_H) * (W + pad_W), C)
            H, W = H + pad_H, W + pad_W
        
        # 1. 局部稀疏注意力计算
        # 将输入重塑为空间维度，便于窗口划分
        x_spatial = x.view(B, H, W, C)
        
        # 窗口划分
        x_local_win, pad_hw = window_partition(x_spatial, self.window_size)
        
        # 计算每个窗口内的注意力
        B_w, H_w, W_w, C = x_local_win.shape
        x_local = x_local_win.reshape(B_w, H_w * W_w, C)
        
        # QKV计算（仅局部注意力头）
        local_qkv = self.local_qkv(x_local).reshape(B_w, H_w * W_w, 3, self.num_local_heads, head_dim)
        local_qkv = local_qkv.permute(2, 0, 3, 1, 4)
        local_q, local_k, local_v = local_qkv[0], local_qkv[1], local_qkv[2]
        
        # 计算局部注意力
        local_attn = (local_q @ local_k.transpose(-2, -1)) * self.scale
        
        # 添加相对位置编码（如果启用）
        if self.use_rel_pos:
            local_attn = add_decomposed_rel_pos(
                local_attn, local_q, self.rel_pos_h, self.rel_pos_w, 
                (H_w, W_w), (H_w, W_w)
            )
        
        local_attn = local_attn.softmax(dim=-1)
        local_attn = self.attn_drop(local_attn)
        
        local_x = (local_attn @ local_v).transpose(1, 2)  # [B_w, H_w*W_w, num_local_heads, head_dim]
        
        # 应用注意力头权重
        local_head_weights = self.head_softmax(self.local_head_weights)
        local_head_weights = local_head_weights.view(1, 1, self.num_local_heads, 1)
        local_x = (local_x * local_head_weights).reshape(B_w, H_w * W_w, self.num_local_heads * head_dim)
        
        # 恢复窗口
        local_x = local_x.reshape(B_w, H_w, W_w, self.num_local_heads * head_dim)
        local_x = window_unpartition(local_x, self.window_size, pad_hw, (H, W))
        
        # 重塑回原始形状
        local_x = local_x.reshape(B, H * W, self.num_local_heads * head_dim)
        local_x = self.local_proj(local_x)
        
        # 还原到原始尺寸（移除padding）
        if pad_H > 0 or pad_W > 0:
            local_x = local_x.view(B, H, W, self.num_local_heads * head_dim)
            local_x = local_x[:, :original_H, :original_W, :].contiguous()
            local_x = local_x.view(B, original_H * original_W, self.num_local_heads * head_dim)
        
        # 2. 基于内容的稀疏注意力计算
        # 检测关键特征向量（使用改进的显著性检测模块和跳跃连接）
        saliency_scores = self.saliency_detector(x).squeeze(-1)  # [B, N]
        shortcut_scores = self.saliency_shortcut(x).squeeze(-1)  # [B, N]
        saliency_scores = saliency_scores + shortcut_scores  # 添加跳跃连接
        
        # 动态k值调整
        if self.dynamic_k:
            # 训练时使用动态k值，推理时使用固定k值以确保输出稳定
            if self.training:
                # 计算特征熵来动态调整k值
                saliency_probs = F.softmax(saliency_scores, dim=-1)
                entropy = -torch.sum(saliency_probs * torch.log(saliency_probs + 1e-8), dim=-1)
                entropy = entropy / torch.log(torch.tensor(N, dtype=saliency_scores.dtype, device=saliency_scores.device))
                current_k_ratio = self.min_k_ratio + (self.max_k_ratio - self.min_k_ratio) * entropy
                current_k_ratio = torch.clamp(current_k_ratio, self.min_k_ratio, self.max_k_ratio)
                top_k = (current_k_ratio * N).long().clamp(min=1)
            else:
                # 推理时使用固定k值，确保输出稳定可复现
                top_k = torch.tensor([max(1, int(N * self.top_k_ratio))], device=saliency_scores.device).expand(B)
        else:
            # 使用固定k值
            top_k = torch.tensor([max(1, int(N * self.top_k_ratio))], device=saliency_scores.device).expand(B)
        
        # 确保top_k不超过N
        top_k = torch.clamp(top_k, max=N)
        
        # 为每个样本选择top-k个关键特征
        _, indices = torch.topk(saliency_scores, top_k.max().item(), dim=1)  # [B, K]
        
        # 构建掩码 - 确保与输入设备对齐
        device = saliency_scores.device
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            mask[b, indices[b, :top_k[b]]] = True
        
        # QKV计算（仅内容注意力头）
        content_qkv = self.content_qkv(x).reshape(B, N, 3, self.num_content_heads, head_dim)
        content_qkv = content_qkv.permute(2, 0, 3, 1, 4)
        content_q, content_k, content_v = content_qkv[0], content_qkv[1], content_qkv[2]
        
        # 计算内容注意力
        content_attn = (content_q @ content_k.transpose(-2, -1)) * self.scale
        
        # 对非关键区域应用稀疏掩码 - 设备对齐
        mask_reshaped = mask.unsqueeze(1).unsqueeze(2).repeat(1, self.num_content_heads, N, 1)
        content_attn = content_attn.masked_fill(~mask_reshaped, -float('inf'))
        
        content_attn = content_attn.softmax(dim=-1)
        content_attn = self.attn_drop(content_attn)
        
        content_x = (content_attn @ content_v).transpose(1, 2)  # [B, N, num_content_heads, head_dim]
        
        # 应用注意力头权重
        content_head_weights = self.head_softmax(self.content_head_weights)
        content_head_weights = content_head_weights.view(1, 1, self.num_content_heads, 1)
        content_x = (content_x * content_head_weights).reshape(B, N, self.num_content_heads * head_dim)
        content_x = self.content_proj(content_x)
        
        # 还原到原始尺寸（移除padding）
        if pad_H > 0 or pad_W > 0:
            content_x = content_x.view(B, H, W, self.num_content_heads * head_dim)
            content_x = content_x[:, :original_H, :original_W, :].contiguous()
            content_x = content_x.view(B, original_H * original_W, self.num_content_heads * head_dim)
        
        # 3. 融合两种注意力的输出（使用可学习的加权融合）
        weights = self.softmax(self.fusion_weight)
        x = weights[0] * local_x + weights[1] * content_x
        x = self.proj_drop(x)
        
        # 如果输入包含CLS token，将其重新添加回去
        if has_cls_token and cls_token is not None:
            x = torch.cat([cls_token, x], dim=1)
        
        # 融合注意力图（用于可视化）
        # 只返回局部注意力图，避免维度不匹配问题
        attn = local_attn.mean(1)
        
        return x, attn

class SparseVisionTransformerBlock(nn.Module):
    """使用稀疏自注意力的Transformer块"""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attention_type: str = "mixed",  # "local", "content", "mixed"
        window_size: int = 16,
        top_k_ratio: float = 0.2,
        local_head_ratio: float = 0.6,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        dynamic_k: bool = False,
        min_k_ratio: float = 0.1,
        max_k_ratio: float = 0.7,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # 根据指定类型创建注意力模块
        if attention_type == "local":
            self.attn = LocalSparseAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                use_rel_pos=use_rel_pos,
                input_size=input_size,
            )
        elif attention_type == "content":
            self.attn = ContentBasedSparseAttention(
                dim=dim,
                num_heads=num_heads,
                top_k_ratio=top_k_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                dynamic_k=dynamic_k,
                min_k_ratio=min_k_ratio,
                max_k_ratio=max_k_ratio,
            )
        elif attention_type == "mixed":
            self.attn = MixedSparseAttention(
                dim=dim,
                num_heads=num_heads,
                local_head_ratio=local_head_ratio,
                window_size=window_size,
                top_k_ratio=top_k_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                use_rel_pos=use_rel_pos,
                input_size=input_size,
                dynamic_k=dynamic_k,
                min_k_ratio=min_k_ratio,
                max_k_ratio=max_k_ratio,
            )
        else:
            raise ValueError(f"不支持的注意力类型: {attention_type}")
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# 辅助函数
class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: Callable = nn.GELU,
        drop: float = 0.0,
    ):
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

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适配不同维度的张量
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        output = x.div(keep_prob) * random_tensor
        return output

def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """将张量划分为不重叠的窗口"""
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)

def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """将窗口合并回原始张量"""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """获取相对位置编码"""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # 必要时插值相对位置编码
    if rel_pos.shape[0] != max_rel_dist:
        # 使用线性插值，适合1D数据
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # 精确计算相对坐标
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None]
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :]
    relative_coords = (q_coords - k_coords) + (k_size - 1)
    # 确保坐标在有效范围内
    relative_coords = torch.clamp(relative_coords, 0, max_rel_dist - 1)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """添加分解的相对位置编码"""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    # 检查q的形状，处理包含注意力头维度的情况
    if len(q.shape) == 4:
        # 形状为 (B, num_heads, q_h*q_w, dim)
        B, num_heads, _, dim = q.shape
        r_q = q.reshape(B, num_heads, q_h, q_w, dim)
        rel_h = torch.einsum("bnhwc,hkc->bnhwk", r_q, Rh)
        rel_w = torch.einsum("bnhwc,wkc->bnhwk", r_q, Rw)

        attn = (
            attn.view(B, num_heads, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, :, None]
            + rel_w[:, :, :, :, None, :]
        ).view(B, num_heads, q_h * q_w, k_h * k_w)
    else:
        # 原始形状 (B, q_h*q_w, dim)
        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (
            attn.view(B, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None]
            + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)

    return attn