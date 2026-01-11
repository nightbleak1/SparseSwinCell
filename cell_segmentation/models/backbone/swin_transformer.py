import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = dim ** -0.5
        
        # 动态km注意力参数
        self.km_k_base = 5
        self.km_k_max = 20
        
        # 全局注意力比例
        self.global_attn_ratio = 0.1
        
        # 稀疏VIT参数
        self.sparsity_threshold = 0.1  # 稀疏性阈值
        self.content_based_sparsity = True  # 基于内容的稀疏性
        self.hierarchical_sparsity = True  # 层级稀疏性

        # 原始qkv线性层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 额外的全局qkv线性层
        self.global_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 内容感知门控
        self.content_gate = nn.Linear(dim, dim)
        
        # 层级稀疏性门控
        self.hierarchical_gate = nn.Linear(dim, 1)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 注意力融合层
        self.attn_fusion = nn.Linear(dim * 2, dim)
        
        # 残差连接
        self.residual = nn.Identity()

    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        # 1. 层级稀疏性：动态决定是否跳过某些头部或窗口
        if self.hierarchical_sparsity:
            # 计算每个样本的重要性分数
            importance = self.hierarchical_gate(x.mean(dim=1)).sigmoid()
            # 根据重要性分数决定是否跳过计算
            if importance.mean() < self.sparsity_threshold:
                # 跳过复杂计算，直接返回残差
                return self.residual(x)
        
        # 2. 基于内容的稀疏性：只处理重要的内容
        if self.content_based_sparsity:
            # 计算内容重要性门控
            content_gate = self.content_gate(x).sigmoid()
            # 只保留重要内容
            important_mask = (content_gate > self.sparsity_threshold).float()
            x = x * important_mask
        
        # 3. 原始局部注意力
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        local_attn = (q @ k.transpose(-2, -1)) * self.scale
        # 只在mask不为None且尺寸匹配时添加mask
        if mask is not None and mask.shape == local_attn.shape:
            local_attn = local_attn + mask
        local_attn = local_attn.softmax(dim=-1)
        local_attn = self.attn_drop(local_attn)

        # 4. 增强的动态km注意力：根据注意力分布动态调整k值
        attn_entropy = -torch.sum(local_attn * torch.log(local_attn + 1e-10), dim=-1).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        # 熵越大，k值越大，保留更多注意力信息
        km_k = self.km_k_base + torch.round((self.km_k_max - self.km_k_base) * attn_entropy).long().item()
        km_k = max(self.km_k_base, min(km_k, self.km_k_max))
        
        # 仅保留top-k的注意力权重
        topk_attn, topk_indices = torch.topk(local_attn, km_k, dim=-1)
        # 创建稀疏注意力掩码
        sparse_mask = torch.zeros_like(local_attn)
        sparse_mask.scatter_(-1, topk_indices, 1.0)
        # 应用掩码，只保留top-k注意力
        local_attn = local_attn * sparse_mask

        local_x = (local_attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 5. 稀疏全局注意力：使用低秩近似和稀疏采样
        global_qkv = self.global_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        g_q, g_k, g_v = global_qkv[0], global_qkv[1], global_qkv[2]
        
        # 降低维度以减少计算量
        reduce_dim = max(1, C // self.num_heads // 8)
        g_k = g_k.reshape(B, self.num_heads, N, reduce_dim, -1).mean(dim=-1)
        g_v = g_v.reshape(B, self.num_heads, N, reduce_dim, -1).mean(dim=-1)
        
        global_attn = (g_q @ g_k.transpose(-2, -1)) * self.scale
        global_attn = global_attn.softmax(dim=-1)
        global_x = (global_attn @ g_v).transpose(1, 2).reshape(B, N, C)
        
        # 6. 融合局部和全局注意力
        combined_x = torch.cat([local_x, global_x], dim=-1)
        fused_x = self.attn_fusion(combined_x)
        
        # 7. 残差连接
        x = self.residual(x) + fused_x
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 layer_scale_init_value=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.layer_scale_init_value = layer_scale_init_value

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, window_size=window_size, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # Swin V2: 添加LayerScale
        if layer_scale_init_value > 0:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale_1 = None

        self.drop_path = nn.Identity() if drop_path <= 0. else StochasticDepth(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Swin V2: 添加LayerScale
        if layer_scale_init_value > 0:
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale_2 = None

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self.calculate_mask(H, W)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        
        # Swin V2: 应用LayerScale
        if self.layer_scale_1 is not None:
            x = self.layer_scale_1 * x
        
        x = shortcut + self.drop_path(x)

        # Swin V2: 应用LayerScale
        mlp_output = self.mlp(self.norm2(x))
        if self.layer_scale_2 is not None:
            mlp_output = self.layer_scale_2 * mlp_output
        
        x = x + self.drop_path(mlp_output)
        return x

    def calculate_mask(self, H, W):
        # 只在需要时计算mask
        if self.window_size <= self.shift_size:
            return None
        
        # 创建mask
        img_mask = torch.zeros((1, H, W, 1))
        
        # 计算mask区域
        h_slices = [
            (0, -self.window_size),
            (-self.window_size, -self.shift_size),
            (-self.shift_size, None)
        ]
        w_slices = [
            (0, -self.window_size),
            (-self.window_size, -self.shift_size),
            (-self.shift_size, None)
        ]
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h[0]:h[1], w[0]:w[1], :] = cnt
                cnt += 1

        # 分割窗口
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        
        # 计算注意力mask
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        
        # 确保H和W是偶数，否则进行裁剪
        if H % 2 != 0:
            H -= 1
            x = x[:, :H, :, :]
        if W % 2 != 0:
            W -= 1
            x = x[:, :, :W, :]

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x, H // 2, W // 2

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 layer_scale_init_value=1e-6):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layer_scale_init_value = layer_scale_init_value

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.downsample = PatchMerging(dim=dim, norm_layer=norm_layer) if depth > 0 else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x, H // self.patch_size, W // self.patch_size

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class SwinTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=128,
                 depths=(2, 4, 12, 4), num_heads=(4, 8, 16, 32), window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True, extract_layers=None, layer_scale_init_value=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.extract_layers = extract_layers if extract_layers is not None else [self.num_layers - 1]
        self.layer_scale_init_value = layer_scale_init_value

        # Swin V2: 使用更大的嵌入维度和更深的网络结构
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.img_size // self.patch_embed.patch_size
        self.patches_resolution = patches_resolution

        # Swin V2: 移除APE，使用相对位置编码
        self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # DropPath: 使用线性增长的drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                layer_scale_init_value=self.layer_scale_init_value)
            self.layers.append(layer)

        # 计算最终通道数
        final_embed_dim = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(final_embed_dim)
        
        # 初始化或修复head层
        if num_classes > 0:
            self.head = nn.Linear(final_embed_dim, num_classes)
        else:
            self.head = None

        # Swin V2: 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, H, W = self.patch_embed(x)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        features = []
        for i, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W)
            if i in self.extract_layers:
                features.append(x.view(-1, H, W, x.size(-1)).permute(0, 3, 1, 2))

        x = self.norm(x)

        if self.head is not None:
            x = self.head(x)

        return features

class SwinTransformerWithKMAttention(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=0, embed_dim=128,
                 depths=(2, 4, 12, 4), num_heads=(4, 8, 16, 32), window_size=8, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, extract_layers=None, layer_scale_init_value=1e-6):
        super().__init__()
        self.swin = SwinTransformerV2(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            extract_layers=extract_layers,
            layer_scale_init_value=layer_scale_init_value
        )

    def forward(self, x):
        # 支持任意尺寸输入
        B, C, H, W = x.shape
        
        # 检查并调整尺寸以适应patch_size，确保能被window_size整除
        orig_H, orig_W = H, W
        
        # 首先确保尺寸能被patch_size整除
        patch_size = self.swin.patch_embed.patch_size
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2], x.shape[3]
        
        # 然后确保patch数量能被window_size整除
        window_size = self.swin.layers[0].blocks[0].window_size
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        pad_patches_h = (window_size - num_patches_h % window_size) % window_size
        pad_patches_w = (window_size - num_patches_w % window_size) % window_size
        
        if pad_patches_h > 0 or pad_patches_w > 0:
            # 将pad_patches转换为像素级别的padding
            pad_h_pixels = pad_patches_h * patch_size
            pad_w_pixels = pad_patches_w * patch_size
            x = F.pad(x, (0, pad_w_pixels, 0, pad_h_pixels))
            H, W = x.shape[2], x.shape[3]
        
        # 前向传播
        features = self.swin(x)
        
        # 裁剪回原始尺寸
        if pad_h > 0 or pad_w > 0:
            for i in range(len(features)):
                scale = orig_H // features[i].shape[2]
                features[i] = features[i][:, :, :orig_H // scale, :orig_W // scale]
        
        return features