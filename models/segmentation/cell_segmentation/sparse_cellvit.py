from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor

from .sparse_utils import Conv2DBlock, Deconv2DBlock, SparseViTCellViT, MultiScaleSparseAttention

class SparseCellViT(nn.Module):
    """使用稀疏自注意力机制的CellViT模型"""
    def __init__(
        self,
        num_nuclei_classes: int,
        num_tissue_classes: int,
        embed_dim: int,
        input_channels: int,
        depth: int,
        num_heads: int,
        extract_layers: List,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,
        attention_type: str = "mixed",  # "local", "content", "mixed"
        window_size: int = 16,
        top_k_ratio: float = 0.2,
        local_head_ratio: float = 0.6,
    ):
        # 为简单起见，我们假设extract_layers的长度必须为4
        super().__init__()
        assert len(extract_layers) == 4, "Please provide 4 layers for skip connections"

        self.patch_size = 16
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

        # 使用稀疏自注意力编码器
        self.encoder = SparseViTCellViT(
            patch_size=self.patch_size,
            num_classes=self.num_tissue_classes,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            extract_layers=self.extract_layers,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            attention_type=attention_type,
            window_size=window_size,
            top_k_ratio=top_k_ratio,
            local_head_ratio=local_head_ratio,
        )

        # 多尺度特征融合前的稀疏注意力增强
        self.scale_attention_layers = nn.ModuleList()
        for i in range(len(extract_layers)):
            # 低分辨率特征图使用较大窗口，高分辨率特征图使用较小窗口
            scale_window_size = window_size // (2 ** (3 - i))
            self.scale_attention_layers.append(
                MultiScaleSparseAttention(
                    dim=self.embed_dim,
                    window_size=scale_window_size,
                    dropout=drop_rate
                )
            )

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # 共享跳跃连接的解码器
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
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
        )
        self.hv_map_decoder = self.create_upsampling_branch(2)
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(
            self.num_nuclei_classes
        )

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        """前向传播"""
        assert (
            x.shape[-2] % self.patch_size == 0
        ), "Img must have a shape of that is divisible by patch_size (token_size)"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), "Img must have a shape of that is divisible by patch_size (token_size)"

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = classifier_logits

        z0, z1, z2, z3, z4 = x, *z

        # 执行重塑以适应卷积层和上采样（恢复空间维度）
        patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        
        # 对每个尺度的特征图应用稀疏注意力增强
        for i in range(len(z)):
            # 重塑为空间维度
            z[i] = z[i][:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
            # 应用多尺度稀疏注意力
            z[i] = self.scale_attention_layers[i](z[i])
        
        z4, z3, z2, z1 = z

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

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        """前向传播上采样分支"""
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        return branch_output

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """创建上采样分支"""
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
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
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
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
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
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
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
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
        """从网络预测计算实例图（Softmax输出后）"""
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
        """将实例图（二进制）转换为细胞核类型实例图"""
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
        """冻结编码器以不训练它"""
        for layer_name, p in self.encoder.named_parameters():
            if layer_name.split(".")[0] != "head":  # 不冻结头部
                p.requires_grad = False

    def unfreeze_encoder(self):
        """解冻编码器以训练整个模型"""
        for p in self.encoder.parameters():
            p.requires_grad = True

# 为了方便使用，我们还可以创建一个SparseCellViT256类，类似于原始的CellViT256
class SparseCellViT256(SparseCellViT):
    """使用稀疏自注意力的CellViT256"""
    def __init__(
        self,
        model256_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,
        attention_type: str = "mixed",
        window_size: int = 16,
        top_k_ratio: float = 0.2,
        local_head_ratio: float = 0.6,
    ):
        self.patch_size = 16
        self.embed_dim = 384
        self.depth = 12
        self.num_heads = 6
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.extract_layers = [3, 6, 9, 12]
        self.input_channels = 3  # RGB
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
            attention_type=attention_type,
            window_size=window_size,
            top_k_ratio=top_k_ratio,
            local_head_ratio=local_head_ratio,
        )

        self.model256_path = model256_path

    def load_pretrained_encoder(self, model256_path: str):
        """从提供的路径加载预训练的ViT-256"""
        state_dict = torch.load(str(model256_path), map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")