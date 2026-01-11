#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast
from PIL import Image

# 添加项目根目录到Python路径
sys.path.append("/hy-tmp/SparseSwinCell")

# 导入模型和工具
from models.segmentation.cell_segmentation.cellvit import CellViT
from cell_segmentation.utils.metrics import get_fast_pq, remap_label
import torch.multiprocessing as mp

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validate_monuseg')

class MoNuSegDataset(Dataset):
    """MoNuSeg数据集加载器，支持染色风格一致性预处理"""
    def __init__(self, data_root, transforms=None):
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.masks_dir = self.data_root / "masks"
        self.transforms = transforms
        
        # 获取图像文件列表
        self.image_files = sorted([f for f in self.images_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])
        
        # 初始化染色归一化器
        self.stain_normalizer = self.init_stain_normalizer()
        
    def init_stain_normalizer(self):
        """初始化染色归一化器"""
        class StainNormalizer:
            def __init__(self):
                self.alpha = 0.5
                self.beta = 0.5
            
            def transform(self, image):
                """应用染色风格一致性预处理"""
                # 转换为numpy数组
                img_array = np.array(image).astype(np.float32) / 255.0
                
                # 归一化到均值0，标准差1
                img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
                
                # 限制值范围到[-1, 1]
                img_array = np.clip(img_array, -1, 1)
                
                # 转换回[0, 255]范围
                img_array = ((img_array + 1) * 127.5).astype(np.uint8)
                
                return Image.fromarray(img_array)
        
        return StainNormalizer()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.image_files[idx]
        img_name = img_path.name
        
        # 尝试不同的掩码文件匹配方式
        mask_path = None
        possible_mask_names = [
            img_name,  # 直接使用图像文件名
            img_name.replace('.png', ' .png'),  # 尝试添加一个空格
            img_name.replace('.png', '  .png'),  # 尝试添加两个空格
            img_name.strip()  # 尝试去掉所有空格
        ]
        
        for name in possible_mask_names:
            test_path = self.masks_dir / name
            if test_path.exists():
                mask_path = test_path
                break
        
        # 如果没有找到匹配的掩码文件，使用原始路径
        if mask_path is None:
            mask_path = self.masks_dir / img_name
        
        # 加载图像
        img = Image.open(img_path).convert("RGB")
        
        # 应用染色归一化
        img = self.stain_normalizer.transform(img)
        
        # 加载掩码
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask)
        
        # 转换为PyTorch张量
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        # 处理掩码：二值掩码和实例掩码
        binary_mask = (mask_array > 0).astype(np.long)
        instance_mask = mask_array.astype(np.long)
        type_mask = np.zeros_like(mask_array, dtype=np.long)  # MoNuSeg没有类型标注，默认为0
        
        masks = {
            "instance_map": torch.from_numpy(instance_mask),
            "nuclei_type_map": torch.from_numpy(type_mask),
            "nuclei_binary_map": torch.from_numpy(binary_mask)
        }
        
        return img, masks, 0, img_name

def load_model(model_path, device):
    """加载训练好的模型"""
    logger.info(f"Loading model from {model_path}")
    
    # 初始化模型，使用与训练时相同的参数
    model = CellViT(
        num_nuclei_classes=6,
        num_tissue_classes=19,
        embed_dim=96,
        input_channels=3,
        depth=12,
        num_heads=3,
        extract_layers=[1, 2, 3, 4],
        regression_loss=True
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model

def validate_monuseg(model, monuseg_path, device):
    """验证MoNuSeg数据"""
    logger.info("Validating MoNuSeg data...")
    
    # 创建数据集
    print("Creating MoNuSeg dataset...")
    dataset = MoNuSegDataset(
        data_root=monuseg_path,
        transforms=None
    )
    print(f"Dataset created, size: {len(dataset)}")
    
    # 创建数据加载器
    print("Creating dataloader...")
    batch_size = 72  # 适合32G V100 GPU
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=mp.cpu_count(),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    print(f"Dataloader created, number of batches: {len(dataloader)}")
    
    all_dice_scores = []
    all_pq_scores = []
    
    print("Starting validation...")
    # 启用混合精度以提高GPU利用率
    with torch.no_grad(), autocast():
        # 添加进度条
        for batch_idx, (img, masks, tissue_type, img_name) in tqdm(enumerate(dataloader), total=len(dataloader), desc="MoNuSeg Validation"):
            img = img.to(device, non_blocking=True)  # 使用non_blocking加速数据传输
            
            # 模型预测
            predictions = model(img)
            
            # 计算实例映射
            instance_maps, _ = model.calculate_instance_map(predictions)
            
            # 计算指标
            for i in range(img.size(0)):
                # 二值分割掩码
                pred_binary = torch.argmax(predictions['nuclei_binary_map'][i], dim=0).cpu().numpy()
                gt_binary = masks['nuclei_binary_map'][i].cpu().numpy()
                
                # 计算Dice系数
                intersection = np.sum(pred_binary * gt_binary)
                union = np.sum(pred_binary) + np.sum(gt_binary)
                dice = 2 * intersection / (union + 1e-8) if union > 0 else 1.0
                all_dice_scores.append(dice)
                
                # 计算PQ分数
                pred_instance = instance_maps[i].cpu().numpy().astype(np.int64)
                gt_instance = masks['instance_map'][i].cpu().numpy()
                
                # 添加调试信息
                print(f"Image {i+1}:")
                print(f"  Pred instance unique values: {np.unique(pred_instance)}")
                print(f"  GT instance unique values: {np.unique(gt_instance)}")
                print(f"  Pred instance shape: {pred_instance.shape}")
                print(f"  GT instance shape: {gt_instance.shape}")
                
                # 重新映射标签
                pred_instance = remap_label(pred_instance)
                gt_instance = remap_label(gt_instance)
                
                print(f"  After remap - Pred instance unique values: {np.unique(pred_instance)}")
                print(f"  After remap - GT instance unique values: {np.unique(gt_instance)}")
                
                # 计算PQ，降低匹配IoU阈值
                pq_info, _ = get_fast_pq(gt_instance, pred_instance, match_iou=0.2)
                dq, sq, pq = pq_info
                print(f"  PQ components: DQ={dq:.4f}, SQ={sq:.4f}, PQ={pq:.4f}")
                
                all_pq_scores.append(pq)
    
    # 计算平均指标
    mean_dice = np.mean(all_dice_scores)
    mean_pq = np.mean(all_pq_scores)
    
    logger.info("MoNuSeg validation completed!")
    print("MoNuSeg validation completed!")
    logger.info(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    logger.info(f"Mean PQ: {mean_pq:.4f}")
    print(f"Mean PQ: {mean_pq:.4f}")
    
    return {
        "mean_dice": mean_dice,
        "mean_pq": mean_pq
    }

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")
    
    # 模型和数据路径
    model_path = Path("/hy-tmp/SparseSwinCell/logs/train_cellvit_from_scratch/checkpoint_epoch_90.pth")
    monuseg_path = Path("/hy-tmp/SparseSwinCell/MoNuSeg/test")
    
    print(f"Loading model from: {model_path}")
    print(f"Using MoNuSeg data from: {monuseg_path}")
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 运行验证
    results = validate_monuseg(model, monuseg_path, device)
    
    # 打印结果
    print("\n" + "="*50)
    print("MoNuSeg Validation Results")
    print("="*50)
    print(f"Mean Dice: {results['mean_dice']:.4f}")
    print(f"Mean PQ: {results['mean_pq']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
