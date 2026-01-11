#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 添加项目根目录到Python路径
sys.path.append("/hy-tmp/SparseSwinCell")

# 导入模型和工具
from models.segmentation.cell_segmentation.cellvit import CellViT
from cell_segmentation.datasets.pannuke import PanNukeDataset
from cell_segmentation.utils.metrics import get_fast_pq, remap_label
import torch.multiprocessing as mp

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validate_pannuke_folds')

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

def validate_fold(model, fold, pannuke_path, device):
    """验证单个fold的数据"""
    logger.info(f"Validating fold {fold}...")
    
    # 创建数据集
    print(f"Creating dataset for fold {fold}...")
    dataset = PanNukeDataset(
        dataset_path=pannuke_path,
        folds=fold,
        transforms=None,
        regression=True
    )
    print(f"Dataset created for fold {fold}, size: {len(dataset)}")
    
    # 创建数据加载器
    print(f"Creating dataloader for fold {fold}...")
    # 增大批处理大小以提高GPU利用率
    batch_size = 72  # 对于32G V100 GPU，批处理大小可以增大到72
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=mp.cpu_count(),
        pin_memory=True,
        prefetch_factor=2,  # 增加预取因子，提前加载更多数据
        persistent_workers=True  # 保持worker持续运行，减少启动开销
    )
    print(f"Dataloader created for fold {fold}, number of batches: {len(dataloader)}, batch_size: {batch_size}")
    
    all_dice_scores = []
    all_pq_scores = []
    
    print(f"Starting validation for fold {fold}...")
    # 启用混合精度以提高GPU利用率
    with torch.no_grad(), autocast():
        # 添加进度条
        for batch_idx, (img, masks, tissue_type, img_name) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Fold {fold} Validation"):
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
                
                # 重新映射标签
                pred_instance = remap_label(pred_instance)
                gt_instance = remap_label(gt_instance)
                
                # 计算PQ
                pq_info, _ = get_fast_pq(gt_instance, pred_instance)
                _, _, pq = pq_info
                all_pq_scores.append(pq)
    
    # 计算平均指标
    mean_dice = np.mean(all_dice_scores)
    mean_pq = np.mean(all_pq_scores)
    
    logger.info(f"Fold {fold} validation completed!")
    print(f"Fold {fold} validation completed!")
    logger.info(f"Fold {fold} Mean Dice: {mean_dice:.4f}")
    print(f"Fold {fold} Mean Dice: {mean_dice:.4f}")
    logger.info(f"Fold {fold} Mean PQ: {mean_pq:.4f}")
    print(f"Fold {fold} Mean PQ: {mean_pq:.4f}")
    
    return {
        "fold": fold,
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
    pannuke_path = Path("/hy-tmp/SparseSwinCell/cell_segmentation/datasets/process/PanNuke")
    
    print(f"Loading model from: {model_path}")
    print(f"Using PanNuke data from: {pannuke_path}")
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 验证每个fold
    folds = [0, 1, 2]
    results = []
    
    # 添加fold验证进度条
    for fold in tqdm(folds, desc="Overall Validation Progress"):
        try:
            fold_result = validate_fold(model, fold, pannuke_path, device)
            results.append(fold_result)
        except Exception as e:
            print(f"Error validating fold {fold}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping fold {fold}...")
    
    if not results:
        print("Error: No folds were successfully validated!")
        return
    
    # 打印所有结果
    print("\n" + "="*50)
    print("PanNuke Fold Validation Results")
    print("="*50)
    for result in results:
        print(f"Fold {result['fold']}: Mean Dice = {result['mean_dice']:.4f}, Mean PQ = {result['mean_pq']:.4f}")
    
    # 计算平均结果
    avg_dice = np.mean([r['mean_dice'] for r in results])
    avg_pq = np.mean([r['mean_pq'] for r in results])
    print("="*50)
    print(f"Average: Mean Dice = {avg_dice:.4f}, Mean PQ = {avg_pq:.4f}")

if __name__ == "__main__":
    main()
