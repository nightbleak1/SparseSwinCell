# -*- coding: utf-8 -*-
# Evaluation script for MoNuSeg dataset using CellViT model

import os
import logging
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchmetrics.functional.classification import f1_score
from torchmetrics.functional.segmentation import dice as seg_dice

# 添加项目根目录到Python路径
import sys
sys.path.append("/hy-tmp/SparseSwinCell")

# 确保正确导入模型和数据集
from models.segmentation.cell_segmentation.cellvit import CellViT
from cell_segmentation.datasets.monuseg import MoNuSegDataset
from cell_segmentation.utils.metrics import get_fast_pq, remap_label

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluation_monuseg')

class MoNuSegEvaluator:
    """Cell segmentation evaluation class for MoNuSeg dataset"""
    
    def __init__(self, model_path, device='cuda'):
        """初始化评估器
        
        Args:
            model_path: 预训练模型权重路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path):
        """加载预训练模型"""
        logger.info(f"加载模型权重: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重文件不存在: {model_path}")
        
        # 初始化模型，使用RGB输入
        model = CellViT(num_classes=1, in_chans=3)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # 处理不同格式的检查点
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            logger.info("模型权重加载成功")
        except Exception as e:
            logger.error(f"模型权重加载失败: {str(e)}")
            raise
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate_dataset(self, dataset_path, visualize=False):
        """评估MoNuSeg数据集
        
        Args:
            dataset_path: MoNuSeg数据集路径
            visualize: 是否可视化评估结果
            
        Returns:
            results: 评估结果字典
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        # 加载MoNuSeg数据集
        dataset = MoNuSegDataset(
            dataset_path=dataset_path,
            transforms=None,
            patching=True,  # MoNuSeg需要patching
            overlap=64
        )
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"开始评估MoNuSeg数据集，共 {len(dataset)} 个样本")
        
        # 初始化指标
        all_dice_scores = []
        all_f1_scores = []
        all_pq_scores = []
        all_dq_scores = []  # Detection Quality
        all_sq_scores = []  # Segmentation Quality
        
        # 为每个样本存储详细结果
        detailed_results = []
        
        # 评估循环
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), desc="评估中", total=len(dataloader)):
                try:
                    # 将数据移至设备
                    img = batch['image'].to(self.device)
                    masks = batch['masks']
                    
                    # 进行预测
                    predictions = self.model(img)
                    
                    # 计算评估指标
                    # 二值分割掩码
                    pred_binary = torch.argmax(predictions['nuclei_binary_map'], dim=1)
                    gt_binary = masks['nuclei_binary_map'].to(self.device)
                    
                    # 计算Dice系数
                    dice_score = seg_dice(pred_binary, gt_binary, ignore_index=0).item()
                    
                    # 计算F1分数
                    f1 = f1_score(pred_binary, gt_binary, ignore_index=0).item()
                    
                    # 计算PQ分数
                    pred_instance = predictions['instance_map'][0].cpu().numpy()
                    gt_instance = masks['instance_map'].cpu().numpy()[0]
                    
                    # 重新映射标签以避免冲突
                    pred_instance = remap_label(pred_instance)
                    gt_instance = remap_label(gt_instance)
                    
                    # 计算PQ、DQ、SQ
                    pq_info, _ = get_fast_pq(gt_instance, pred_instance)
                    dq, sq, pq = pq_info
                    
                    # 保存指标
                    all_dice_scores.append(dice_score)
                    all_f1_scores.append(f1)
                    all_pq_scores.append(pq)
                    all_dq_scores.append(dq)
                    all_sq_scores.append(sq)
                    
                    # 保存详细结果
                    sample_result = {
                        'sample_index': i,
                        'dice': dice_score,
                        'f1': f1,
                        'pq': pq,
                        'dq': dq,
                        'sq': sq
                    }
                    detailed_results.append(sample_result)
                    
                except Exception as e:
                    logger.error(f"评估样本 {i} 失败: {str(e)}")
                    continue
        
        # 计算平均指标
        results = {
            'mean_dice': np.mean(all_dice_scores),
            'mean_f1': np.mean(all_f1_scores),
            'mean_pq': np.mean(all_pq_scores),
            'mean_dq': np.mean(all_dq_scores),
            'mean_sq': np.mean(all_sq_scores),
            'detailed_results': detailed_results
        }
        
        # 打印评估结果
        logger.info(f"\n评估结果:")
        logger.info(f"平均Dice系数: {results['mean_dice']:.4f}")
        logger.info(f"平均F1分数: {results['mean_f1']:.4f}")
        logger.info(f"平均PQ分数: {results['mean_pq']:.4f}")
        logger.info(f"平均DQ分数: {results['mean_dq']:.4f}")
        logger.info(f"平均SQ分数: {results['mean_sq']:.4f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='评估CellViT模型在MoNuSeg数据集上的性能')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型权重路径')
    parser.add_argument('--dataset_path', type=str, required=True, help='MoNuSeg数据集路径')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备 (cuda或cpu)')
    parser.add_argument('--visualize', action='store_true', help='是否可视化评估结果')
    
    args = parser.parse_args()
    
    try:
        evaluator = MoNuSegEvaluator(args.model_path, args.device)
        results = evaluator.evaluate_dataset(args.dataset_path, args.visualize)
        
        # 保存结果到文件
        results_df = pd.DataFrame(results['detailed_results'])
        results_df.to_csv('monuseg_evaluation_results.csv', index=False)
        logger.info(f"评估结果已保存到 monuseg_evaluation_results.csv")
        
    except Exception as e:
        logger.error(f"评估失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
