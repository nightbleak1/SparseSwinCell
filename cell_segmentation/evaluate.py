# -*- coding: utf-8 -*-
# Evaluation script for cell segmentation models

import os
import sys
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
sys.path.append(os.path.abspath('/hy-tmp/SparseSwinCell'))

# 确保正确导入模型和数据集
from models.segmentation.cell_segmentation.cellvit import CellViT, CellViT256
from cell_segmentation.datasets.pannuke import PanNukeDataset
from cell_segmentation.datasets.monuseg import MoNuSegDataset
from cell_segmentation.utils.metrics import get_fast_pq, remap_label

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluation')

class CellSegmentationEvaluator:
    """Cell segmentation evaluation class with support for H-DAB input and weak staining analysis"""
    
    def __init__(self, model_path, device='cuda', model_type='cellvit'):
        """初始化评估器
        
        Args:
            model_path: 预训练模型权重路径
            device: 运行设备 ('cuda' 或 'cpu')
            model_type: 模型类型，目前支持'cellvit'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {self.device}")
        self.model_type = model_type
        
        # 加载模型
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path):
        """加载预训练模型"""
        logger.info(f"加载模型权重: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重文件不存在: {model_path}")
        
        # 初始化CellViT模型，使用训练时相同的参数
        model = CellViT(
            num_nuclei_classes=6,  # 包括背景
            num_tissue_classes=19,
            embed_dim=96,  # 使用训练时的embed_dim
            input_channels=3,  # RGB输入
            depth=12,  # 总深度，会被按比例分配到4个阶段
            num_heads=3,  # 使用训练时的num_heads
            extract_layers=[3, 6, 9, 12],
            regression_loss=True
        )
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
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
    
    def _detect_weak_staining(self, img):
        """检测弱染色区域
        
        基于DAB通道的强度来判断是否为弱染色区域
        
        Args:
            img: 输入图像，shape为[B, C, H, W]，C=2（H-DAB通道）
            
        Returns:
            bool: 是否为弱染色区域
        """
        # DAB通道通常在索引1位置
        if img.shape[1] >= 2:
            dab_channel = img[:, 1, :, :]
            # 计算DAB通道的平均强度
            dab_mean = torch.mean(dab_channel).item()
            # 使用阈值判断弱染色（阈值可以根据实际数据调整）
            return dab_mean < 0.2
        else:
            # 如果通道不足，使用整体强度
            img_mean = torch.mean(img).item()
            return img_mean < 0.3
    
    def evaluate_dataset(self, dataset_path, folds=[0], visualize=False):
        """评估数据集
        
        Args:
            dataset_path: 数据集路径
            folds: 要评估的fold列表
            visualize: 是否可视化评估结果
            
        Returns:
            results: 评估结果字典
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        # 根据数据集路径自动选择数据集类型
        if 'MoNuSeg' in dataset_path:
            # 加载MoNuSeg数据集
            logger.info("加载MoNuSeg数据集...")
            dataset = MoNuSegDataset(
                dataset_path=dataset_path,
                transforms=None,
                patching=True,
                overlap=64
            )
        else:
            # 加载PanNuke数据集，确保使用RGB输入
            logger.info("加载PanNuke数据集...")
            dataset = PanNukeDataset(
                dataset_path=dataset_path,
                folds=folds,
                transforms=None,
                in_channels=3,
                mode='test'
            )
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"开始评估数据集，共 {len(dataset)} 个样本")
        
        # 初始化指标
        all_dice_scores = []
        all_f1_scores = []
        all_pq_scores = []
        all_dq_scores = []  # Detection Quality
        all_sq_scores = []  # Segmentation Quality
        weak_stain_scores = []
        normal_stain_scores = []
        
        # 为每个样本存储详细结果
        detailed_results = []
        
        # 评估循环
        for i, (img, masks, tissue_type, img_name) in tqdm(enumerate(dataloader), desc="评估中", total=len(dataloader)):
            try:
                # 将数据移至设备
                img = img.to(self.device)
                
                # 进行预测
                with torch.no_grad():
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
                
                # 检测弱染色区域
                is_weak_stain = self._detect_weak_staining(img)
                
                # 保存详细结果
                sample_result = {
                    'image_name': img_name[0] if isinstance(img_name, list) else img_name,
                    'tissue_type': tissue_type.item() if hasattr(tissue_type, 'item') else tissue_type,
                    'dice': dice_score,
                    'f1': f1,
                    'pq': pq,
                    'dq': dq,
                    'sq': sq,
                    'is_weak_stain': is_weak_stain
                }
                detailed_results.append(sample_result)
                
                # 根据染色强度分组保存指标
                if is_weak_stain:
                    weak_stain_scores.append({
                        'dice': dice_score,
                        'f1': f1,
                        'pq': pq,
                        'dq': dq,
                        'sq': sq
                    })
                else:
                    normal_stain_scores.append({
                        'dice': dice_score,
                        'f1': f1,
                        'pq': pq,
                        'dq': dq,
                        'sq': sq
                    })
                    
            except Exception as e:
                logger.error(f"评估样本时出错: {img_name}, 错误: {str(e)}")
                continue
        
        # 计算平均指标
        results = {
            'overall': {
                'dice': np.mean(all_dice_scores),
                'f1': np.mean(all_f1_scores),
                'pq': np.mean(all_pq_scores),
                'dq': np.mean(all_dq_scores),
                'sq': np.mean(all_sq_scores),
                'count': len(all_dice_scores)
            },
            'detailed_results': detailed_results
        }
        
        # 计算弱染色区域性能
        if weak_stain_scores:
            results['weak_stain'] = {
                'dice': np.mean([s['dice'] for s in weak_stain_scores]),
                'f1': np.mean([s['f1'] for s in weak_stain_scores]),
                'pq': np.mean([s['pq'] for s in weak_stain_scores]),
                'dq': np.mean([s['dq'] for s in weak_stain_scores]),
                'sq': np.mean([s['sq'] for s in weak_stain_scores]),
                'count': len(weak_stain_scores)
            }
        
        # 计算正常染色区域性能
        if normal_stain_scores:
            results['normal_stain'] = {
                'dice': np.mean([s['dice'] for s in normal_stain_scores]),
                'f1': np.mean([s['f1'] for s in normal_stain_scores]),
                'pq': np.mean([s['pq'] for s in normal_stain_scores]),
                'dq': np.mean([s['dq'] for s in normal_stain_scores]),
                'sq': np.mean([s['sq'] for s in normal_stain_scores]),
                'count': len(normal_stain_scores)
            }
        
        return results
    
    def save_results(self, results, output_path):
        """保存评估结果
        
        Args:
            results: 评估结果字典
            output_path: 输出文件路径
        """
        # 创建结果目录
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 转换结果为DataFrame
        summary_data = []
        for category, metrics in results.items():
            if category != 'detailed_results':
                row = {'category': category}
                row.update({k: v for k, v in metrics.items() if k != 'count'})
                row['count'] = metrics.get('count', 0)
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存汇总结果为CSV
        summary_csv_path = output_path
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"评估汇总结果已保存至: {summary_csv_path}")
        
        # 保存详细结果
        if 'detailed_results' in results:
            detailed_df = pd.DataFrame(results['detailed_results'])
            detailed_csv_path = str(output_path).replace('.csv', '_detailed.csv')
            detailed_df.to_csv(detailed_csv_path, index=False)
            logger.info(f"详细评估结果已保存至: {detailed_csv_path}")
        
        # 打印结果
        logger.info("评估结果汇总:")
        for category, metrics in results.items():
            if category != 'detailed_results':
                logger.info(f"{category}:")
                for metric, value in metrics.items():
                    if metric != 'count':
                        logger.info(f"  {metric}: {value:.4f}")
                logger.info(f"  样本数量: {metrics.get('count', 0)}")

def main():
    parser = argparse.ArgumentParser(description='细胞核分割模型评估工具')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型权重路径')
    parser.add_argument('--dataset_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--output_path', type=str, default='./evaluation_results.csv', help='评估结果输出路径')
    parser.add_argument('--folds', type=int, nargs='+', default=[0], help='要评估的fold列表')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='运行设备')
    parser.add_argument('--model_type', type=str, default='cellvit', choices=['cellvit'], help='模型类型')
    parser.add_argument('--visualize', action='store_true', help='是否可视化评估结果')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = CellSegmentationEvaluator(args.model_path, args.device, args.model_type)
    
    # 评估数据集
    results = evaluator.evaluate_dataset(args.dataset_path, args.folds, args.visualize)
    
    # 保存结果
    evaluator.save_results(results, args.output_path)

if __name__ == "__main__":
    main()