# -*- coding: utf-8 -*-
# Inference script for cell segmentation using pretrained model

import os
import logging
from pathlib import Path
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.segmentation.cell_segmentation.cellvit import CellViT
from cell_segmentation.datasets.pannuke import PanNukeDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('inference')

class CellSegmentationInference:
    """Cell segmentation inference class"""
    
    def __init__(self, model_path, device='cuda'):
        """初始化推理器
        
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
        
        # 初始化模型，支持RGB输入
        model = CellViT(num_classes=1, in_chans=3)
        
        # 加载权重
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            logger.info("模型权重加载成功")
        except Exception as e:
            logger.error(f"模型权重加载失败: {str(e)}")
            raise
        
        # 设置为评估模式
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path):
        """预处理输入图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            preprocessed_image: 预处理后的图像张量
        """
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为H-DAB双通道
        hed_img = PanNukeDataset.rgb_to_hed(img)  # [H, W, 2]
        
        # 转换为PyTorch张量
        img_tensor = torch.from_numpy(hed_img).permute(2, 0, 1).float()
        if torch.max(img_tensor) >= 5:
            img_tensor = img_tensor / 255
        
        # 添加批次维度
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def predict(self, image_path):
        """进行预测
        
        Args:
            image_path: 图像路径
            
        Returns:
            predictions: 预测结果字典
        """
        # 预处理图像
        img_tensor = self.preprocess_image(image_path)
        
        # 进行推理
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        return predictions
    
    def visualize_results(self, image_path, predictions, output_path):
        """可视化预测结果
        
        Args:
            image_path: 原始图像路径
            predictions: 预测结果
            output_path: 输出图像路径
        """
        # 读取原始图像
        original_img = cv2.imread(str(image_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 获取预测结果
        nuclei_binary_map = predictions['nuclei_binary_map'][0].cpu().numpy()
        nuclei_binary_mask = np.argmax(nuclei_binary_map, axis=0)
        
        # 创建可视化结果
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(original_img)
        plt.imshow(nuclei_binary_mask, alpha=0.5, cmap='jet')
        plt.title('细胞核分割结果')
        plt.axis('off')
        
        # 保存结果
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"预测结果已保存至: {output_path}")
    
    def batch_inference(self, input_dir, output_dir):
        """批量推理
        
        Args:
            input_dir: 输入图像目录
            output_dir: 输出结果目录
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 获取所有图像文件
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f'**/*{ext}')))
        
        image_files = [f for f in image_files if not f.name.startswith('.')]
        
        logger.info(f"找到 {len(image_files)} 个图像文件进行推理")
        
        # 批量推理
        for img_path in tqdm(image_files, desc="批量推理"):
            try:
                # 预测
                predictions = self.predict(img_path)
                
                # 可视化结果
                output_img_path = output_path / f"{img_path.stem}_pred.png"
                self.visualize_results(img_path, predictions, output_img_path)
                
                # 保存原始预测结果
                output_np_path = output_path / f"{img_path.stem}_pred.npz"
                np.savez_compressed(
                    output_np_path,
                    nuclei_binary_map=predictions['nuclei_binary_map'][0].cpu().numpy(),
                    hv_map=predictions['hv_map'][0].cpu().numpy(),
                    nuclei_type_map=predictions['nuclei_type_map'][0].cpu().numpy()
                )
                
            except Exception as e:
                logger.error(f"处理图像时出错: {img_path}, 错误: {str(e)}")
        
        logger.info("批量推理完成!")

def main():
    parser = argparse.ArgumentParser(description='细胞核分割推理工具')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型权重路径')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出结果目录')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='运行设备')
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = CellSegmentationInference(args.model_path, args.device)
    
    # 批量推理
    inference.batch_inference(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()