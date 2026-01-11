# -*- coding: utf-8 -*-
# Data preparation script for converting RGB images to H-DAB dual channel
# and organizing according to PanNuke dataset format

import os
import logging
from pathlib import Path
import argparse
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_preparation')

class DataPreparer:
    """Data preparation class for converting RGB images to H-DAB dual channel and organizing
    according to PanNuke dataset format"""
    
    @staticmethod
    def rgb_to_hed(rgb):
        """将RGB图像转换为H-DAB双通道 (Hematoxylin, DAB)"""
        # 将BGR转换为RGB（因为OpenCV读取的是BGR格式）
        if rgb.shape[2] == 3:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # 使用颜色解混矩阵转换到HED空间
        color_matrix = np.array([
            [0.65, 0.70, 0.29],
            [0.09, 0.99, 0.07],
            [0.27, 0.57, 0.78]
        ])
        hed = np.dot(rgb, np.linalg.inv(color_matrix).T)
        
        # 只返回H通道和DAB通道
        return hed[:, :, [0, 2]]  # H通道 (0), DAB通道 (2)
    
    def prepare_dataset(self, input_dir, output_dir, folds=5):
        """准备数据集，将RGB图像转换为H-DAB双通道并按照PanNuke格式组织
        保持输入目录的fold结构，不进行重新划分
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 创建输出目录结构
        output_path.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"开始准备数据集，输入目录: {input_dir}, 输出目录: {output_dir}")
        
        # 1. 收集所有fold目录 (以'fold'开头的目录)
        fold_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('fold')]
        fold_dirs.sort(key=lambda x: int(x.name.split('fold')[1]))  # 按fold0, fold1, fold2排序
        
        logger.info(f"找到 {len(fold_dirs)} 个fold目录: {', '.join([d.name for d in fold_dirs])}")
        
        if len(fold_dirs) == 0:
            logger.error("未找到fold目录，请确保输入目录包含fold0, fold1等子目录")
            return
        
        # 2. 创建输出目录结构 (保持相同的fold结构)
        for fold_dir in fold_dirs:
            fold_name = fold_dir.name
            output_fold_path = output_path / fold_name
            (output_fold_path / "Images").mkdir(exist_ok=True, parents=True)
            (output_fold_path / "Masks").mkdir(exist_ok=True, parents=True)
        
        # 3. 处理每个fold目录
        for fold_dir in fold_dirs:
            fold_name = fold_dir.name
            fold_input_path = input_path / fold_name
            fold_output_path = output_path / fold_name
            
            # 3.1 处理images目录
            images_dir = fold_input_path / "Images"
            if not images_dir.exists():
                logger.warning(f"未找到images目录: {images_dir}")
                continue
            
            # 3.2 处理Masks目录 (PanNuke数据集使用Masks目录)
            
            # 3.3 遍历所有图像文件
            image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(images_dir.glob(f'*{ext}')))
            
            image_files = [f for f in image_files if not f.name.startswith('.')]
            image_files = sorted(image_files)
            
            logger.info(f"处理fold {fold_name}: 找到 {len(image_files)} 个图像文件")
            
            for img_path in tqdm(image_files, desc=f"处理 {fold_name}"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"无法读取图像: {img_path}")
                        continue
                    
                    # 转换为H-DAB双通道
                    hed_img = self.rgb_to_hed(img)
                    
                    # 保存为PNG格式
                    output_img_path = fold_output_path / "Images" / img_path.name
                    # 归一化到0-255范围再保存
                    hed_img_normalized = cv2.normalize(
                        hed_img, None, 0, 255, cv2.NORM_MINMAX
                    ).astype(np.uint8)
                    # 为了保存为标准格式，将2通道转换为3通道
                    hed_img_3ch = np.zeros((hed_img.shape[0], hed_img.shape[1], 3), dtype=np.uint8)
                    hed_img_3ch[:, :, 0] = hed_img_normalized[:, :, 0]  # H通道到R
                    hed_img_3ch[:, :, 1] = hed_img_normalized[:, :, 1]  # DAB通道到G
                    hed_img_3ch[:, :, 2] = hed_img_normalized[:, :, 0]  # 复制H通道到B
                    
                    cv2.imwrite(str(output_img_path), hed_img_3ch)
                    
                    # 处理对应的标签文件 (使用Masks目录)
                    masks_dir = fold_input_path / "Masks"
                    if masks_dir.exists():
                        # PanNuke数据集使用masks.npy文件存储所有掩码
                        masks_file = masks_dir / "masks.npy"
                        if masks_file.exists():
                            output_masks_dir = fold_output_path / "Masks"
                            output_masks_file = output_masks_dir / "masks.npy"
                            
                            try:
                                # 对于PanNuke数据集，直接处理并保存masks.npy文件
                                # 这个操作只需要为每个fold执行一次，不需要为每个图像重复执行
                                # 这里保持代码结构，确保在处理第一个图像时处理mask文件
                                if not output_masks_file.exists():
                                    mask = np.load(str(masks_file))
                                    
                                    # 根据H-DAB数据特点处理mask
                                    if len(mask.shape) == 4:  # PanNuke格式通常是 [num_samples, channels, height, width]
                                        # 对于H-DAB数据，确保使用正确的分割通道
                                        # 保留第一个通道作为细胞核分割
                                        if mask.shape[1] > 0:
                                            processed_mask = mask[:, 0:1]  # 保持单通道维度
                                        else:
                                            processed_mask = mask
                                    else:
                                        processed_mask = mask
                                    
                                    # 保存处理后的mask
                                    np.save(str(output_masks_file), processed_mask)
                                    logger.info(f"已处理并保存 {output_masks_file}")
                            except Exception as e:
                                logger.error(f"处理掩码文件时出错: {masks_file}, 错误: {str(e)}")
                                # 出错时复制原始文件
                                import shutil
                                shutil.copy(str(masks_file), str(output_masks_file))
                        else:
                            logger.warning(f"找不到掩码文件: {masks_file}")
                
                except Exception as e:
                    logger.error(f"处理图像时出错: {img_path}, 错误: {str(e)}")
            
            # 4. 创建types.csv (每个fold单独创建)
            image_names = [f.stem for f in image_files]
            df = pd.DataFrame({
                'image_id': image_names,
                'type': [0] * len(image_names)  # 默认乳腺组织类型0
            })
            types_csv_path = fold_output_path / "types.csv"
            df.to_csv(types_csv_path, index=False)
            logger.info(f"已创建 {types_csv_path}")
        
        logger.info("数据集准备完成! (已按输入fold结构组织)")

def main():
    parser = argparse.ArgumentParser(description='数据准备工具：将RGB图像转换为H-DAB双通道并按照PanNuke格式组织')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录，包含fold0, fold1等子目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录，将保持fold结构')
    parser.add_argument('--folds', type=int, default=5, help='输入目录中的fold数量 (用于验证，实际处理所有fold)')
    
    args = parser.parse_args()
    
    preparer = DataPreparer()
    preparer.prepare_dataset(
        args.input_dir,
        args.output_dir,
        folds=args.folds
    )

if __name__ == "__main__":
    main()