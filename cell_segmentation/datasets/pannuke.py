# -*- coding: utf-8 -*-
# PanNuke Dataset
#
# Dataset information: https://arxiv.org/abs/2003.10778
# Please Prepare Dataset as described here: docs/readmes/pannuke.md
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import logging
import sys  # remove
from pathlib import Path
from typing import Callable, Tuple, Union

sys.path.append("/homes/fhoerst/histo-projects/CellViT/")  # remove

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from numba import njit
from PIL import Image
from scipy.ndimage import center_of_mass, distance_transform_edt, gaussian_filter
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import entropy

from cell_segmentation.datasets.base_cell import CellDataset
from cell_segmentation.utils.tools import fix_duplicates, get_bounding_box

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())

from natsort import natsorted


class PanNukeDataset(CellDataset):
    """PanNuke dataset

    Args:
        dataset_path (Union[Path, str]): Path to PanNuke dataset. Structure is described under ./docs/readmes/cell_segmentation.md
        folds (Union[int, list[int]]): Folds to use for this dataset
        transforms (Callable, optional): PyTorch transformations. Defaults to None.
        stardist (bool, optional): Return StarDist labels. Defaults to False
        regression (bool, optional): Return Regression of cells in x and y direction. Defaults to False
        cache_dataset: If the dataset should be loaded to host memory in first epoch.
            Be careful, workers in DataLoader needs to be persistent to have speedup.
            Recommended to false, just use if you have enough RAM and your I/O operations might be limited.
            Defaults to False.
    """

    class StainNormalizer:
        """基于风格一致性的染色归一化方法
        
        利用局部相位增强技术和风格一致性熵度量模型，实现对染色差异的精细化调整
        """
        def __init__(self, alpha=0.5, beta=0.5):
            self.alpha = alpha  # 局部相位增强权重
            self.beta = beta  # 风格一致性调整权重
            self.target_stats = None  # 目标图像统计信息
        
        def local_phase_enhancement(self, image):
            """局部相位增强技术，用于校正细胞核与细胞质的色彩失真
            
            Args:
                image: 输入图像，形状为(H, W, 3)，支持RGB或BGR格式
            
            Returns:
                enhanced_image: 增强后的图像，保持与输入相同的格式
            """
            # 检查输入图像是否为BGR格式（通过比较像素值分布）
            # 这里我们假设如果绿色通道的平均值较高，则可能是BGR格式
            # 这是一种简单的判断方法，实际应用中可能需要更复杂的检测
            is_bgr = False
            if image.shape[2] == 3:
                b, g, r = cv2.split(image)
                if np.mean(g) > np.mean(r) and np.mean(g) > np.mean(b):
                    is_bgr = True
            
            # 转换为Lab颜色空间，便于相位处理
            if is_bgr:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            else:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            l, a, b = cv2.split(lab)
            
            # 对亮度通道应用局部相位增强
            # 傅里叶变换获取相位信息
            f = fft2(l)
            fshift = fftshift(f)
            magnitude = np.abs(fshift)
            phase = np.angle(fshift)
            
            # 相位增强：锐化相位信息
            enhanced_phase = phase * (1 + self.alpha * np.log(1 + magnitude / np.max(magnitude)))
            
            # 逆变换恢复图像
            fshift_enhanced = magnitude * np.exp(1j * enhanced_phase)
            f_enhanced = ifftshift(fshift_enhanced)
            l_enhanced = np.real(ifft2(f_enhanced))
            l_enhanced = cv2.normalize(l_enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # 合并通道并转换回原格式
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            if is_bgr:
                enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            else:
                enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            return enhanced_image
        
        def compute_style_stats(self, image):
            """计算图像的风格统计信息
            
            Args:
                image: 输入图像，形状为(H, W, 3)，支持RGB或BGR格式
            
            Returns:
                stats: 包含直方图熵和均值方差的统计字典
            """
            # 计算每个通道的直方图熵
            entropies = []
            means = []
            stds = []
            
            for channel in range(3):
                hist, _ = np.histogram(image[:, :, channel], bins=256, range=(0, 255))
                hist = hist / (image.shape[0] * image.shape[1])  # 归一化直方图
                entropies.append(entropy(hist + 1e-10))  # 添加平滑项避免log(0)
                means.append(np.mean(image[:, :, channel]))
                stds.append(np.std(image[:, :, channel]))
            
            return {
                'entropies': np.array(entropies),
                'means': np.array(means),
                'stds': np.array(stds)
            }
        
        def style_consistency_adjustment(self, image, target_stats):
            """基于风格一致性熵度量模型的自适应调整
            
            Args:
                image: RGB图像，形状为(H, W, 3)
                target_stats: 目标图像的风格统计信息
            
            Returns:
                adjusted_image: 调整后的RGB图像
            """
            # 计算当前图像的统计信息
            current_stats = self.compute_style_stats(image)
            
            # 计算统计差异
            entropy_diff = target_stats['entropies'] - current_stats['entropies']
            mean_diff = target_stats['means'] - current_stats['means']
            std_ratio = target_stats['stds'] / (current_stats['stds'] + 1e-10)
            
            # 对每个通道进行调整
            adjusted_image = image.copy().astype(np.float32)
            for channel in range(3):
                # 应用均值调整
                adjusted_channel = adjusted_image[:, :, channel] + mean_diff[channel]
                # 应用标准差调整
                adjusted_channel = (adjusted_channel - current_stats['means'][channel]) * std_ratio[channel] + target_stats['means'][channel]
                # 应用熵调整
                # 熵调整通过直方图匹配实现
                hist, bins = np.histogram(adjusted_channel, bins=256, range=(0, 255))
                cdf = hist.cumsum()
                # 避免除零警告
                if cdf[-1] == 0:
                    # 如果CDF总和为0，说明该通道所有像素值相同，无需调整
                    adjusted_channel = adjusted_channel
                else:
                    cdf_normalized = cdf / cdf[-1]  # 归一化CDF
                    
                    # 生成目标CDF（基于熵差异）
                    target_hist = np.ones(256) * (1 / 256)  # 均匀分布
                    target_cdf = np.cumsum(target_hist)
                    
                    # 直方图匹配
                    adjusted_channel = np.interp(adjusted_channel.flatten(), bins[:-1], cdf_normalized)
                    adjusted_channel = np.interp(adjusted_channel, target_cdf, bins[:-1])
                    adjusted_channel = adjusted_channel.reshape(image.shape[:2])
                
                adjusted_image[:, :, channel] = adjusted_channel
            
            # 移除NaN和无穷大值，并限制在合理范围内
            adjusted_image = np.nan_to_num(adjusted_image, nan=0.0, posinf=255.0, neginf=0.0)
            # 归一化到0-255范围
            adjusted_image = cv2.normalize(adjusted_image, None, 0, 255, cv2.NORM_MINMAX)
            # 确保所有值都在0-255范围内，然后转换为uint8
            adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
            
            return adjusted_image
        
        def fit(self, target_image):
            """拟合目标图像的风格
            
            Args:
                target_image: 目标风格图像
            """
            self.target_stats = self.compute_style_stats(target_image)
        
        def transform(self, image):
            """对图像进行染色归一化转换
            
            Args:
                image: 输入图像，支持RGB或BGR格式
            
            Returns:
                normalized_image: 归一化后的图像，保持与输入相同的格式
            """
            # 1. 应用局部相位增强
            enhanced_image = self.local_phase_enhancement(image)
            
            # 2. 应用风格一致性调整
            if self.target_stats is None:
                # 如果没有目标风格，使用当前图像作为目标
                self.fit(enhanced_image)
            
            normalized_image = self.style_consistency_adjustment(enhanced_image, self.target_stats)
            
            # 3. 融合原始图像和归一化图像
            normalized_image = cv2.addWeighted(enhanced_image, 1 - self.beta, normalized_image, self.beta, 0)
            
            return normalized_image

    def __init__(
        self,
        dataset_path: Union[Path, str],
        folds: Union[int, list[int]],
        transforms: Callable = None,
        stardist: bool = False,
        regression: bool = False,
        cache_dataset: bool = False,
    ) -> None:
        if isinstance(folds, int):
            folds = [folds]

        self.dataset = Path(dataset_path).resolve()
        self.transforms = transforms
        self.images = []
        self.masks = []
        self.types = {}
        self.cell_counts = {}
        self.img_names = []
        self.folds = folds
        
        # 加载配置文件
        self.config = {}
        self.weights = {}
        
        # 加载dataset_config.yaml
        dataset_config_path = self.dataset / "dataset_config.yaml"
        if dataset_config_path.exists():
            with open(dataset_config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded dataset config from {dataset_config_path}")
        
        # 加载weight_config.yaml
        weight_config_path = self.dataset / "weight_config.yaml"
        if weight_config_path.exists():
            with open(weight_config_path, "r") as f:
                self.weights = yaml.safe_load(f)
            logger.info(f"Loaded weight config from {weight_config_path}")
        self.cache_dataset = cache_dataset
        self.stardist = stardist
        self.regression = regression
        
        # 初始化染色归一化器
        self.stain_normalizer = self.StainNormalizer(alpha=0.5, beta=0.5)
        
        # 加载numpy数组数据（PanNuke原格式）
        self.npy_data = {}
        self.is_npy_format = False
        
        for fold in folds:
            # 检查是否存在numpy数组格式的数据集
            images_npy_path = self.dataset / f"fold{fold}" / "Images" / "images.npy"
            masks_npy_path = self.dataset / f"fold{fold}" / "Masks" / "masks.npy"
            types_npy_path = self.dataset / f"fold{fold}" / "Images" / "types.npy"
            
            if images_npy_path.exists() and masks_npy_path.exists() and types_npy_path.exists():
                self.is_npy_format = True
                # 加载numpy数组到内存
                if fold not in self.npy_data:
                    self.npy_data[fold] = {
                        'images': np.load(images_npy_path),
                        'masks': np.load(masks_npy_path, allow_pickle=True),
                        'types': np.load(types_npy_path)
                    }
                
                # 获取当前fold的图像数量
                num_images = len(self.npy_data[fold]['images'])
                for i in range(num_images):
                    # 存储fold和索引信息，用于后续加载
                    self.images.append((fold, i))
                    self.masks.append((fold, i))
                    self.img_names.append(f"fold{fold}_img{i}.png")
                    # 类型信息直接从types.npy获取
                    self.types[f"fold{fold}_img{i}.png"] = self.npy_data[fold]['types'][i]
                
                logger.debug(f"Loaded fold {fold} with {num_images} images from numpy arrays")
            else:
                # 尝试原始的单文件格式
                image_path = self.dataset / f"fold{fold}" / "images"
                fold_images = [
                    f for f in natsorted(image_path.glob("*.png")) if f.is_file()
                ]

                # sanity_check: mask must exist for image
                for fold_image in fold_images:
                    mask_path = (
                        self.dataset / f"fold{fold}" / "labels" / f"{fold_image.stem}.npy"
                    )
                    if mask_path.is_file():
                        self.images.append(fold_image)
                        self.masks.append(mask_path)
                        self.img_names.append(fold_image.name)

                    else:
                        logger.debug(
                            f"Found image {fold_image}, but no corresponding annotation file!"
                        )
                
                # 尝试加载types.csv
                types_csv_path = self.dataset / f"fold{fold}" / "types.csv"
                if types_csv_path.exists():
                    fold_types = pd.read_csv(types_csv_path)
                    # 检查列名，如果是image_id则使用，否则使用img
                    if "image_id" in fold_types.columns:
                        fold_type_dict = fold_types.set_index("image_id")["type"].to_dict()
                    elif "img" in fold_types.columns:
                        fold_type_dict = fold_types.set_index("img")["type"].to_dict()
                    elif "Image" in fold_types.columns:
                        fold_type_dict = fold_types.set_index("Image")["type"].to_dict()
                    else:
                        logger.warning(f"types.csv in fold{fold} has no valid image ID column. Using default tissue type.")
                        fold_type_dict = {}
                    
                    # 将字符串组织类型转换为整数ID
                    if "tissue_types" in self.config:
                        tissue_map = self.config["tissue_types"]
                        mapped_fold_type_dict = {}
                        for img_id, tissue_str in fold_type_dict.items():
                            # 将组织字符串转换为整数ID
                            if tissue_str in tissue_map:
                                mapped_fold_type_dict[img_id] = tissue_map[tissue_str]
                            else:
                                # 如果在映射中找不到，使用默认值0
                                mapped_fold_type_dict[img_id] = 0
                                logger.warning(f"Unknown tissue type '{tissue_str}' for image {img_id}, using default 0")
                        fold_type_dict = mapped_fold_type_dict
                    else:
                        logger.warning("No tissue_types in config, using string tissue types")
                    
                    self.types = {
                        **self.types,
                        **fold_type_dict,
                    }  # careful - should all be named differently
                
                # 加载cell_count.csv
                cell_count_csv_path = self.dataset / f"fold{fold}" / "cell_count.csv"
                if cell_count_csv_path.exists():
                    fold_cell_counts = pd.read_csv(cell_count_csv_path)
                    # 检查列名，如果是image_id则使用，否则使用img
                    if "image_id" in fold_cell_counts.columns:
                        fold_cell_count_dict = fold_cell_counts.set_index("image_id").to_dict('index')
                    elif "img" in fold_cell_counts.columns:
                        fold_cell_count_dict = fold_cell_counts.set_index("img").to_dict('index')
                    elif "Image" in fold_cell_counts.columns:
                        fold_cell_count_dict = fold_cell_counts.set_index("Image").to_dict('index')
                    else:
                        logger.warning(f"cell_count.csv in fold{fold} has no valid image ID column. Not using cell counts.")
                        fold_cell_count_dict = {}
                    self.cell_counts = {
                        **self.cell_counts,
                        **fold_cell_count_dict,
                    }

        logger.info(f"Created Pannuke Dataset by using fold(s) {self.folds}")
        logger.info(f"Resulting dataset length: {self.__len__()}")

        if self.cache_dataset:
            self.cached_idx = []  # list of idx that should be cached
            self.cached_imgs = {}  # keys: idx, values: numpy array of imgs
            self.cached_masks = {}  # keys: idx, values: numpy array of masks
            logger.info("Using cached dataset. Cache is built up during first epoch.")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, str, str]:
        """Get one dataset item consisting of transformed image,
        masks (instance_map, nuclei_type_map, nuclei_binary_map, hv_map) and tissue type as string

        Args:
            index (int): Index of element to retrieve

        Returns:
            Tuple[torch.Tensor, dict, str, str]:
                torch.Tensor: Image, with shape (3, H, W), in this case (3, 256, 256) - HED双通道图像
                dict:
                    "instance_map": Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (256, 256)
                    "nuclei_type_map": Nuclei-Type-Map, for each nucleus (instance) the class is indicated by an integer. Shape (256, 256)
                    "nuclei_binary_map": Binary Nuclei-Mask, Shape (256, 256)
                    "hv_map": Horizontal and vertical instance map.
                        Shape: (2 , H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
                        last is vertical (vertical gradient (-1 to 1)) Shape (2, 256, 256)
                    [Optional if stardist]
                    "dist_map": Probability distance map. Shape (256, 256)
                    "stardist_map": Stardist vector map. Shape (n_rays, 256, 256)
                    [Optional if regression]
                    "regression_map": Regression map. Shape (2, 256, 256). First is vertical, second horizontal.
                str: Tissue type
                str: Image Name
        """
        img_info = self.images[index]

        if self.cache_dataset:
            if index in self.cached_idx:
                img = self.cached_imgs[index]
                mask = self.cached_masks[index]
            else:
                # cache file
                img = self.load_imgfile(index)
                mask = self.load_maskfile(index)
                self.cached_imgs[index] = img
                self.cached_masks[index] = mask
                self.cached_idx.append(index)

        else:
            img = self.load_imgfile(index)
            mask = self.load_maskfile(index)

        # 应用染色归一化（保持原格式）
        img = self.stain_normalizer.transform(img)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        # 获取组织类型
        img_name = self.img_names[index]
        try:
            tissue_type = self.types[img_name]
        except KeyError:
            # 如果失败，尝试去掉.png扩展名
            img_name_no_ext = img_name.split('.')[0]
            try:
                tissue_type = self.types[img_name_no_ext]
            except KeyError:
                # 如果仍然失败，使用默认值
                tissue_type = 0  # 默认组织类型
        # 处理mask，确保是numpy数组
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        
        # 提取各通道数据
        inst_map = mask_np[:, :, 0].copy()
        type_map = mask_np[:, :, 1].copy()
        np_map = mask_np[:, :, 0].copy()
        np_map[np_map > 0] = 1
        hv_map = PanNukeDataset.gen_instance_hv_map(inst_map)

        # torch convert - 检查是否已经是Tensor
        if isinstance(img, torch.Tensor):
            # 已经是Tensor，只需要确保通道顺序正确
            if img.dim() == 3 and img.shape[0] == 1:
                # 单通道图像，转换为双通道
                img = img.repeat(2, 1, 1)
        else:
            # 是numpy数组，转换为Tensor
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            if torch.max(img) >= 5:
                img = img / 255

        masks = {
            "instance_map": torch.Tensor(inst_map).type(torch.int64),
            "nuclei_type_map": torch.Tensor(type_map).type(torch.int64),
            "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
            "hv_map": torch.Tensor(hv_map).type(torch.float32),
        }

        # load stardist transforms if neccessary
        if self.stardist:
            dist_map = PanNukeDataset.gen_distance_prob_maps(inst_map)
            stardist_map = PanNukeDataset.gen_stardist_maps(inst_map)
            masks["dist_map"] = torch.Tensor(dist_map).type(torch.float32)
            masks["stardist_map"] = torch.Tensor(stardist_map).type(torch.float32)
        if self.regression:
            masks["regression_map"] = PanNukeDataset.gen_regression_map(inst_map)

        return img, masks, tissue_type, self.img_names[index]

    def __len__(self) -> int:
        """Length of Dataset

        Returns:
            int: Length of Dataset
        """
        return len(self.images)

    def set_transforms(self, transforms: Callable) -> None:
        """Set the transformations, can be used tp exchange transformations

        Args:
            transforms (Callable): PyTorch transformations
        """
        self.transforms = transforms

    def load_imgfile(self, index: int) -> np.ndarray:
        """Load image from file (disk)

        Args:
            index (int): Index of file

        Returns:
            np.ndarray: Image as array with shape (H, W, 3)
        """
        img_info = self.images[index]
        
        if isinstance(img_info, tuple):
            # numpy数组格式
            fold, img_idx = img_info
            img = self.npy_data[fold]['images'][img_idx]
            # PanNuke的numpy数组可能是float64格式，需要先转换为uint8
            if img.dtype == np.float64:
                # 假设图像值在0-1范围内，转换为0-255的uint8
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                # 其他格式直接转换为uint8
                img = img.astype(np.uint8)
            # 保持原始RGB格式，不进行转换
            return img
        else:
            # 原始文件格式
            img_path = img_info
            # 使用OpenCV读取图像，返回BGR格式
            img = cv2.imread(str(img_path))
            return img.astype(np.uint8)

    def load_maskfile(self, index: int) -> np.ndarray:
        """Load mask from file (disk)

        Args:
            index (int): Index of file

        Returns:
            np.ndarray: Mask as array with shape (H, W, 2)
        """
        mask_info = self.masks[index]
        
        if isinstance(mask_info, tuple):
            # numpy数组格式
            fold, mask_idx = mask_info
            mask = self.npy_data[fold]['masks'][mask_idx]
            
            # PanNuke数据集的masks.npy是3D数组，每个通道代表不同信息
            # 通道0: 实例图，通道1: 类型图
            try:
                # 直接从数组通道中提取实例图和类型图
                inst_map = mask[:, :, 0].astype(np.int32)
                type_map = mask[:, :, 1].astype(np.int32)
            except Exception as e:
                logger.error(f"Error loading mask: {e}")
                logger.error(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
                # 返回空掩码作为 fallback
                return np.zeros((256, 256, 2), dtype=np.int32)
            
            mask = np.stack([inst_map, type_map], axis=-1)
            return mask
        else:
            # 原始文件格式
            mask_path = mask_info
            mask = np.load(mask_path, allow_pickle=True)
            inst_map = mask[()]["inst_map"].astype(np.int32)
            type_map = mask[()]["type_map"].astype(np.int32)
            mask = np.stack([inst_map, type_map], axis=-1)
            return mask

    def load_cell_count(self):
        """Load Cell count from cell_count.csv file. File must be located inside the fold folder
        and named "cell_count.csv"

        Example file beginning:
            Image,Neoplastic,Inflammatory,Connective,Dead,Epithelial
            0_0.png,4,2,2,0,0
            0_1.png,8,1,1,0,0
            0_10.png,17,0,1,0,0
            0_100.png,10,0,11,0,0
            ...
        """
        df_placeholder = []
        for fold in self.folds:
            csv_path = self.dataset / f"fold{fold}" / "cell_count.csv"
            cell_count = pd.read_csv(csv_path, index_col=0)
            df_placeholder.append(cell_count)
        self.cell_count = pd.concat(df_placeholder)
        self.cell_count = self.cell_count.reindex(self.img_names)

    def get_sampling_weights_tissue(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights calculated by tissue type statistics

        For this, a file named "weight_config.yaml" with the content:
            tissue:
                tissue_1: xxx
                tissue_2: xxx (name of tissue: count)
                ...
        Must exists in the dataset main folder (parent path, not inside the folds)

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        with open(
            (self.dataset / "weight_config.yaml").resolve(), "r"
        ) as run_config_file:
            yaml_config = yaml.safe_load(run_config_file)
            tissue_counts = dict(yaml_config)["tissue"]

        # calculate weight for each tissue
        weights_dict = {}
        k = np.sum(list(tissue_counts.values()))
        for tissue, count in tissue_counts.items():
            w = k / (gamma * count + (1 - gamma) * k)
            weights_dict[tissue] = w

        weights = []
        for idx in range(self.__len__()):
            img_idx = self.img_names[idx]
            type_str = self.types[img_idx]
            weights.append(weights_dict[type_str])

        return torch.Tensor(weights)

    def get_sampling_weights_cell(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights calculated by cell type statistics

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        assert hasattr(self, "cell_count"), "Please run .load_cell_count() in advance!"
        binary_weight_factors = np.array([4191, 4132, 6140, 232, 1528])
        k = np.sum(binary_weight_factors)
        cell_counts_imgs = np.clip(self.cell_count.to_numpy(), 0, 1)
        weight_vector = k / (gamma * binary_weight_factors + (1 - gamma) * k)
        img_weight = (1 - gamma) * np.max(cell_counts_imgs, axis=-1) + gamma * np.sum(
            cell_counts_imgs * weight_vector, axis=-1
        )
        img_weight[np.where(img_weight == 0)] = np.min(
            img_weight[np.nonzero(img_weight)]
        )

        return torch.Tensor(img_weight)

    def get_sampling_weights_cell_tissue(self, gamma: float = 1) -> torch.Tensor:
        """Get combined sampling weights by calculating tissue and cell sampling weights,
        normalizing them and adding them up to yield one score.

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        tw = self.get_sampling_weights_tissue(gamma)
        cw = self.get_sampling_weights_cell(gamma)
        weights = tw / torch.max(tw) + cw / torch.max(cw)

        return weights

    @staticmethod
    def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
        """Obtain the horizontal and vertical distance maps for each
        nuclear instance.

        Args:
            inst_map (np.ndarray): Instance map with each instance labelled as a unique integer
                Shape: (H, W)
        Returns:
            np.ndarray: Horizontal and vertical instance map.
                Shape: (2, H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
                last is vertical (vertical gradient (-1 to 1))
        """
        orig_inst_map = inst_map.copy()  # instance ID map

        x_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(orig_inst_map))
        if 0 in inst_list:
            inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(orig_inst_map == inst_id, np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            if inst_box[0] >= 2:
                inst_box[0] -= 2
            if inst_box[2] >= 2:
                inst_box[2] -= 2
            if inst_box[1] <= orig_inst_map.shape[0] - 2:
                inst_box[1] += 2
            if inst_box[3] <= orig_inst_map.shape[0] - 2:
                inst_box[3] += 2

            # improvement
            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        hv_map = np.stack([x_map, y_map])
        return hv_map

    @staticmethod
    def gen_distance_prob_maps(inst_map: np.ndarray) -> np.ndarray:
        """Generate distance probability maps

        Args:
            inst_map (np.ndarray): Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (H, W)

        Returns:
            np.ndarray: Distance probability map, shape (H, W)
        """
        inst_map = fix_duplicates(inst_map)
        dist = np.zeros_like(inst_map, dtype=np.float64)
        inst_list = list(np.unique(inst_map))
        if 0 in inst_list:
            inst_list.remove(0)

        for inst_id in inst_list:
            inst = np.array(inst_map == inst_id, np.uint8)

            y1, y2, x1, x2 = get_bounding_box(inst)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2

            inst = inst[y1:y2, x1:x2]

            if inst.shape[0] < 2 or inst.shape[1] < 2:
                continue

            # chessboard distance map generation
            # normalize distance to 0-1
            inst_dist = distance_transform_edt(inst)
            inst_dist = inst_dist.astype("float64")

            max_value = np.amax(inst_dist)
            if max_value <= 0:
                continue
            inst_dist = inst_dist / (np.max(inst_dist) + 1e-10)

            dist_map_box = dist[y1:y2, x1:x2]
            dist_map_box[inst > 0] = inst_dist[inst > 0]

        return dist

    @staticmethod
    @njit
    def gen_stardist_maps(inst_map: np.ndarray) -> np.ndarray:
        """Generate StarDist map with 32 nrays

        Args:
            inst_map (np.ndarray): Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (H, W)

        Returns:
            np.ndarray: Stardist vector map, shape (n_rays, H, W)
        """
        n_rays = 32
        # inst_map = fix_duplicates(inst_map)
        dist = np.empty(inst_map.shape + (n_rays,), np.float32)

        st_rays = np.float32((2 * np.pi) / n_rays)
        for i in range(inst_map.shape[0]):
            for j in range(inst_map.shape[1]):
                value = inst_map[i, j]
                if value == 0:
                    dist[i, j] = 0
                else:
                    for k in range(n_rays):
                        phi = np.float32(k * st_rays)
                        dy = np.cos(phi)
                        dx = np.sin(phi)
                        x, y = np.float32(0), np.float32(0)
                        while True:
                            x += dx
                            y += dy
                            ii = int(round(i + x))
                            jj = int(round(j + y))
                            if (
                                ii < 0
                                or ii >= inst_map.shape[0]
                                or jj < 0
                                or jj >= inst_map.shape[1]
                                or value != inst_map[ii, jj]
                            ):
                                # small correction as we overshoot the boundary
                                t_corr = 1 - 0.5 / max(np.abs(dx), np.abs(dy))
                                x -= t_corr * dx
                                y -= t_corr * dy
                                dst = np.sqrt(x**2 + y**2)
                                dist[i, j, k] = dst
                                break

        return dist.transpose(2, 0, 1)

    @staticmethod
    def gen_regression_map(inst_map: np.ndarray):
        n_directions = 2
        dist = np.zeros(inst_map.shape + (n_directions,), np.float32).transpose(2, 0, 1)
        inst_map = fix_duplicates(inst_map)
        inst_list = list(np.unique(inst_map))
        if 0 in inst_list:
            inst_list.remove(0)
        for inst_id in inst_list:
            inst = np.array(inst_map == inst_id, np.uint8)
            y1, y2, x1, x2 = get_bounding_box(inst)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2

            inst = inst[y1:y2, x1:x2]
            y_mass, x_mass = center_of_mass(inst)
            x_map = np.repeat(np.arange(1, x2 - x1 + 1)[None, :], y2 - y1, axis=0)
            y_map = np.repeat(np.arange(1, y2 - y1 + 1)[:, None], x2 - x1, axis=1)
            # we use a transposed coordinate system to align to HV-map, correct would be -1*x_dist_map and -1*y_dist_map
            x_dist_map = (x_map - x_mass) * np.clip(inst, 0, 1)
            y_dist_map = (y_map - y_mass) * np.clip(inst, 0, 1)
            dist[0, y1:y2, x1:x2] = x_dist_map
            dist[1, y1:y2, x1:x2] = y_dist_map

        return dist
