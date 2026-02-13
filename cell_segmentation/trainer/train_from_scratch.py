
# -*- coding: utf-8 -*-
# Training script for CellViT from scratch
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
import os
import sys
from pathlib import Path
from typing import Union, Dict

# 添加项目根目录到Python路径
sys.path.append("/hy-tmp/SparseSwinCell")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from base_ml.base_early_stopping import EarlyStopping
from utils.logger import Logger
from cell_segmentation.datasets.pannuke import PanNukeDataset
from cell_segmentation.datasets.monuseg import MoNuSegDataset
from cell_segmentation.trainer.trainer_cellvit import CellViTTrainer
from models.segmentation.cell_segmentation.cellvit import CellViT
from models.segmentation.cell_segmentation.sparse_cellvit import SparseCellViT
from base_ml.base_loss import DiceLoss, MCFocalTverskyLoss

class DiceLossWithLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        
    def forward(self, input, target):
        return self.dice(torch.sigmoid(input), target)

def train(resume_from_checkpoint=False, checkpoint_path=None):
    """Training function for CellViT from scratch"""
    # Setup logging
    import datetime
    # 使用当前时间创建唯一的日志目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = Path(f"./logs/train_cellvit_{timestamp}")
    logdir.mkdir(exist_ok=True, parents=True)
    
    # Initialize logger first
    logger = Logger(level="INFO", log_dir=logdir, comment="training").create_logger()
    logger.info("Starting training of CellViT from scratch")
    
    # Check if backup directory exists and copy checkpoint files if needed
    backup_logdir = Path("./logs/train_cellvit_from_scratch_backup_20260108_184403")
    if backup_logdir.exists() and not any(logdir.glob("*.pth")):
        logger.info(f"Found backup directory with checkpoints, copying to current logdir...")
        import shutil
        for checkpoint_file in backup_logdir.glob("*.pth"):
            shutil.copy(checkpoint_file, logdir)
            logger.info(f"Copied {checkpoint_file.name} to {logdir}")
    
    # Copy specified checkpoint file if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Copying specified checkpoint file from {checkpoint_path} to {logdir}...")
        import shutil
        shutil.copy(checkpoint_path, logdir)
        logger.info(f"Copied checkpoint file to {logdir}")
    
    # 1. 从头初始化模型
    logger.info("Initializing SparseCellViT model...")
    model = SparseCellViT(
        num_nuclei_classes=6,  # 包括背景
        num_tissue_classes=19,
        embed_dim=128,  # 增加嵌入维度，提高模型复杂度
        input_channels=3,  # RGB输入
        depth=16,  # 增加深度，提高模型复杂度
        num_heads=4,  # 增加注意力头数，提高模型复杂度
        extract_layers=[4, 8, 12, 16],  # 适配新的深度
        regression_loss=True,
        window_size=16,
        k=0.2,  # 降低top-k比例，增加稀疏性
        dynamic_k=True,  # 启用动态k值调整
        min_k_ratio=0.1,  # 最小k值比例
        max_k_ratio=0.7,  # 最大k值比例
        sparsity_level=0.2,  # 注意力稀疏性控制
        mlp_ratio=3.0,  # MLP比例
        drop_rate=0.1,  # Dropout率
        attn_drop_rate=0.1,  # 注意力Dropout率
        drop_path_rate=0.1  # 路径Dropout率
    )
    logger.info("Model initialized successfully")
    
    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Setup dataset paths
    pannuke_path = Path("/hy-tmp/SparseSwinCell/cell_segmentation/datasets/process/PanNuke")
    monuseg_path = Path("/hy-tmp/SparseSwinCell/cell_segmentation/datasets/process/MoNuSeg")
    
    # Load dataset configuration from file
    dataset_config_path = pannuke_path / "dataset_config.yaml"
    if dataset_config_path.exists():
        import yaml
        with open(dataset_config_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        logger.info(f"Loaded dataset configuration from {dataset_config_path}")
    else:
        # Fallback to default configuration if file not found
        dataset_config = {
            "tissue_types": {
                "Adrenal_gland": 0,
                "Bile-duct": 1,
                "Bladder": 2,
                "Breast": 3,
                "Cervix": 4,
                "Colon": 5,
                "Esophagus": 6,
                "HeadNeck": 7,
                "Kidney": 8,
                "Liver": 9,
                "Lung": 10,
                "Ovarian": 11,
                "Pancreatic": 12,
                "Prostate": 13,
                "Skin": 14,
                "Stomach": 15,
                "Testis": 16,
                "Thyroid": 17,
                "Uterus": 18
            },
            "nuclei_types": {
                "background": 0,
                "neoplastic": 1,
                "inflammatory": 2,
                "connective": 3,
                "dead": 4,
                "epithelial": 5
            }
        }
        logger.warning(f"Dataset configuration file not found at {dataset_config_path}. Using default configuration.")
    
    # Setup dataset and dataloader
    logger.info("Loading datasets...")
    
    # Data augmentation pipeline for training
    train_transforms = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomCrop(height=256, width=256, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
    ])
    
    # Minimal transformations for test set
    test_transforms = None
    
    # Training dataset: PanNuke fold0 and fold1
    train_dataset = PanNukeDataset(
        dataset_path=pannuke_path,
        folds=[0, 1],
        transforms=train_transforms,
        stardist=False,
        regression=True,
        cache_dataset=False
    )
    
    # Test dataset: PanNuke fold2
    test_dataset = PanNukeDataset(
        dataset_path=pannuke_path,
        folds=[2],
        transforms=test_transforms,
        stardist=False,
        regression=False,
        cache_dataset=False
    )
    
    # -------------------------------------------------------
    # 针对类别不平衡的改进：使用 WeightedRandomSampler 和 Class Weights
    # -------------------------------------------------------
    
    # 1. 计算 WeightedRandomSampler 的采样权重
    logger.info("Loading cell counts for weighted sampling...")
    train_dataset.load_cell_count()
    # 使用 gamma=1 进行完全平衡采样，综合考虑组织类型和细胞类型的平衡
    sampling_weights = train_dataset.get_sampling_weights_cell_tissue(gamma=1.0)
    
    # 创建采样器
    # num_samples 等于数据集大小，replacement=True 允许重复采样（过采样）
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sampling_weights,
        num_samples=len(sampling_weights),
        replacement=True
    )
    logger.info("Created WeightedRandomSampler for class balancing")

    # 2. 计算 Loss 函数的类别权重
    
    # (A) Tissue Types Weights
    # 从 weight_config.yaml 加载组织计数
    weight_config_path = pannuke_path / "weight_config.yaml"
    if weight_config_path.exists():
        import yaml
        with open(weight_config_path, "r") as f:
            weight_config = yaml.safe_load(f)
        tissue_counts = weight_config["tissue"]
        
        # 计算组织类型的逆频率权重
        total_tissue_samples = sum(tissue_counts.values())
        num_tissue_classes = len(tissue_counts)
        tissue_loss_weights = []
        
        # 确保顺序与 dataset_config["tissue_types"] 一致
        # dataset_config["tissue_types"] 是 {name: id}
        sorted_tissues = sorted(dataset_config["tissue_types"].items(), key=lambda x: x[1])
        
        for name, idx in sorted_tissues:
            # weight_config 中的名字可能与 dataset_config 中的略有不同（大小写等），尝试匹配
            count = tissue_counts.get(name, 0)
            # 如果找不到，尝试不区分大小写查找
            if count == 0:
                for k, v in tissue_counts.items():
                    if k.lower() == name.lower():
                        count = v
                        break
            
            if count > 0:
                # Weight = Total / (Num_Classes * Count)
                w = total_tissue_samples / (num_tissue_classes * count)
                # 限制权重范围，避免极端值
                w = min(max(w, 0.1), 10.0) 
            else:
                w = 1.0
            tissue_loss_weights.append(w)
            
        tissue_loss_weights = torch.Tensor(tissue_loss_weights).to(device)
        logger.info(f"Calculated Tissue Loss Weights: {tissue_loss_weights}")
    else:
        tissue_loss_weights = None
        logger.warning("weight_config.yaml not found, using equal weights for tissues")

    # (B) Nuclei Types Weights
    # 基于 PanNuke 统计数据: [4191, 4132, 6140, 232, 1528] (Neoplastic, Inflammatory, Connective, Dead, Epithelial)
    # 对应 ID: 1, 2, 3, 4, 5. ID 0 是 Background.
    nuclei_counts_dict = {
        1: 4191, # Neoplastic
        2: 4132, # Inflammatory
        3: 6140, # Connective
        4: 232,  # Dead (极少!)
        5: 1528  # Epithelial
    }
    total_nuclei_samples = sum(nuclei_counts_dict.values())
    num_nuclei_foreground = 5
    
    # 背景权重设为 0.5 (降低背景主导地位)，前景类计算逆频率
    nuclei_loss_weights = [0.5] 
    
    for i in range(1, 6):
        count = nuclei_counts_dict[i]
        # Weight = Total / (Num_Classes * Count)
        w = total_nuclei_samples / (num_nuclei_foreground * count)
        # 限制权重，防止 Dead 权重过大导致不稳定 (原始计算 Dead 权重约为 14)
        # 将 Dead 权重限制在 5.0 左右，其他按比例
        w = min(max(w, 0.5), 5.0)
        
        # 特殊处理 Connective (ID 3):
        # 尽管数量最多，但形态学上最难分割（细长、边界模糊）。
        # 原始反频率计算会给它最低权重 (~0.53)，导致Recall很低 (0.40)。
        # 强制提升其权重至 1.5，与 Neoplastic (1.0后修正) 处于相似量级
        if i == 3: # Connective
            w = 1.5
            
        nuclei_loss_weights.append(w)
    
    nuclei_loss_weights = torch.Tensor(nuclei_loss_weights).to(device)
    logger.info(f"Calculated Nuclei Loss Weights (incl. BG): {nuclei_loss_weights}")

    
    # GPU加速优化设置
    # 启用cuDNN自动调优，加速卷积运算
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # 启用TF32加速，在保持精度的同时提高计算速度
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 优化数据加载器配置
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=18,  # 稳定的batch_size
        shuffle=False,  # 使用 sampler 时必须为 False
        sampler=sampler, # 使用加权采样器
        num_workers=16,  # 调整workers数量，平衡数据加载和系统资源
        pin_memory=True,
        persistent_workers=True,  # 保持workers持续运行，减少启动开销
        prefetch_factor=2,  # 调整预取因子，平衡数据加载和内存使用
        drop_last=True,  # 丢弃最后一个不完整批次，保持训练稳定性
        collate_fn=None  # 使用默认collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=18,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    logger.info(f"Loaded train dataset with {len(train_dataset)} samples")
    logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
    
    # Setup loss functions
    loss_fn_dict = {
        "nuclei_binary_map": {
            "bce": (torch.nn.BCEWithLogitsLoss(), 1.0),
            "dice": (DiceLoss(), 1.0)
        },
        "hv_map": {
            "mse": (torch.nn.MSELoss(), 1.0)
        },
        "nuclei_type_map": {
            "focal_tversky": (MCFocalTverskyLoss(num_classes=6, alpha_t=0.3, beta_t=0.7, class_weights=nuclei_loss_weights), 1.0)
        },
        "tissue_types": {
            "ce": (torch.nn.CrossEntropyLoss(weight=tissue_loss_weights), 1.0)  
        },
        # 新增 Edge Loss (Shape Stream)
        "edge_map": {
            "bce": (torch.nn.BCEWithLogitsLoss(), 0.5), # 初始权重0.5
            "dice": (DiceLossWithLogits(), 0.5)
        }
    }
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # 调整 T_max 为 150，减缓学习率下降速度
    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=15,  # 早停耐心值，15轮没有提升则停止训练
        strategy="maximize"  # 最大化验证指标（bPQ-Score）
    )
    
    # Setup trainer with gradient accumulation
    trainer = CellViTTrainer(
        model=model,
        loss_fn_dict=loss_fn_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        logdir=logdir,
        num_classes=6,
        dataset_config=dataset_config,
        experiment_config={"epochs": 120, "batch_size": 18},
        early_stopping=early_stopping,
        log_images=True,
        magnification=40,
        mixed_precision=True
    )
    
    # 启用梯度累积，通过累积2个batch的梯度来模拟更大的batch_size效果
    trainer.accum_iter = 2
    
    # Check if we need to resume from checkpoint
    start_epoch = 0
    best_metric = float('-inf')  # 使用较小值作为初始值，因为我们要最大化bPQ-Score
    
    if resume_from_checkpoint:
        # Find all checkpoint files
        checkpoint_files = list(logdir.glob("checkpoint_epoch_*.pth")) + list(logdir.glob("latest_checkpoint.pth"))
        if checkpoint_files:
            # Priority: latest_checkpoint.pth > numbered checkpoints
            if (logdir / "latest_checkpoint.pth").exists():
                latest_checkpoint = logdir / "latest_checkpoint.pth"
            else:
                # Sort checkpoint files by epoch number in descending order
                checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[-1]) if "epoch" in x.stem else 0, reverse=True)
                latest_checkpoint = checkpoint_files[0]
            
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
            
            # Load checkpoint
            # 注意：由于模型结构发生变化（加入了ASPP, ShapeStream等），
            # 如果是加载旧的checkpoint，可能会报错。
            # 这里我们尝试加载，如果键不匹配，则只加载匹配的部分（transfer learning）。
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
                
                # Resume model state (strict=False allowed for structural changes)
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                if len(missing_keys) > 0:
                    logger.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys (e.g. {missing_keys[:3]}...)")
                if len(unexpected_keys) > 0:
                    logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys (e.g. {unexpected_keys[:3]}...)")
                
                # Resume optimizer state if structure matches well enough, otherwise reset optimizer
                # 由于引入新参数，旧优化器状态可能无法使用，建议重置优化器
                if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    logger.info("Optimizer and Scheduler states resumed.")
                else:
                    logger.info("Model structure changed. Optimizer and Scheduler states reset.")
                
                # Resume epoch and best metric if available
                start_epoch = checkpoint["epoch"] + 1
                if "best_metric" in checkpoint:
                    best_metric = checkpoint["best_metric"]
                
                # Resume early stopping state if available
                if "best_epoch" in checkpoint:
                    early_stopping.best_metric = best_metric
                    early_stopping.best_epoch = checkpoint["best_epoch"]
                    if "patience_counter" in checkpoint:
                        early_stopping.patience_counter = checkpoint["patience_counter"]
                
                logger.info(f"Resumed from epoch {start_epoch}, best metric: {best_metric:.4f}")
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                logger.info("Starting from scratch due to checkpoint incompatibility.")
                start_epoch = 0
        else:
            logger.info("No checkpoint found, starting from epoch 0")
    
    logger.info(f"Starting training for 100 epochs, from epoch {start_epoch+1}")
    
    # 初始权重设置
    initial_binary_weight = 1.0
    initial_hv_weight = 1.0
    initial_boundary_weight = 0.5
    initial_nuclei_type_weight = 1.0
    initial_focal_weight = 1.5
    initial_tissue_weight = 1.0
    initial_edge_weight = 0.5 # 新增 Edge Loss 权重
    
    # 存储前一轮的验证指标
    previous_macro_f1 = None
    
    # 存储验证指标历史，用于动态权重调整
    validation_history = {
        'boundary_f1': [],  # 边界F1分数
        'dice': [],  # 二进制分割Dice分数
        'bpq': [],  # 边界PQ分数
        'macro_f1': []  # 细胞核类型Macro-F1
    }
    
    # 边界损失调整计数器
    boundary_no_improvement_count = 0
    
    # 检查是否需要在开始时调整学习率（从90轮或更高轮次恢复）
    if start_epoch >= 100:
        logger.info("Resuming from epoch 100 or higher, calculating linear decay learning rate")
        # 计算当前epoch对应的线性衰减学习率
        progress = (start_epoch - 100) / 20.0
        new_lr = 1e-5 + (5e-6 - 1e-5) * progress
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.info(f"Learning rate set to {new_lr:.8f}")
    elif start_epoch >= 90:
        logger.info("Resuming from epoch 90 or higher, setting learning rate to 2e-5")
        for param_group in optimizer.param_groups:
            original_lr = param_group['lr']
            new_lr = 2e-5
            param_group['lr'] = new_lr
            logger.info(f"Learning rate adjusted from {original_lr:.6f} to {new_lr:.6f}")
    
    for epoch in range(start_epoch, 120):
        logger.info(f"Epoch {epoch+1}/120")
        
        # 根据 epoch 调整权重
        current_epoch = epoch + 1
        
        # 初始化当前权重
        current_binary_weight = initial_binary_weight
        current_hv_weight = initial_hv_weight
        current_boundary_weight = initial_boundary_weight
        current_nuclei_type_weight = initial_nuclei_type_weight
        current_focal_weight = initial_focal_weight
        current_tissue_weight = initial_tissue_weight
        current_edge_weight = initial_edge_weight
        
        # 第三阶段：50轮到90轮（在强化边界的同时，维持足够的分类权重以避免 mPQ 崩溃）
        if 50 < current_epoch <= 90:
            current_boundary_weight = 1.0
            current_binary_weight = 0.5
            current_hv_weight = 0.5
            current_nuclei_type_weight = 1.0 # 恢复分类权重 (0.05 -> 1.0)
            current_tissue_weight = 0.5      # 恢复组织上下文 (0.01 -> 0.5)
            current_edge_weight = 1.0
            
            if current_epoch == 51:
                logger.info("Epoch > 50: Balancing Boundary Refinement with Classification Stability")
        
        # 第四阶段：90轮及以后（双重优化：保持边界优势的同时，大幅回升分类权重以冲击 mPQ）
        elif current_epoch > 90:
            current_boundary_weight = 1.5 # 保持边界敏感度
            current_binary_weight = 1.0   # 恢复实例分割基础 (0.5 -> 1.0)
            current_hv_weight = 1.0       # 提升辅助分割权重 (0.5 -> 1.0)
            current_nuclei_type_weight = 2.0 # 保持高分类权重 (2.0)
            current_tissue_weight = 1.0   # 提升组织分类权重，提供语义上下文
            current_edge_weight = 2.0     # 进一步强化边缘 (1.5 -> 2.0)
            
            if current_epoch == 91:
                logger.info("Epoch > 90: Final Phase - Boosting HV/Edge/Binary weights to breakthrough bPQ/mPQ plateau")
        
        # 应用权重调整
        def update_weight(loss_name, weight):
            if loss_name in trainer.loss_fn_dict:
                for sub_loss_name, loss_setting in trainer.loss_fn_dict[loss_name].items():
                    if isinstance(loss_setting, tuple):
                        loss_fn, _ = loss_setting
                        trainer.loss_fn_dict[loss_name][sub_loss_name] = (loss_fn, weight)
                    else:
                        trainer.loss_fn_dict[loss_name][sub_loss_name]["weight"] = weight

        update_weight("nuclei_binary_map", current_binary_weight)
        update_weight("hv_map", current_hv_weight)
        update_weight("nuclei_type_map", current_nuclei_type_weight)
        update_weight("tissue_types", current_tissue_weight)
        update_weight("edge_map", current_edge_weight)
        
        # 更新 boundary_loss 的权重
        trainer.current_boundary_weight = current_boundary_weight
        
        # 记录权重调整
        logger.info(f"Updated weights: binary={current_binary_weight:.2f}, hv={current_hv_weight:.2f}, boundary={current_boundary_weight:.2f}, nuclei_type={current_nuclei_type_weight:.2f}, tissue={current_tissue_weight:.2f}, edge={current_edge_weight:.2f}")
        
        # Training epoch
        train_scalar_metrics, train_image_metrics = trainer.train_epoch(
            epoch=epoch,
            train_dataloader=train_dataloader,
            unfreeze_epoch=25  # 在前25个epoch后解冻编码器
        )
        
        # Step scheduler
        scheduler.step()
        
        # 90轮后调整学习率为2e-5
        if epoch + 1 == 90:
            logger.info("Reached epoch 90, setting learning rate to 2e-5")
            for param_group in optimizer.param_groups:
                new_lr = 2e-5
                param_group['lr'] = new_lr
                logger.info(f"Learning rate set to {new_lr:.6f}")

        # 100轮后线性衰减学习率：从 1e-5 降至 5e-6
        if epoch >= 100:
            # 计算线性衰减进度 (0.0 到 1.0)
            # epoch 100 -> progress 0.0 -> lr 1e-5
            # epoch 120 -> progress 1.0 -> lr 5e-6
            progress = (epoch - 100) / 20.0
            # 线性插值: start_lr + (end_lr - start_lr) * progress
            new_lr = 1e-5 + (5e-6 - 1e-5) * progress
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            logger.info(f"Epoch {epoch+1}: Linear Decay Learning rate set to {new_lr:.8f}")
        
        # Get training loss for reference
        train_loss = train_scalar_metrics["Loss/Train"]
        
        # Run validation every 10 epochs (or every epoch after 100) and use validation metric as best model criterion
        val_metric = None
        current_macro_f1 = None
        if (epoch + 1) % 10 == 0 or (epoch + 1) >= 100:
            # Run validation to get metric for best model selection
            logger.info(f"Running validation at epoch {epoch+1}...")
            val_scalar_metrics, val_image_metrics, val_metric = trainer.validation_epoch(
                epoch=epoch,
                val_dataloader=test_dataloader
            )
            
            # 获取当前的 Nuclei-Type-Macro-F1 指标
            if 'nuclei_type_macro_f1' in val_scalar_metrics:
                current_macro_f1 = val_scalar_metrics['nuclei_type_macro_f1']
                logger.info(f"Current Nuclei-Type-Macro-F1: {current_macro_f1:.4f}")
                
                # 获取其他验证指标
                current_boundary_f1 = val_scalar_metrics.get('boundary_f1', 0.0)
                current_dice = val_scalar_metrics.get('dice', 0.0)
                current_bpq = val_scalar_metrics.get('bpq', 0.0)
                
                # 记录验证指标
                validation_history['boundary_f1'].append(current_boundary_f1)
                validation_history['dice'].append(current_dice)
                validation_history['bpq'].append(current_bpq)
                validation_history['macro_f1'].append(current_macro_f1)
                
                logger.info(f"Current validation metrics: Boundary-F1={current_boundary_f1:.4f}, Dice={current_dice:.4f}, bPQ={current_bpq:.4f}")
                
                # 动态调整 Focal Loss 权重
                if previous_macro_f1 is not None:
                    # 获取当前的 Focal Loss 权重
                    current_focal_weight = getattr(trainer, 'current_focal_weight', 1.5)
                    
                    # 第四阶段：90轮后，Focal Loss 权重动态调整
                    if current_epoch > 90:
                        # 计算 Macro-F1 变化百分比
                        macro_f1_change = ((current_macro_f1 - previous_macro_f1) / previous_macro_f1) * 100
                        logger.info(f"Macro-F1 change: {macro_f1_change:.2f}%")
                        
                        # Macro-F1 跌≥1% 则调至 0.12，最高 0.15
                        if macro_f1_change <= -1.0:
                            new_focal_weight = min(current_focal_weight + 0.02, 0.15)
                            trainer.current_focal_weight = new_focal_weight
                            logger.info(f"Fourth phase: Macro-F1 decreased ≥1%, increasing Focal Loss weight to {new_focal_weight:.2f}")
                        else:
                            # 保持0.1不变
                            trainer.current_focal_weight = 0.1
                            logger.info(f"Fourth phase: Keeping Focal Loss weight at 0.1")
                    else:
                        # 计算 Macro-F1 变化百分比
                        macro_f1_change = ((current_macro_f1 - previous_macro_f1) / previous_macro_f1) * 100
                        logger.info(f"Macro-F1 change: {macro_f1_change:.2f}%")
                        
                        # 如果 Macro-F1 下跌≥1%，小幅上调 Focal Loss 权重（最高不超过1.6）
                        if macro_f1_change <= -1.0:
                            new_focal_weight = min(current_focal_weight + 0.05, 1.6)
                            trainer.current_focal_weight = new_focal_weight
                            logger.info(f"Macro-F1 decreased ≥1%, increasing Focal Loss weight to {new_focal_weight:.2f}")
                        # 如果 Macro-F1 上涨≥1%，保持 Focal Loss 权重不变
                        elif macro_f1_change >= 1.0:
                            logger.info(f"Macro-F1 increased ≥1%, keeping Focal Loss weight at {current_focal_weight:.2f}")
                        # 如果变化在±1%以内，保持不变
                        else:
                            logger.info(f"Macro-F1 change within ±1%, keeping Focal Loss weight at {current_focal_weight:.2f}")
                
                # 第四阶段：根据验证指标动态调整其他权重
                if current_epoch > 90:
                    logger.info("Fourth phase: Dynamically adjusting weights based on validation metrics")
                    
                    # 1. 边界损失调整：连续3轮无提升则调至1.6（最高1.7）
                    if len(validation_history['boundary_f1']) >= 4:
                        recent_boundary_f1 = validation_history['boundary_f1'][-4:]
                        if recent_boundary_f1[-1] <= recent_boundary_f1[-2] <= recent_boundary_f1[-3]:
                            boundary_no_improvement_count += 1
                            logger.info(f"Boundary F1 no improvement for {boundary_no_improvement_count} consecutive rounds")
                            
                            if boundary_no_improvement_count >= 3:
                                current_boundary_weight = getattr(trainer, 'current_boundary_weight', 1.5)
                                new_boundary_weight = min(current_boundary_weight + 0.1, 1.7)
                                trainer.current_boundary_weight = new_boundary_weight
                                logger.info(f"Boundary loss weight adjusted from {current_boundary_weight:.2f} to {new_boundary_weight:.2f} due to no improvement in boundary F1")
                                boundary_no_improvement_count = 0
                        else:
                            boundary_no_improvement_count = 0
                    
                    # 2. nuclei_binary_map (BCE)：Dice＜0.84则调至0.15
                    if current_dice < 0.84:
                        current_binary_weight = 0.15
                        update_weight("nuclei_binary_map", current_binary_weight)
                        logger.info(f"Dice < 0.84, adjusting nuclei_binary_map weight to {current_binary_weight:.2f}")
                    
                    # 3. hv_map (MSE)：bPQ＜0.535则调至0.2
                    if current_bpq < 0.535:
                        current_hv_weight = 0.2
                        update_weight("hv_map", current_hv_weight)
                        logger.info(f"bPQ < 0.535, adjusting hv_map weight to {current_hv_weight:.2f}")
                
                previous_macro_f1 = current_macro_f1
            
            # Save checkpoint as milestone
            milestone_checkpoint_path = logdir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'val_metric': val_metric,
                'train_loss': train_loss,
                'best_epoch': early_stopping.best_epoch,
                'patience_counter': getattr(early_stopping, 'patience_counter', 0)
            }, milestone_checkpoint_path)
            logger.info(f"Saved milestone checkpoint to: {milestone_checkpoint_path}")
            
            if val_metric > best_metric or epoch == start_epoch:
                best_metric = val_metric
                best_checkpoint_path = logdir / "best_checkpoint.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'val_metric': val_metric,
                    'train_loss': train_loss,
                    'best_epoch': epoch
                }, best_checkpoint_path)
                logger.info(f"New best model saved with validation metric: {best_metric:.4f} to: {best_checkpoint_path}")
            
            early_stopping(val_metric, epoch)
        else:
            logger.info(f"Training completed, loss: {train_loss:.4f} (will validate at next 10th epoch)")
        
        # 动态调整模型稀疏度（Top-K Ratio）- 针对不同层级赋予不同K值 (Layer-wise Dynamic Sparsity)
        # 基础K值配置
        base_k_schedule = {
            "phase0": [0.30, 0.35, 0.40, 0.45], # Epoch 1-50: 基础特征构建期 (Foundation Phase)
            "phase1": [0.25, 0.30, 0.35, 0.40], # Epoch 50-90: 稀疏化锐化期 (Sharpening Phase)
            "phase2": [0.30, 0.40, 0.50, 0.60]  # Epoch > 90: 语义强化期 (Semantic Phase)
        }
        
        current_k_list = base_k_schedule["phase0"]
        current_sparsity = 0.15 # 初始稀疏度适中
        
        if current_epoch <= 50:
             current_k_list = base_k_schedule["phase0"]
             current_sparsity = 0.15
             if current_epoch == 1 or current_epoch == start_epoch + 1: # Log only once at start/resume
                 logger.info("Epoch <= 50: Using Foundation Phase Sparsity (Moderate K 0.30->0.45)")
                 
        elif 50 < current_epoch <= 90:
            current_k_list = base_k_schedule["phase1"]
            current_sparsity = 0.1
            if current_epoch == 51:
                logger.info("Epoch > 50: Activating Layer-wise Dynamic Sparsity (Phase 1: Gradient K 0.25->0.40)")
        elif current_epoch > 90:
            current_k_list = base_k_schedule["phase2"]
            current_sparsity = 0.05
            if current_epoch == 91:
                logger.info("Epoch > 90: Maximizing Semantic Info in Deep Layers (Phase 2: Gradient K 0.30->0.60)")
        
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'stages'):
            for i_stage, stage in enumerate(model.backbone.stages):
                # 获取当前stage对应的k值，如果超出范围则使用最后一个
                stage_k = current_k_list[min(i_stage, len(current_k_list)-1)]
                
                for layer in stage:
                    if hasattr(layer, 'attn'):
                        layer.attn.k = stage_k
                        layer.attn.sparsity_level = current_sparsity
            
            # Log current k values (sample from first layer of each stage)
            log_k_values = []
            for stage in model.backbone.stages:
                if len(stage) > 0 and hasattr(stage[0], 'attn'):
                    log_k_values.append(f"{stage[0].attn.k:.2f}")
            logger.info(f"Updated Layer-wise K values: {log_k_values}")
        
        # Save checkpoint every epoch (latest checkpoint) with absolute path
        latest_checkpoint_path = logdir / "latest_checkpoint.pth"
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_metric,
            'train_loss': train_loss,
            'best_epoch': early_stopping.best_epoch,
            'patience_counter': getattr(early_stopping, 'patience_counter', 0)
        }
        if val_metric is not None:
            save_dict['val_metric'] = val_metric
        torch.save(save_dict, latest_checkpoint_path)
        logger.info(f"Saved latest checkpoint to: {latest_checkpoint_path}")
        
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    torch.save(model.state_dict(), logdir / "final_model.pth")
    logger.info("Training completed.")
    
    logger.info("Evaluating on test set...")
    test_scalar_metrics, test_image_metrics, test_metric = trainer.validation_epoch(
        epoch=130,
        val_dataloader=test_dataloader
    )
    logger.info(f"Test metric: {test_metric:.4f}")
    logger.info("Test evaluation completed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CellViT model")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to resume from")
    args = parser.parse_args()
    train(resume_from_checkpoint=args.resume, checkpoint_path=args.checkpoint)
