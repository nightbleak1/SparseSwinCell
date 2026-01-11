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

def train(resume_from_checkpoint=True):
    """Training function for CellViT from scratch"""
    # Setup logging
    logdir = Path("./logs/train_cellvit_from_scratch")
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
    
    # 1. 从头初始化模型
    logger.info("Initializing CellViT256 model...")
    model = CellViT(
        num_nuclei_classes=6,  # 包括背景
        num_tissue_classes=19,
        embed_dim=96,  # 使用默认的embed_dim
        input_channels=3,  # RGB输入
        depth=12,  # 总深度，会被按比例分配到4个阶段
        num_heads=3,  # 使用默认的num_heads
        extract_layers=[3, 6, 9, 12],
        regression_loss=True
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
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), ratio=(0.8, 1.2), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
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
        batch_size=72,  # 稳定的batch_size
        shuffle=True,
        num_workers=16,  # 调整workers数量，平衡数据加载和系统资源
        pin_memory=True,
        persistent_workers=True,  # 保持workers持续运行，减少启动开销
        prefetch_factor=2,  # 调整预取因子，平衡数据加载和内存使用
        drop_last=True,  # 丢弃最后一个不完整批次，保持训练稳定性
        sampler=None,  # 使用默认采样器
        collate_fn=None  # 使用默认collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=72,
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
            "bce": (torch.nn.BCEWithLogitsLoss(), 1.0)
        },
        "hv_map": {
            "mse": (torch.nn.MSELoss(), 1.0)
        },
        "nuclei_type_map": {
            "ce": (torch.nn.CrossEntropyLoss(), 1.0)
        },
        "tissue_types": {
            "ce": (torch.nn.CrossEntropyLoss(), 1.0)  
        }
    }
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
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
        experiment_config={"epochs": 100, "batch_size": 72},
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
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            
            # Resume model state
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # Resume optimizer state
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Resume scheduler state
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            # Resume epoch and best metric if available
            start_epoch = checkpoint["epoch"] + 1
            if "best_metric" in checkpoint:
                best_metric = checkpoint["best_metric"]
            
            # Resume early stopping state if available
            if "best_epoch" in checkpoint:
                early_stopping.best_metric = best_metric
                early_stopping.best_epoch = checkpoint["best_epoch"]
                # 如果检查点中保存了patience_counter，也恢复它
                if "patience_counter" in checkpoint:
                    early_stopping.patience_counter = checkpoint["patience_counter"]
            
            logger.info(f"Resumed from epoch {start_epoch}, best metric: {best_metric:.4f}")
        else:
            logger.info("No checkpoint found, starting from epoch 0")
    
    logger.info(f"Starting training for 100 epochs, from epoch {start_epoch+1}")
    
    for epoch in range(start_epoch, 100):
        logger.info(f"Epoch {epoch+1}/100")
        
        # Training epoch
        train_scalar_metrics, train_image_metrics = trainer.train_epoch(
            epoch=epoch,
            train_dataloader=train_dataloader,
            unfreeze_epoch=25  # 在前25个epoch后解冻编码器
        )
        
        # Step scheduler
        scheduler.step()
        
        # Get training loss for reference
        train_loss = train_scalar_metrics["Loss/Train"]
        
        # Run validation every 10 epochs and use validation metric as best model criterion
        val_metric = None
        if (epoch + 1) % 10 == 0:
            # Run validation to get metric for best model selection
            logger.info(f"Running validation at epoch {epoch+1}...")
            val_scalar_metrics, val_image_metrics, val_metric = trainer.validation_epoch(
                epoch=epoch,
                val_dataloader=test_dataloader
            )
            
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
            
            # Save best model based on validation metric (higher is better)
            # 第一次验证时，总是保存模型，因为之前的best_metric可能基于不同的评判标准
            # 或者当新的val_metric高于当前best_metric时保存
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
            
            # Early stopping check using validation metric
            early_stopping(val_metric, epoch)
        else:
            # For non-validation epochs, use training loss for reference only
            logger.info(f"Training completed, loss: {train_loss:.4f} (will validate at next 10th epoch)")
        
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
        # Add validation metric to save_dict if available
        if val_metric is not None:
            save_dict['val_metric'] = val_metric
        torch.save(save_dict, latest_checkpoint_path)
        logger.info(f"Saved latest checkpoint to: {latest_checkpoint_path}")
        
        # Early stopping check
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), logdir / "final_model.pth")
    logger.info("Training completed.")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_scalar_metrics, test_image_metrics, test_metric = trainer.validation_epoch(
        epoch=130,
        val_dataloader=test_dataloader
    )
    logger.info(f"Test metric: {test_metric:.4f}")
    logger.info("Test evaluation completed")
    
    # Final evaluation on MoNuSeg training set (validation set)
    logger.info("Final evaluation on MoNuSeg training set...")
    # Initialize MoNuSeg validation dataset correctly
    monuseg_val_dataset = MoNuSegDataset(
        dataset_path=monuseg_path / "training",
        transforms=None,
        patching=True,
        overlap=64
    )
    monuseg_val_dataloader = DataLoader(
        monuseg_val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    monuseg_val_scalar_metrics, monuseg_val_image_metrics, monuseg_val_metric = trainer.validation_epoch(
        epoch=100,
        val_dataloader=monuseg_val_dataloader
    )
    logger.info(f"MoNuSeg validation metric: {monuseg_val_metric:.4f}")
    logger.info("MoNuSeg evaluation completed")

if __name__ == "__main__":
    train()
