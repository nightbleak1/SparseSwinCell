# -*- coding: utf-8 -*-
# Training script for CellViT from scratch with new log directory
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from base_ml.base_early_stopping import EarlyStopping
from utils.logger import Logger
from cell_segmentation.datasets.pannuke import PanNukeDataset
from cell_segmentation.datasets.monuseg import MoNuSegDataset
from cell_segmentation.trainer.trainer_cellvit import CellViTTrainer
from models.segmentation.cell_segmentation.cellvit import CellViT

def train(resume_from_checkpoint=False):
    """Training function for CellViT from scratch with new log directory"""
    # Setup logging with new log directory
    logdir = Path("./logs/train_cellvit_new")
    logdir.mkdir(exist_ok=True, parents=True)
    logger = Logger(level="INFO", log_dir=logdir, comment="training").create_logger()
    logger.info("Starting training of CellViT from scratch with new log directory")
    
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
    
    # Training dataset: PanNuke fold0 and fold1
    train_dataset = PanNukeDataset(
        dataset_path=pannuke_path,
        folds=[0, 1],
        transforms=None,
        stardist=False,
        regression=True,
        cache_dataset=False
    )
    
    # Test dataset: PanNuke fold2
    test_dataset = PanNukeDataset(
        dataset_path=pannuke_path,
        folds=[2],
        transforms=None,
        stardist=False,
        regression=False,
        cache_dataset=False
    )
    
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
        strategy="maximize"  # 最大化验证指标
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
    best_metric = 0.0
    
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
        
        # Run validation to get metric for best model selection
        val_scalar_metrics, val_image_metrics, val_metric = trainer.validation_epoch(
            epoch=epoch,
            val_dataloader=test_dataloader
        )
        
        # Save checkpoint every epoch (latest checkpoint) with absolute path
        latest_checkpoint_path = logdir / "latest_checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_metric,
            'val_metric': val_metric,
            'best_epoch': early_stopping.best_epoch,
            'patience_counter': getattr(early_stopping, 'patience_counter', 0)
        }, latest_checkpoint_path)
        logger.info(f"Saved latest checkpoint to: {latest_checkpoint_path}")
        
        # Save checkpoint every 10 epochs as milestone
        if (epoch + 1) % 10 == 0:
            milestone_checkpoint_path = logdir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'val_metric': val_metric,
                'best_epoch': early_stopping.best_epoch,
                'patience_counter': getattr(early_stopping, 'patience_counter', 0)
            }, milestone_checkpoint_path)
            logger.info(f"Saved milestone checkpoint to: {milestone_checkpoint_path}")
        
        # Save best model based on validation metric
        if val_metric > best_metric:
            best_metric = val_metric
            best_checkpoint_path = logdir / "best_checkpoint.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'val_metric': val_metric,
                'best_epoch': epoch
            }, best_checkpoint_path)
            logger.info(f"New best model saved with metric: {best_metric:.4f} to: {best_checkpoint_path}")
        
        # Early stopping check
        early_stopping(val_metric, epoch)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    final_model_path = logdir / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed. Saved final model to: {final_model_path}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_scalar_metrics, test_image_metrics, test_metric = trainer.validation_epoch(
        epoch=100,
        val_dataloader=test_dataloader
    )
    logger.info(f"Test metric: {test_metric:.4f}")
    logger.info("Test evaluation completed")

if __name__ == "__main__":
    train()