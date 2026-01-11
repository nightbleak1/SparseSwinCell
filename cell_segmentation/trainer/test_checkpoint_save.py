# -*- coding: utf-8 -*-
# Test script to check if checkpoint saving works correctly
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append("/hy-tmp/SparseSwinCell")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from base_ml.base_early_stopping import EarlyStopping
from utils.logger import Logger
from cell_segmentation.datasets.pannuke import PanNukeDataset
from cell_segmentation.trainer.trainer_cellvit import CellViTTrainer
from models.segmentation.cell_segmentation.cellvit import CellViT

def test_checkpoint_save():
    """Test function to check if checkpoint saving works correctly"""
    # Setup logging with test log directory
    logdir = Path("./logs/test_checkpoint_save")
    logdir.mkdir(exist_ok=True, parents=True)
    logger = Logger(level="INFO", log_dir=logdir, comment="test").create_logger()
    logger.info("Starting test for checkpoint saving")
    
    # 1. 初始化一个小型模型用于测试
    logger.info("Initializing CellViT256 model for testing...")
    model = CellViT(
        num_nuclei_classes=6,  # 包括背景
        num_tissue_classes=19,
        embed_dim=96,  # 使用默认的embed_dim
        input_channels=3,  # RGB输入
        depth=2,  # 使用更浅的模型，加快测试速度
        num_heads=2,  # 减少头数，加快测试速度
        extract_layers=[1, 2],
        regression_loss=True
    )
    logger.info("Model initialized successfully")
    
    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Setup dataset paths
    pannuke_path = Path("/hy-tmp/SparseSwinCell/cell_segmentation/datasets/process/PanNuke")
    
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
    
    # Setup dataset and dataloader with only first 100 samples from fold0
    logger.info("Loading test datasets...")
    
    # Training dataset: First 100 samples from PanNuke fold0
    full_train_dataset = PanNukeDataset(
        dataset_path=pannuke_path,
        folds=[0],
        transforms=None,
        stardist=False,
        regression=True,
        cache_dataset=False
    )
    
    # Take only first 100 samples for testing
    test_train_dataset = Subset(full_train_dataset, list(range(100)))
    
    # Test dataset: First 50 samples from PanNuke fold1
    full_test_dataset = PanNukeDataset(
        dataset_path=pannuke_path,
        folds=[1],
        transforms=None,
        stardist=False,
        regression=False,
        cache_dataset=False
    )
    
    test_test_dataset = Subset(full_test_dataset, list(range(50)))
    
    # 启用cuDNN自动调优，加速卷积运算
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 使用较小的batch_size，加快测试速度
    train_dataloader = DataLoader(
        test_train_dataset,
        batch_size=4,  # 小batch_size，加快测试速度
        shuffle=True,
        num_workers=4,  # 减少workers，加快测试速度
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    test_dataloader = DataLoader(
        test_test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    logger.info(f"Loaded train dataset with {len(test_train_dataset)} samples")
    logger.info(f"Loaded test dataset with {len(test_test_dataset)} samples")
    
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
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=5,
        strategy="maximize"
    )
    
    # Setup trainer
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
        experiment_config={"epochs": 10, "batch_size": 4},
        early_stopping=early_stopping,
        log_images=False,  # 不记录图像，加快测试速度
        magnification=40,
        mixed_precision=True
    )
    
    # 设置梯度累积
    trainer.accum_iter = 2
    
    # 运行一轮训练和验证，测试checkpoint保存
    logger.info("Starting test training and validation...")
    
    # 只运行一轮训练
    for epoch in range(0, 1):
        logger.info(f"Epoch {epoch+1}/10")
        
        # Training epoch with fewer iterations for testing
        logger.info("Starting training step...")
        train_scalar_metrics, train_image_metrics = trainer.train_epoch(
            epoch=epoch,
            train_dataloader=train_dataloader,
            unfreeze_epoch=1
        )
        logger.info("Training step completed")
        
        # Step scheduler
        scheduler.step()
        
        # Run validation
        logger.info("Starting validation step...")
        val_scalar_metrics, val_image_metrics, val_metric = trainer.validation_epoch(
            epoch=epoch,
            val_dataloader=test_dataloader
        )
        logger.info("Validation step completed")
        
        # 手动保存最新checkpoint，测试基本保存功能
        logger.info("Testing checkpoint saving...")
        latest_checkpoint_path = logdir / "latest_checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': 0.0,
            'val_metric': val_metric,
            'best_epoch': 0,
            'patience_counter': 0
        }, latest_checkpoint_path)
        logger.info(f"Saved latest checkpoint to: {latest_checkpoint_path}")
        
        # 测试最佳checkpoint保存
        best_checkpoint_path = logdir / "best_checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': val_metric,
            'val_metric': val_metric,
            'best_epoch': epoch
        }, best_checkpoint_path)
        logger.info(f"Saved best checkpoint to: {best_checkpoint_path}")
        
        # 测试checkpoint加载
        logger.info("Testing checkpoint loading...")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
            logger.info("Successfully loaded checkpoint")
            # 验证checkpoint内容
            if "model_state_dict" in checkpoint:
                logger.info("✓ Checkpoint contains model_state_dict")
            if "optimizer_state_dict" in checkpoint:
                logger.info("✓ Checkpoint contains optimizer_state_dict")
            if "scheduler_state_dict" in checkpoint:
                logger.info("✓ Checkpoint contains scheduler_state_dict")
            if "epoch" in checkpoint:
                logger.info(f"✓ Checkpoint contains epoch: {checkpoint['epoch']}")
        except Exception as e:
            logger.error(f"✗ Failed to load checkpoint: {e}")
            return False
    
    # 检查文件是否实际生成
    logger.info("Checking if checkpoint files were actually created...")
    
    import os
    if os.path.exists(latest_checkpoint_path):
        logger.info(f"✓ Latest checkpoint file exists: {latest_checkpoint_path}")
        logger.info(f"  File size: {os.path.getsize(latest_checkpoint_path)} bytes")
    else:
        logger.error(f"✗ Latest checkpoint file not found: {latest_checkpoint_path}")
        return False
    
    if os.path.exists(best_checkpoint_path):
        logger.info(f"✓ Best checkpoint file exists: {best_checkpoint_path}")
        logger.info(f"  File size: {os.path.getsize(best_checkpoint_path)} bytes")
    else:
        logger.error(f"✗ Best checkpoint file not found: {best_checkpoint_path}")
        return False
    
    # 检查是否能从checkpoint恢复模型
    logger.info("Testing model recovery from checkpoint...")
    try:
        # 创建一个新模型
        new_model = CellViT(
            num_nuclei_classes=6,
            num_tissue_classes=19,
            embed_dim=96,
            input_channels=3,
            depth=2,
            num_heads=2,
            extract_layers=[1, 2],
            regression_loss=True
        )
        # 从checkpoint加载状态
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("✓ Successfully recovered model from checkpoint")
    except Exception as e:
        logger.error(f"✗ Failed to recover model from checkpoint: {e}")
        return False
    
    logger.info("All checkpoint save tests passed successfully!")
    return True

if __name__ == "__main__":
    success = test_checkpoint_save()
    if success:
        print("✓ Checkpoint save test passed!")
        sys.exit(0)
    else:
        print("✗ Checkpoint save test failed!")
        sys.exit(1)