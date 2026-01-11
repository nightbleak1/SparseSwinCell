# -*- coding: utf-8 -*-
# Fine-tuning script for CellViT with MAE pretrained weights
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
import os
from pathlib import Path
from typing import Union, Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from base_ml.base_early_stopping import EarlyStopping
from base_ml.base_utils import setup_logger, load_config
from cell_segmentation.datasets.pannuke import PanNukeDataset
from cell_segmentation.trainer.trainer_cellvit import CellViTTrainer
from models.segmentation.cell_segmentation.cellvit import CellViT

def train():
    """Fine-tuning training function for CellViT with MAE pretrained weights"""
    # Setup logging
    logdir = Path("./logs/finetune_cellvit")
    logdir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger("cellvit_finetune", logdir / "training.log")
    logger.info("Starting fine-tuning of CellViT with MAE pretrained weights")
    
    # 1. 加载MAE预训练权重
    logger.info("Loading MAE pretrained weights...")
    model = CellViT(num_classes=1, in_chans=3)
    try:
        pretrained_weights = torch.load('mae_pretrained.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(pretrained_weights, strict=False)  # 忽略不匹配的权重
        logger.info("Successfully loaded MAE pretrained weights")
    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {str(e)}")
        raise
    
    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Load dataset configuration
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
    
    # Setup dataset and dataloader (using existing dataset class)
    train_dataset = PanNukeDataset(
        base_dir=Path("./data/PanNuke"),
        split="train",
        transform="default",
        in_channels=2  # H-DAB双通道输入
    )
    
    val_dataset = PanNukeDataset(
        base_dir=Path("./data/PanNuke"),
        split="validation",
        transform="default",
        in_channels=2  # H-DAB双通道输入
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Loaded train dataset with {len(train_dataset)} samples")
    logger.info(f"Loaded validation dataset with {len(val_dataset)} samples")
    
    # Setup loss functions (using existing loss function structure)
    loss_fn_dict = {
        "nuclei_binary_map": {
            "bce": (torch.nn.BCEWithLogitsLoss(), 1.0),
            "dice": (torch.nn.BCELoss(), 1.0)
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
    
    # 2. 仅需50 epochs微调
    # 设置AdamW优化器和余弦退火学习率调度器
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=10,
        verbose=True,
        delta=0.001,
        path=logdir / "best_model.pth"
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
        experiment_config={"epochs": 50, "batch_size": 8},
        early_stopping=early_stopping,
        log_images=True,
        magnification=40,
        mixed_precision=True
    )
    
    logger.info("Starting fine-tuning for 50 epochs...")
    
    # 3. 保留原始训练逻辑，但epoch数从200→50
    best_metric = 0.0
    
    for epoch in range(50):
        logger.info(f"Epoch {epoch+1}/50")
        
        # Training epoch
        train_scalar_metrics, train_image_metrics = trainer.train_epoch(
            epoch=epoch,
            train_dataloader=train_dataloader,
            unfreeze_epoch=5  # 在前5个epoch后解冻编码器
        )
        
        # Validation epoch
        val_scalar_metrics, val_image_metrics, val_metric = trainer.validation_epoch(
            epoch=epoch,
            val_dataloader=val_dataloader
        )
        
        # Step scheduler
        scheduler.step()
        
        # Early stopping check
        if early_stopping(val_metric, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Save best model
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), logdir / "best_model_custom.pth")
            logger.info(f"New best model saved with metric: {best_metric:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric
            }, logdir / f"checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), logdir / "final_model.pth")
    logger.info(f"Fine-tuning completed. Best validation metric: {best_metric:.4f}")

if __name__ == "__main__":
    train()