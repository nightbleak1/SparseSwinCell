
# -*- coding: utf-8 -*-
# Ablation Study Training Script for SparseSwinCell
#
# This script supports multiple ablation configurations to study the impact
# of different model components on performance.

import logging
import os
import sys
from pathlib import Path
from typing import Union, Dict

sys.path.append("/hy-tmp/SparseSwinCell")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import albumentations as A

from base_ml.base_early_stopping import EarlyStopping
from utils.logger import Logger
from cell_segmentation.datasets.pannuke import PanNukeDataset
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


ABLATION_CONFIGS = {
    "full_model": {
        "name": "Full Model (SparseSwinCell)",
        "description": "Complete model with all components",
        "use_sparse_attention": True,
        "use_shape_stream": True,
        "use_aspp": True,
        "use_attention_gates": True,
        "k": 0.2,
        "dynamic_k": True
    },
    "no_sparse_attention": {
        "name": "No Sparse Attention",
        "description": "Full attention without sparsity",
        "use_sparse_attention": False,
        "use_shape_stream": True,
        "use_aspp": True,
        "use_attention_gates": True,
        "k": 1.0,
        "dynamic_k": False
    },
    "no_shape_stream": {
        "name": "No Shape Stream",
        "description": "Remove boundary detection branch",
        "use_sparse_attention": True,
        "use_shape_stream": False,
        "use_aspp": True,
        "use_attention_gates": True,
        "k": 0.2,
        "dynamic_k": True
    },
    "no_aspp": {
        "name": "No ASPP",
        "description": "Remove Atrous Spatial Pyramid Pooling",
        "use_sparse_attention": True,
        "use_shape_stream": True,
        "use_aspp": False,
        "use_attention_gates": True,
        "k": 0.2,
        "dynamic_k": True
    },
    "no_attention_gates": {
        "name": "No Attention Gates",
        "description": "Remove attention gate mechanism",
        "use_sparse_attention": True,
        "use_shape_stream": True,
        "use_aspp": True,
        "use_attention_gates": False,
        "k": 0.2,
        "dynamic_k": True
    }
}


def train_ablation(ablation_type: str, resume_from_checkpoint: bool = False):
    """Train a specific ablation configuration"""
    
    if ablation_type not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation type: {ablation_type}. Available: {list(ABLATION_CONFIGS.keys())}")
    
    config = ABLATION_CONFIGS[ablation_type]
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = Path(f"./logs/ablation_{ablation_type}_{timestamp}")
    logdir.mkdir(exist_ok=True, parents=True)
    
    logger = Logger(level="INFO", log_dir=logdir, comment=f"ablation_{ablation_type}").create_logger()
    logger.info("=" * 80)
    logger.info(f"Starting Ablation Study: {config['name']}")
    logger.info(f"Description: {config['description']}")
    logger.info("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    logger.info("Initializing SparseCellViT model with ablation configuration...")
    model = SparseCellViT(
        num_nuclei_classes=6,
        num_tissue_classes=19,
        embed_dim=128,
        input_channels=3,
        depth=16,
        num_heads=4,
        extract_layers=[4, 8, 12, 16],
        regression_loss=True,
        window_size=16,
        k=config["k"],
        dynamic_k=config["dynamic_k"],
        min_k_ratio=0.1,
        max_k_ratio=0.7,
        sparsity_level=0.2,
        mlp_ratio=3.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        use_shape_stream=config["use_shape_stream"],
        use_aspp=config["use_aspp"],
        use_attention_gates=config["use_attention_gates"]
    )
    
    model = model.to(device)
    logger.info("Model initialized successfully")
    
    pannuke_path = Path("/hy-tmp/SparseSwinCell/cell_segmentation/datasets/process/PanNuke")
    
    dataset_config = {
        "tissue_types": {
            "Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2, "Breast": 3, "Cervix": 4,
            "Colon": 5, "Esophagus": 6, "HeadNeck": 7, "Kidney": 8, "Liver": 9,
            "Lung": 10, "Ovarian": 11, "Pancreatic": 12, "Prostate": 13, "Skin": 14,
            "Stomach": 15, "Testis": 16, "Thyroid": 17, "Uterus": 18
        },
        "nuclei_types": {
            "background": 0, "neoplastic": 1, "inflammatory": 2, "connective": 3,
            "dead": 4, "epithelial": 5
        }
    }
    
    train_transforms = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    ])
    
    train_dataset = PanNukeDataset(
        dataset_path=pannuke_path,
        folds=[0, 1],
        transforms=train_transforms,
        stardist=False,
        regression=True,
        cache_dataset=False
    )
    
    test_dataset = PanNukeDataset(
        dataset_path=pannuke_path,
        folds=[2],
        transforms=None,
        stardist=False,
        regression=False,
        cache_dataset=False
    )
    
    train_dataset.load_cell_count()
    sampling_weights = train_dataset.get_sampling_weights_cell_tissue(gamma=1.0)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sampling_weights,
        num_samples=len(sampling_weights),
        replacement=True
    )
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=21,
        shuffle=False,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=21,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    logger.info(f"Loaded train dataset with {len(train_dataset)} samples")
    logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
    
    nuclei_loss_weights = [0.5, 1.0, 1.0, 1.5, 2.0, 1.0]
    nuclei_loss_weights = torch.Tensor(nuclei_loss_weights).to(device)
    
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
            "ce": (torch.nn.CrossEntropyLoss(), 1.0)
        }
    }
    
    if config["use_shape_stream"]:
        loss_fn_dict["edge_map"] = {
            "bce": (torch.nn.BCEWithLogitsLoss(), 0.5),
            "dice": (DiceLossWithLogits(), 0.5)
        }
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    early_stopping = EarlyStopping(
        patience=10,
        strategy="maximize"
    )
    
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
        experiment_config={"epochs": 100, "batch_size": 16, "ablation_type": ablation_type},
        early_stopping=early_stopping,
        log_images=True,
        magnification=40,
        mixed_precision=True
    )
    
    trainer.accum_iter = 2
    
    logger.info("=" * 80)
    logger.info(f"Starting training for ablation: {config['name']}")
    logger.info("=" * 80)
    
    best_metric = float('-inf')
    
    for epoch in range(100):
        logger.info(f"Epoch {epoch+1}/100")
        
        train_scalar_metrics, train_image_metrics = trainer.train_epoch(
            epoch=epoch,
            train_dataloader=train_dataloader,
            unfreeze_epoch=25
        )
        
        scheduler.step()
        
        val_metric = None
        if (epoch + 1) % 5 == 0 or epoch == 99:
            logger.info(f"Running validation at epoch {epoch+1}...")
            val_scalar_metrics, val_image_metrics, val_metric = trainer.validation_epoch(
                epoch=epoch,
                val_dataloader=test_dataloader
            )
            
            milestone_checkpoint_path = logdir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'val_metric': val_metric,
                'ablation_config': config
            }, milestone_checkpoint_path)
            logger.info(f"Saved checkpoint: {milestone_checkpoint_path}")
            
            if val_metric > best_metric or epoch == 0:
                best_metric = val_metric
                best_checkpoint_path = logdir / "best_checkpoint.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'val_metric': val_metric,
                    'ablation_config': config
                }, best_checkpoint_path)
                logger.info(f"New best model saved with metric: {best_metric:.4f}")
            
            early_stopping(val_metric, epoch)
        
        latest_checkpoint_path = logdir / "latest_checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_metric,
            'ablation_config': config
        }, latest_checkpoint_path)
        
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info("=" * 80)
    logger.info("Training completed. Final evaluation on test set...")
    logger.info("=" * 80)
    
    test_scalar_metrics, test_image_metrics, test_metric = trainer.validation_epoch(
        epoch=100,
        val_dataloader=test_dataloader
    )
    
    results_file = logdir / "ablation_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Ablation Study: {config['name']}\n")
        f.write(f"Description: {config['description']}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Final Test Metric (bPQ): {test_metric:.4f}\n")
        f.write(f"Best Validation Metric: {best_metric:.4f}\n")
        f.write("\nDetailed Metrics:\n")
        for key, value in test_scalar_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    logger.info(f"Results saved to: {results_file}")
    logger.info("Ablation study completed!")
    
    return test_metric, best_metric


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation study for SparseSwinCell")
    parser.add_argument("--ablation", type=str, 
                       choices=list(ABLATION_CONFIGS.keys()),
                       help="Ablation configuration to run")
    parser.add_argument("--list", action="store_true", help="List all available ablation configurations")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available ablation configurations:")
        print("=" * 80)
        for key, config in ABLATION_CONFIGS.items():
            print(f"\n{key}:")
            print(f"  Name: {config['name']}")
            print(f"  Description: {config['description']}")
        sys.exit(0)
    
    if args.ablation is None:
        parser.print_help()
        print("\nError: --ablation is required unless --list is specified")
        sys.exit(1)
    
    train_ablation(args.ablation, resume_from_checkpoint=args.resume)

