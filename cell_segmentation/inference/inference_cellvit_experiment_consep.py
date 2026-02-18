# -*- coding: utf-8 -*-
# CellViT Inference Method for CoNSeP Dataset (Zero-shot Transfer from PanNuke)
#
# Adapted from inference_cellvit_experiment_pannuke.py
#

import argparse
import inspect
import os
import sys
import logging

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from base_ml.base_experiment import BaseExperiment

BaseExperiment.seed_run(1232)

import json
from pathlib import Path
from typing import List, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from skimage.color import rgba2rgb
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchmetrics.functional import f1_score as dice
from torchmetrics.functional.classification import binary_jaccard_index
from torchvision import transforms

from cell_segmentation.datasets.consep import CoNSePDataset
from models.segmentation.cell_segmentation.cellvit import DataclassHVStorage
from cell_segmentation.utils.metrics import (
    cell_detection_scores,
    cell_type_detection_scores,
    get_fast_pq,
    remap_label,
    binarize,
)
from cell_segmentation.utils.post_proc_cellvit import calculate_instances
from cell_segmentation.utils.tools import cropping_center, pair_coordinates
from models.segmentation.cell_segmentation.sparse_cellvit import SparseCellViT

from utils.logger import Logger


class InferenceCellViTConsep:
    def __init__(
        self,
        run_dir: Union[Path, str],
        gpu: int,
        magnification: int = 40,
        checkpoint_name: str = "model_best.pth",
    ) -> None:
        """Inference for CellViT on CoNSeP
        """
        self.run_dir = Path(run_dir)
        self.device = f"cuda:{gpu}"
        self.run_conf: dict = None
        self.logger: Logger = None
        self.magnification = magnification
        self.checkpoint_name = checkpoint_name

        self.__load_run_conf()

        # Hardcoded PanNuke config for consistency with trained model
        self.dataset_config = {
            "nuclei_types": {
                "Background": 0,
                "Neoplastic": 1,
                "Inflammatory": 2,
                "Connective": 3,
                "Dead": 4,
                "Epithelial": 5
            },
            "tissue_types": {
                "None": 0 
            }
        }
        
        self.__instantiate_logger()
        self.__check_eval_model()
        self.__setup_amp()

        self.logger.info(f"Loaded run: {run_dir}")
        self.num_classes = self.run_conf["data"]["num_nuclei_classes"]

    def __load_run_conf(self) -> None:
        with open((self.run_dir / "config.yaml").resolve(), "r") as run_config_file:
            yaml_config = yaml.safe_load(run_config_file)
            self.run_conf = dict(yaml_config)

    def __instantiate_logger(self) -> None:
        logger = Logger(
            level=self.run_conf["logging"]["level"].upper(),
            log_dir=Path(self.run_dir).resolve(),
            comment="inference_consep",
            use_timestamp=False,
            formatter="%(message)s",
        )
        self.logger = logger.create_logger()

    def __check_eval_model(self) -> None:
        assert (self.run_dir / "checkpoints" / self.checkpoint_name).is_file()

    def __setup_amp(self) -> None:
        self.mixed_precision = self.run_conf["training"].get("mixed_precision", False)

    def get_model(
        self, model_type: str
    ) -> Union[
        SparseCellViT
    ]:
        model = SparseCellViT(
            num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
            embed_dim=self.run_conf["model"]["embed_dim"],
            input_channels=self.run_conf["model"].get("input_channels", 3),
            depth=self.run_conf["model"]["depth"],
            num_heads=self.run_conf["model"]["num_heads"],
            extract_layers=self.run_conf["model"]["extract_layers"],
            regression_loss=self.run_conf["model"].get("regression_loss", False),
        )
        return model

    def setup_patch_inference(
        self
    ) -> tuple[
        Union[
            SparseCellViT
        ],
        DataLoader,
        dict,
    ]:
        # get model for inference
        checkpoint = torch.load(
            self.run_dir / "checkpoints" / self.checkpoint_name, map_location="cpu", weights_only=False
        )
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            model_type = checkpoint.get("arch", "SparseCellViT")
        else:
            state_dict = checkpoint
            model_type = "SparseCellViT"

        model = self.get_model(model_type=model_type)
        self.logger.info(
            f"Loading best model from {str(self.run_dir / 'checkpoints' / self.checkpoint_name)}"
        )
        
        model.load_state_dict(state_dict)

        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        transforms = A.Compose([
            A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=0),
            A.Normalize(mean=mean, std=std),
        ])

        # CoNSeP Data Path
        consep_path = Path("/hy-tmp/SparseSwinCell/cell_segmentation/datasets/process/CoNSeP/testing")
        
        inference_dataset = CoNSePDataset(
            dataset_path=consep_path,
            transforms=transforms,
        )

        inference_dataloader = DataLoader(
            inference_dataset,
            batch_size=1, # Use batch size 1 for safety with different image sizes if any, though CoNSeP is patches
            num_workers=4,
            pin_memory=False,
            shuffle=False,
        )

        return model, inference_dataloader, self.dataset_config

    def run_patch_inference(
        self,
        model: Union[
            SparseCellViT
        ],
        inference_dataloader: DataLoader,
        dataset_config: dict,
        generate_plots: bool = False,
        limit: int = None,
    ) -> None:
        model.to(device=self.device)
        model.eval()

        # setup score tracker
        image_names = []
        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []
        dq_scores = []
        sq_scores = []
        
        paired_all_global = []
        unpaired_true_all_global = []
        unpaired_pred_all_global = []
        true_inst_type_all_global = []
        pred_inst_type_all_global = []

        true_idx_offset = 0
        pred_idx_offset = 0

        total_batches = len(inference_dataloader)
        if limit is not None:
            total_batches = min(total_batches, limit)

        inference_loop = tqdm.tqdm(
            enumerate(inference_dataloader), total=total_batches
        )

        with torch.no_grad():
            for batch_idx, batch in inference_loop:
                if limit is not None and batch_idx >= limit:
                    break
                
                # Remap CoNSeP labels to PanNuke labels for evaluation
                # Batch: [img, masks, filename]
                # masks is dict
                
                masks = batch[1]
                
                # Update binary map:
                # We assume anything > 0 in type map is a nucleus
                # CoNSeP masks already have types 1-7. 
                # We don't care about specific types anymore, just binary.
                # However, we must ensure type values are within [0, num_classes-1] for one-hot encoding in unpack_masks
                
                type_map = masks["nuclei_type_map"]
                new_type_map = torch.zeros_like(type_map)
                # Map all valid cells to class 1 (Neoplastic) for binary-only purpose
                new_type_map[type_map > 0] = 1
                
                masks["nuclei_type_map"] = new_type_map
                masks["nuclei_binary_map"] = (new_type_map > 0).long()
                
                # Fix batch structure if tissue types are missing (CoNSeP case)
                # CoNSePDataset returns [img, masks, names] (len=3)
                if len(batch) == 3:
                    imgs, masks, names = batch
                    # names is a tuple or list from DataLoader collate
                    tissue_types = ["None"] * len(names)
                    batch = [imgs, masks, tissue_types, names]
                
                batch_metrics = self.inference_step(
                    model, batch, generate_plots=generate_plots
                )
                
                image_names = image_names + batch_metrics["image_names"]

                binary_dice_scores = (
                    binary_dice_scores + batch_metrics["binary_dice_scores"]
                )
                binary_jaccard_scores = (
                    binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
                )

                pq_scores = pq_scores + batch_metrics["pq_scores"]
                dq_scores = dq_scores + batch_metrics["dq_scores"]
                sq_scores = sq_scores + batch_metrics["sq_scores"]
                
                # detection scores
                true_idx_offset = (
                    true_idx_offset + true_inst_type_all_global[-1].shape[0]
                    if batch_idx != 0
                    else 0
                )
                pred_idx_offset = (
                    pred_idx_offset + pred_inst_type_all_global[-1].shape[0]
                    if batch_idx != 0
                    else 0
                )
                true_inst_type_all_global.append(batch_metrics["true_inst_type_all"])
                pred_inst_type_all_global.append(batch_metrics["pred_inst_type_all"])
                
                batch_metrics["paired_all"][:, 0] += true_idx_offset
                batch_metrics["paired_all"][:, 1] += pred_idx_offset
                paired_all_global.append(batch_metrics["paired_all"])

                batch_metrics["unpaired_true_all"] += true_idx_offset
                batch_metrics["unpaired_pred_all"] += pred_idx_offset
                unpaired_true_all_global.append(batch_metrics["unpaired_true_all"])
                unpaired_pred_all_global.append(batch_metrics["unpaired_pred_all"])

        paired_all = np.concatenate(paired_all_global, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all_global, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all_global, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all_global, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all_global, axis=0)
        
        # Safe indexing
        if len(paired_all) > 0:
            paired_true_type = true_inst_type_all[paired_all[:, 0]]
            paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
        else:
            paired_true_type = np.array([])
            paired_pred_type = np.array([])
            
        unpaired_true_type = true_inst_type_all[unpaired_true_all]
        unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)
        dq_scores = np.array(dq_scores)
        sq_scores = np.array(sq_scores)

        f1_d, prec_d, rec_d = cell_detection_scores(
            paired_true=paired_true_type,
            paired_pred=paired_pred_type,
            unpaired_true=unpaired_true_type,
            unpaired_pred=unpaired_pred_type,
        )
        dataset_metrics = {
            "Binary-Cell-Dice-Mean": float(np.nanmean(binary_dice_scores)),
            "Binary-Cell-Jacard-Mean": float(np.nanmean(binary_jaccard_scores)),
            "bPQ": float(np.nanmean(pq_scores)),
            "bDQ": float(np.nanmean(dq_scores)),
            "bSQ": float(np.nanmean(sq_scores)),
            "f1_detection": float(f1_d),
            "precision_detection": float(prec_d),
            "recall_detection": float(rec_d),
        }

        # print final results
        self.logger.info(f"{20*'*'} Binary Dataset metrics {20*'*'}")
        [self.logger.info(f"{f'{k}:': <25} {v}") for k, v in dataset_metrics.items()]
        
        # save all folds
        image_metrics = {}
        for idx, image_name in enumerate(image_names):
            image_metrics[image_name] = {
                "Dice": float(binary_dice_scores[idx]),
                "Jaccard": float(binary_jaccard_scores[idx]),
                "bPQ": float(pq_scores[idx]),
            }
        all_metrics = {
            "dataset": dataset_metrics,
            "image_metrics": image_metrics,
        }

        with open(str(self.run_dir / "inference_results_consep.json"), "w") as outfile:
            json.dump(all_metrics, outfile, indent=2)

    def inference_step(
        self,
        model: Union[
            SparseCellViT
        ],
        batch: tuple,
        generate_plots: bool = False,
    ) -> dict:
        imgs = batch[0].to(self.device)
        masks = batch[1]
        tissue_types = list(batch[2])
        image_names = list(batch[3])

        model.zero_grad()
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = model.forward(imgs)
        else:
            predictions = model.forward(imgs)
        predictions = self.unpack_predictions(predictions=predictions, model=model)
        gt = self.unpack_masks(masks=masks, tissue_types=tissue_types, model=model)

        batch_metrics, scores = self.calculate_step_metric(predictions, gt, image_names)
        batch_metrics["tissue_types"] = tissue_types
        if generate_plots:
            self.plot_results(
                imgs=imgs,
                predictions=predictions.get_dict(),
                ground_truth=gt.get_dict(),
                img_names=image_names,
                num_nuclei_classes=self.num_classes,
                outdir=Path(self.run_dir / "inference_predictions_consep"),
                scores=scores,
            )

        return batch_metrics

    def unpack_predictions(
        self, predictions: dict, model: SparseCellViT
    ) -> DataclassHVStorage:
        predictions["tissue_types"] = predictions["tissue_types"].to(self.device)
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )
        (
            predictions["instance_map"],
            predictions["instance_types"],
        ) = model.calculate_instance_map(
            predictions, magnification=self.magnification
        )
        predictions["instance_types_nuclei"] = model.generate_instance_nuclei_map(
            predictions["instance_map"], predictions["instance_types"]
        ).to(
            self.device
        )
        predictions = DataclassHVStorage(
            nuclei_binary_map=predictions["nuclei_binary_map"],
            hv_map=predictions["hv_map"],
            nuclei_type_map=predictions["nuclei_type_map"],
            tissue_types=predictions["tissue_types"],
            instance_map=predictions["instance_map"],
            instance_types=predictions["instance_types"],
            instance_types_nuclei=predictions["instance_types_nuclei"],
            batch_size=predictions["tissue_types"].shape[0],
        )

        return predictions

    def unpack_masks(
        self, masks: dict, tissue_types: list, model: SparseCellViT
    ) -> DataclassHVStorage:
        gt_nuclei_binary_map_onehot = (
            F.one_hot(masks["nuclei_binary_map"].long(), num_classes=2)
        ).type(
            torch.float32
        )
        nuclei_type_maps = masks["nuclei_type_map"].type(torch.int64)
        gt_nuclei_type_maps_onehot = F.one_hot(
            nuclei_type_maps, num_classes=self.num_classes
        ).type(
            torch.float32
        )

        # Handle tissue types (dummy for CoNSeP if missing)
        tissue_types_tensor = torch.zeros(len(tissue_types)).long().to(self.device)

        gt = {
            "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),
            "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),
            "hv_map": masks["hv_map"].to(self.device),
            "instance_map": masks["instance_map"].to(
                self.device
            ),
            "instance_types_nuclei": (
                gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
            )
            .permute(0, 3, 1, 2)
            .to(
                self.device
            ),
            "tissue_types": tissue_types_tensor,
        }
        gt["instance_types"] = calculate_instances(
            gt["nuclei_type_map"], gt["instance_map"]
        )
        gt = DataclassHVStorage(**gt, batch_size=gt["tissue_types"].shape[0])
        return gt

    def calculate_step_metric(
        self,
        predictions: DataclassHVStorage,
        gt: DataclassHVStorage,
        image_names: list[str],
    ) -> Tuple[dict, list]:
        predictions = predictions.get_dict()
        gt = gt.get_dict()

        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        predictions["instance_types_nuclei"] = (
            predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        instance_maps_gt = gt["instance_map"].detach().cpu()
        gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
            torch.uint8
        )
        gt["instance_types_nuclei"] = (
            gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )

        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []
        dq_scores = []
        sq_scores = []
        scores = []

        paired_all = []
        unpaired_true_all = []
        unpaired_pred_all = []
        true_inst_type_all = []
        pred_inst_type_all = []

        true_idx_offset = 0
        pred_idx_offset = 0

        for i in range(len(image_names)):
            pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
            target_binary_map = gt["nuclei_binary_map"][i]
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0, task="binary")
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(cell_dice))

            cell_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(cell_jaccard))

            if len(np.unique(instance_maps_gt[i])) == 1:
                dq, sq, pq = np.nan, np.nan, np.nan
            else:
                remapped_instance_pred = binarize(
                    predictions["instance_types_nuclei"][i][1:].transpose(1, 2, 0)
                )
                remapped_gt = remap_label(instance_maps_gt[i])
                [dq, sq, pq], _ = get_fast_pq(
                    true=remapped_gt, pred=remapped_instance_pred
                )
            pq_scores.append(pq)
            dq_scores.append(dq)
            sq_scores.append(sq)
            scores.append(
                [
                    cell_dice.detach().cpu().numpy(),
                    cell_jaccard.detach().cpu().numpy(),
                    pq,
                ]
            )

            true_centroids = np.array(
                [v["centroid"] for k, v in gt["instance_types"][i].items()]
            )
            true_instance_type = np.array(
                [v["type"] for k, v in gt["instance_types"][i].items()]
            )
            pred_centroids = np.array(
                [v["centroid"] for k, v in predictions["instance_types"][i].items()]
            )
            pred_instance_type = np.array(
                [v["type"] for k, v in predictions["instance_types"][i].items()]
            )

            if true_centroids.shape[0] == 0:
                true_centroids = np.array([[0, 0]])
                true_instance_type = np.array([0])
            if pred_centroids.shape[0] == 0:
                pred_centroids = np.array([[0, 0]])
                pred_instance_type = np.array([0])
            if self.magnification == 40:
                pairing_radius = 12
            else:
                pairing_radius = 6
            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_centroids, pred_centroids, pairing_radius
            )
            true_idx_offset = (
                true_idx_offset + true_inst_type_all[-1].shape[0] if i != 0 else 0
            )
            pred_idx_offset = (
                pred_idx_offset + pred_inst_type_all[-1].shape[0] if i != 0 else 0
            )
            true_inst_type_all.append(true_instance_type)
            pred_inst_type_all.append(pred_instance_type)

            if paired.shape[0] != 0:
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                paired_all.append(paired)

            unpaired_true += true_idx_offset
            unpaired_pred += pred_idx_offset
            unpaired_true_all.append(unpaired_true)
            unpaired_pred_all.append(unpaired_pred)

        paired_all = np.concatenate(paired_all, axis=0) if len(paired_all) > 0 else np.empty((0, 2))
        unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

        batch_metrics = {
            "image_names": image_names,
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
            "dq_scores": dq_scores,
            "sq_scores": sq_scores,
            "paired_all": paired_all,
            "unpaired_true_all": unpaired_true_all,
            "unpaired_pred_all": unpaired_pred_all,
            "true_inst_type_all": true_inst_type_all,
            "pred_inst_type_all": pred_inst_type_all,
        }

        return batch_metrics, scores

    def plot_results(
        self,
        imgs: Union[torch.Tensor, np.ndarray],
        predictions: dict,
        ground_truth: dict,
        img_names: List,
        num_nuclei_classes: int,
        outdir: Union[Path, str],
        scores: List[List[float]] = None,
    ) -> None:
        outdir = Path(outdir) / "plots"
        outdir.mkdir(exist_ok=True, parents=True)

        h = ground_truth["hv_map"].shape[2]
        w = ground_truth["hv_map"].shape[3]

        sample_images = (
            imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        )
        sample_images = cropping_center(sample_images, (h, w), True)

        pred_sample_binary_map = (
            predictions["nuclei_binary_map"].permute(0, 2, 3, 1).detach().cpu().numpy()
        )
        # hv maps not needed for plot but kept in dict
        pred_sample_instance_maps = predictions["instance_map"].detach().cpu().numpy()

        gt_sample_binary_map = ground_truth["nuclei_binary_map"].permute(0, 2, 3, 1).detach().cpu().numpy()
        # gt_sample_hv_map not needed
        gt_sample_instance_map = ground_truth["instance_map"].detach().cpu().numpy()
        
        pred_sample_binary_map = pred_sample_binary_map[..., 1]
        gt_sample_binary_map = np.argmax(gt_sample_binary_map, axis=-1)

        binary_cmap = plt.get_cmap("Greys_r")
        instance_map = plt.get_cmap("viridis")
        # For type map, use jet or distinct colors. PanNuke uses jet for type map in original code
        
        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        
        # Robust inverse normalization from monuseg.py
        safe_mean = []
        safe_std = []
        for m, s in zip(mean, std):
             if s == 0: s = 1
             safe_std.append(1/s)
             safe_mean.append(-m/s)
             
        inv_normalize = transforms.Normalize(
            mean=safe_mean,
            std=safe_std,
        )
        
        inv_samples = inv_normalize(torch.tensor(sample_images).permute(0, 3, 1, 2))
        sample_images = inv_samples.permute(0, 2, 3, 1).detach().cpu().numpy()

        for i in range(len(img_names)):
            fig, axs = plt.subplots(figsize=(8, 4), dpi=300) # Adjusted size for 4 columns
            placeholder = np.zeros((2 * h, 4 * w, 3))
            
            # 1. Image
            placeholder[:h, :w, :3] = sample_images[i]
            placeholder[h : 2 * h, :w, :3] = sample_images[i]
            
            # 2. Binary Prediction
            placeholder[:h, w : 2 * w, :3] = rgba2rgb(
                binary_cmap(gt_sample_binary_map[i] * 255)
            )
            placeholder[h : 2 * h, w : 2 * w, :3] = rgba2rgb(
                binary_cmap(pred_sample_binary_map[i])
            )

            # 3. Instance Map
            placeholder[:h, 2 * w : 3 * w, :3] = rgba2rgb(
                instance_map(
                    (gt_sample_instance_map[i] - np.min(gt_sample_instance_map[i]))
                    / (
                        np.max(gt_sample_instance_map[i])
                        - np.min(gt_sample_instance_map[i] + 1e-10)
                    )
                )
            )
            placeholder[h : 2 * h, 2 * w : 3 * w, :3] = rgba2rgb(
                instance_map(
                    (
                        pred_sample_instance_maps[i]
                        - np.min(pred_sample_instance_maps[i])
                    )
                    / (
                        np.max(pred_sample_instance_maps[i])
                        - np.min(pred_sample_instance_maps[i] + 1e-10)
                    )
                )
            )
            
            # 4. Contours
            gt_contours_polygon = [
                v["contour"] for v in ground_truth["instance_types"][i].values()
            ]
            gt_contours_polygon = [
                list(zip(poly[:, 0], poly[:, 1])) for poly in gt_contours_polygon
            ]
            gt_cell_image = Image.fromarray(
                (sample_images[i] * 255).astype(np.uint8)
            ).convert("RGB")
            gt_drawing = ImageDraw.Draw(gt_cell_image)
            # Draw all GT contours in Green
            for poly in gt_contours_polygon:
                gt_drawing.polygon(poly, outline="#00ff00", width=2)

            gt_cell_image.save(outdir / f"raw_gt_{img_names[i]}")
            placeholder[:h, 3 * w : 4 * w, :3] = np.asarray(gt_cell_image) / 255
            
            pred_contours_polygon = [
                v["contour"] for v in predictions["instance_types"][i].values()
            ]
            pred_contours_polygon = [
                list(zip(poly[:, 0], poly[:, 1])) for poly in pred_contours_polygon
            ]
            pred_cell_image = Image.fromarray(
                (sample_images[i] * 255).astype(np.uint8)
            ).convert("RGB")
            pred_drawing = ImageDraw.Draw(pred_cell_image)
            # Draw all Pred contours in Green
            for poly in pred_contours_polygon:
                pred_drawing.polygon(poly, outline="#00ff00", width=2)

            pred_cell_image.save(outdir / f"raw_pred_{img_names[i]}")
            placeholder[h : 2 * h, 3 * w : 4 * w, :3] = (
                np.asarray(pred_cell_image) / 255
            )

            axs.imshow(placeholder)
            axs.set_xticks(np.arange(w / 2, 4 * w, w))
            axs.set_xticklabels(
                [
                    "Image",
                    "Binary",
                    "Instances",
                    "Contours",
                ],
                fontsize=8,
            )
            axs.xaxis.tick_top()

            axs.set_yticks(np.arange(h / 2, 2 * h, h))
            axs.set_yticklabels(["GT", "Pred."], fontsize=8)
            axs.tick_params(axis="both", which="both", length=0)
            grid_x = np.arange(w, 3 * w, w)
            grid_y = np.arange(h, 2 * h, h)

            for x_seg in grid_x:
                axs.axvline(x_seg, color="black")
            for y_seg in grid_y:
                axs.axhline(y_seg, color="black")
            
            if scores is not None:
                axs.text(
                    20,
                    1.85 * h,
                    f"Dice: {str(np.round(scores[i][0], 2))}\nJac.: {str(np.round(scores[i][1], 2))}\nbPQ: {str(np.round(scores[i][2], 2))}",
                    bbox={"facecolor": "white", "pad": 2, "alpha": 0.5},
                    fontsize=6,
                )
            fig.suptitle(f"Patch Predictions for {img_names[i]}")
            fig.tight_layout()
            fig.savefig(outdir / f"pred_{img_names[i]}")
            plt.close()


class InferenceCellViTParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT inference on CoNSeP",
        )

        parser.add_argument(
            "--run_dir",
            type=str,
            help="Logging directory of a training run.",
            required=True,
        )
        parser.add_argument(
            "--checkpoint_name",
            type=str,
            default="model_best.pth",
        )
        parser.add_argument(
            "--gpu", type=int, help="Cuda-GPU ID for inference", default=5
        )
        parser.add_argument(
            "--magnification",
            type=int,
            choices=[20, 40],
            default=40,
        )
        parser.add_argument(
            "--plots",
            action="store_true",
            help="Generate inference plots",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
        )

        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = InferenceCellViTParser()
    configuration = configuration_parser.parse_arguments()
    print(configuration)
    inf = InferenceCellViTConsep(
        run_dir=configuration["run_dir"],
        checkpoint_name=configuration["checkpoint_name"],
        gpu=configuration["gpu"],
        magnification=configuration["magnification"],
    )
    model, dataloader, conf = inf.setup_patch_inference()

    inf.run_patch_inference(
        model, dataloader, conf, generate_plots=configuration["plots"], limit=configuration["limit"]
    )
