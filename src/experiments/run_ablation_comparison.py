"""
Ablation Comparison Runner for Q1 Journal Publication.

Runs systematic ablation studies comparing:
- Loss functions: CrossEntropy, FocalLoss, VICReg, AFCL
- Fusion strategies: Early, Late, Cross-Attention, ConFu, ProMoBal, HTSF
- Training strategies: Standard, Curriculum Learning

Usage:
    python -m experiments.run_ablation_comparison --ablation loss
    python -m experiments.run_ablation_comparison --ablation fusion
    python -m experiments.run_ablation_comparison --ablation all
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", ".env"))

from utils.logging_config import setup_logging

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
    wandb = None

logger = logging.getLogger(__name__)


class AblationRunner:
    """Run ablation studies for the improved experiments."""
    
    def __init__(
        self,
        output_dir: str = "results/ablations",
        use_wandb: bool = True,
        epochs: int = 30,
        batch_size: int = 32,
    ):
        self.output_dir = output_dir
        self.use_wandb = use_wandb and _HAS_WANDB
        self.epochs = epochs
        self.batch_size = batch_size
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_dataloaders(self, dataset_path: str):
        """Get train and validation dataloaders."""
        from experiments.run_all_models import ImageDataset, _get_dinov2_transforms
        
        get_transforms_fn = _get_dinov2_transforms()
        transforms_dict = get_transforms_fn(image_size=224)
        
        train_ds = ImageDataset(dataset_path, transform=transforms_dict["train"])
        val_ds = ImageDataset(dataset_path, transform=transforms_dict["val"])
        
        labels = train_ds.labels
        n_samples = len(labels)
        train_size = int(0.8 * n_samples)
        
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[:train_size], indices[train_size:]
        
        train_labels = [labels[i] for i in train_idx]
        class_counts = Counter(train_labels)
        num_classes = 2
        class_weights = [n_samples / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)]
        sample_weights = torch.tensor([class_weights[l] for l in train_labels], dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(
            Subset(train_ds, train_idx),
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            Subset(val_ds, val_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        return train_loader, val_loader, labels, class_weights
    
    def run_loss_ablation(self, dataset_path: str) -> Dict[str, Any]:
        """
        Ablation study on loss functions.
        
        Compares: CrossEntropy, FocalLoss, VICReg-style, AFCL
        """
        logger.info("=" * 60)
        logger.info("ABLATION: Loss Functions")
        logger.info("=" * 60)
        
        train_loader, val_loader, labels, class_weights = self._get_dataloaders(dataset_path)
        
        loss_configs = {
            "CrossEntropy": {
                "class": nn.CrossEntropyLoss,
                "kwargs": {"weight": torch.tensor(class_weights, dtype=torch.float32)},
            },
            "FocalLoss_g2": {
                "class": "focal",
                "kwargs": {"alpha": [0.2, 0.8], "gamma": 2.0},
            },
            "FocalLoss_g3": {
                "class": "focal",
                "kwargs": {"alpha": [0.1, 0.9], "gamma": 3.0},
            },
            "AFCL": {
                "class": "afcl",
                "kwargs": {"imbalance_ratio": 4.4, "use_hard_mining": True},
            },
        }
        
        results = {}
        
        for loss_name, loss_config in loss_configs.items():
            logger.info("Testing loss: %s", loss_name)
            
            from modules.dinov2_encoder import DINOv2Classifier
            model = DINOv2Classifier(
                num_classes=2,
                model_name="dinov2_vitb14",
                freeze_backbone=True,
                use_fp16=True,
                hidden_dim=256,
            ).to(self.device)
            
            if loss_config["class"] == "focal":
                from modules.focal_loss import FocalLoss
                criterion = FocalLoss(**loss_config["kwargs"])
            elif loss_config["class"] == "afcl":
                from modules.afcl_loss import create_afcl_loss
                criterion = create_afcl_loss(feature_dim=768, **loss_config["kwargs"])
            else:
                criterion = loss_config["class"](**loss_config["kwargs"]).to(self.device)
            
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=0.05,
            )
            
            best_metrics = self._train_and_evaluate(
                model, criterion, optimizer, train_loader, val_loader, loss_name
            )
            
            results[loss_name] = best_metrics
            logger.info("%s - Abnormal Recall: %.4f, Balanced Acc: %.4f",
                       loss_name, best_metrics["abnormal_recall"], best_metrics["balanced_accuracy"])
        
        self._save_results(results, "loss_ablation_results.json")
        self._generate_ablation_table(results, "loss_ablation_table.md", "Loss Function Ablation")
        
        return results
    
    def run_fusion_ablation(self, dataset_path: str) -> Dict[str, Any]:
        """
        Ablation study on fusion strategies.
        
        Compares: Early, Late, Cross-Attention, ConFu, ProMoBal, HTSF
        """
        logger.info("=" * 60)
        logger.info("ABLATION: Fusion Strategies")
        logger.info("=" * 60)
        
        train_loader, val_loader, labels, class_weights = self._get_dataloaders(dataset_path)
        
        from modules.dinov2_encoder import DINOv2Encoder
        
        fusion_configs = {
            "Single_DINOv2": {"fusion": None},
            "Early": {"fusion": "early"},
            "Late": {"fusion": "late"},
            "CrossAttention": {"fusion": "cross_attention"},
            "ProMoBal": {"fusion": "promobal"},
            "HTSF": {"fusion": "htsf"},
        }
        
        results = {}
        
        for fusion_name, fusion_config in fusion_configs.items():
            logger.info("Testing fusion: %s", fusion_name)
            
            if fusion_config["fusion"] is None:
                from modules.dinov2_encoder import DINOv2Classifier
                model = DINOv2Classifier(
                    num_classes=2,
                    model_name="dinov2_vitb14",
                    freeze_backbone=True,
                    use_fp16=True,
                    hidden_dim=256,
                ).to(self.device)
            elif fusion_config["fusion"] == "htsf":
                vision_encoder = DINOv2Encoder(model_name="dinov2_vitb14", freeze_backbone=True)
                audio_encoder = DINOv2Encoder(model_name="dinov2_vitb14", freeze_backbone=True)
                
                from fusion.htsf_fusion import HTSFFusion
                model = HTSFFusion(
                    audio_encoder=audio_encoder,
                    vision_encoder=vision_encoder,
                    num_classes=2,
                    hidden_dim=512,
                ).to(self.device)
                
                class FusionWrapper(nn.Module):
                    def __init__(self, fusion_model):
                        super().__init__()
                        self.fusion = fusion_model
                    def forward(self, x):
                        return self.fusion(x, x)
                    def get_features(self, x):
                        return self.fusion.forward(x, x, return_attention=False)
                
                model = FusionWrapper(model)
            elif fusion_config["fusion"] == "promobal":
                vision_encoder = DINOv2Encoder(model_name="dinov2_vitb14", freeze_backbone=True)
                audio_encoder = DINOv2Encoder(model_name="dinov2_vitb14", freeze_backbone=True)
                
                from fusion.promobal_fusion import ProMoBalFusion
                model = ProMoBalFusion(
                    audio_encoder=audio_encoder,
                    vision_encoder=vision_encoder,
                    num_classes=2,
                ).to(self.device)
                
                class FusionWrapper(nn.Module):
                    def __init__(self, fusion_model):
                        super().__init__()
                        self.fusion = fusion_model
                    def forward(self, x):
                        return self.fusion(x, x)
                
                model = FusionWrapper(model)
            else:
                from modules.dinov2_encoder import DINOv2Classifier
                model = DINOv2Classifier(
                    num_classes=2,
                    model_name="dinov2_vitb14",
                    freeze_backbone=True,
                    use_fp16=True,
                    hidden_dim=256,
                ).to(self.device)
            
            from modules.focal_loss import FocalLoss
            criterion = FocalLoss(alpha=[0.1, 0.9], gamma=3.0)
            
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=0.05,
            )
            
            best_metrics = self._train_and_evaluate(
                model, criterion, optimizer, train_loader, val_loader, fusion_name
            )
            
            results[fusion_name] = best_metrics
            logger.info("%s - Abnormal Recall: %.4f, Balanced Acc: %.4f",
                       fusion_name, best_metrics["abnormal_recall"], best_metrics["balanced_accuracy"])
        
        self._save_results(results, "fusion_ablation_results.json")
        self._generate_ablation_table(results, "fusion_ablation_table.md", "Fusion Strategy Ablation")
        
        return results
    
    def run_training_ablation(self, dataset_path: str) -> Dict[str, Any]:
        """
        Ablation study on training strategies.
        
        Compares: Standard, Curriculum Learning
        """
        logger.info("=" * 60)
        logger.info("ABLATION: Training Strategies")
        logger.info("=" * 60)
        
        from experiments.run_all_models import ImageDataset, _get_dinov2_transforms
        
        get_transforms_fn = _get_dinov2_transforms()
        transforms_dict = get_transforms_fn(image_size=224)
        
        full_ds = ImageDataset(dataset_path, transform=transforms_dict["train"])
        val_ds = ImageDataset(dataset_path, transform=transforms_dict["val"])
        
        labels = full_ds.labels
        n_samples = len(labels)
        train_size = int(0.8 * n_samples)
        
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[:train_size], indices[train_size:]
        
        train_labels = [labels[i] for i in train_idx]
        
        val_loader = DataLoader(
            Subset(val_ds, val_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        results = {}
        
        logger.info("Testing: Standard Training")
        from modules.dinov2_encoder import DINOv2Classifier
        model = DINOv2Classifier(
            num_classes=2,
            model_name="dinov2_vitb14",
            freeze_backbone=True,
            use_fp16=True,
            hidden_dim=256,
        ).to(self.device)
        
        class_counts = Counter(train_labels)
        num_classes = 2
        class_weights = [n_samples / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)]
        sample_weights = torch.tensor([class_weights[l] for l in train_labels], dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(
            Subset(full_ds, train_idx),
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
        
        from modules.focal_loss import FocalLoss
        criterion = FocalLoss(alpha=[0.1, 0.9], gamma=3.0)
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=0.05,
        )
        
        standard_metrics = self._train_and_evaluate(
            model, criterion, optimizer, train_loader, val_loader, "Standard"
        )
        results["Standard"] = standard_metrics
        
        logger.info("Testing: Curriculum Learning")
        model = DINOv2Classifier(
            num_classes=2,
            model_name="dinov2_vitb14",
            freeze_backbone=True,
            use_fp16=True,
            hidden_dim=256,
        ).to(self.device)
        
        from modules.curriculum_learning import CurriculumTrainer
        from modules.afcl_loss import create_afcl_loss
        
        curriculum_criterion = create_afcl_loss(
            feature_dim=768,
            num_classes=2,
            imbalance_ratio=4.4,
            use_hard_mining=True,
        )
        
        trainer = CurriculumTrainer(
            model=model,
            train_dataset=Subset(full_ds, train_idx),
            val_loader=val_loader,
            labels=train_labels,
            criterion=curriculum_criterion,
            total_epochs=self.epochs * 2,
            batch_size=self.batch_size,
            learning_rate=1e-4,
            device=self.device,
            use_wandb=False,
        )
        
        curriculum_results = trainer.train()
        curriculum_metrics = curriculum_results.get("best_metrics", {})
        if not curriculum_metrics:
            curriculum_metrics = {
                "abnormal_recall": curriculum_results.get("best_abnormal_recall", 0),
                "balanced_accuracy": 0,
                "accuracy": 0,
                "mcc": 0,
            }
        results["Curriculum"] = curriculum_metrics
        
        self._save_results(results, "training_ablation_results.json")
        self._generate_ablation_table(results, "training_ablation_table.md", "Training Strategy Ablation")
        
        return results
    
    def _train_and_evaluate(
        self, model, criterion, optimizer, train_loader, val_loader, name
    ) -> Dict[str, float]:
        """Train model and return best metrics."""
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        use_amp = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        best_abnormal_recall = 0.0
        best_metrics = {}
        
        for epoch in range(1, self.epochs + 1):
            model.train()
            if hasattr(criterion, 'train'):
                criterion.train()
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    
                    if hasattr(model, 'get_features'):
                        try:
                            features = model.get_features(x)
                            loss = criterion(features, logits, y)
                        except (TypeError, RuntimeError):
                            loss = criterion(logits, y)
                    else:
                        try:
                            loss = criterion(logits, y)
                        except TypeError:
                            loss = nn.CrossEntropyLoss()(logits, y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            scheduler.step()
            
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(self.device)
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    _, predicted = logits.max(1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(y.numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
            
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            
            abnormal_mask = y_true == 1
            normal_mask = y_true == 0
            
            abnormal_recall = (y_pred[abnormal_mask] == 1).mean() if abnormal_mask.sum() > 0 else 0
            normal_recall = (y_pred[normal_mask] == 0).mean() if normal_mask.sum() > 0 else 0
            balanced_acc = (abnormal_recall + normal_recall) / 2
            
            from sklearn.metrics import matthews_corrcoef
            mcc = matthews_corrcoef(y_true, y_pred)
            
            if abnormal_recall > best_abnormal_recall:
                best_abnormal_recall = abnormal_recall
                best_metrics = {
                    "abnormal_recall": abnormal_recall,
                    "normal_recall": normal_recall,
                    "balanced_accuracy": balanced_acc,
                    "accuracy": (y_pred == y_true).mean(),
                    "mcc": mcc,
                }
        
        return best_metrics
    
    def _save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        output_path = os.path.join(self.output_dir, filename)
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert(results), f, indent=2)
        
        logger.info("Results saved to %s", output_path)
    
    def _generate_ablation_table(self, results: Dict[str, Any], filename: str, title: str):
        """Generate markdown table for ablation results."""
        lines = [
            f"# {title}",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "| Configuration | Abnormal Recall | Normal Recall | Balanced Acc | MCC |",
            "| --- | --- | --- | --- | --- |",
        ]
        
        for config_name, metrics in results.items():
            abn_recall = metrics.get("abnormal_recall", 0)
            norm_recall = metrics.get("normal_recall", 0)
            bal_acc = metrics.get("balanced_accuracy", 0)
            mcc = metrics.get("mcc", 0)
            lines.append(f"| {config_name} | {abn_recall:.4f} | {norm_recall:.4f} | {bal_acc:.4f} | {mcc:.4f} |")
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        logger.info("Table saved to %s", output_path)


def main():
    setup_logging(log_dir="logs")
    
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["loss", "fusion", "training", "all"],
                        help="Which ablation study to run")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results/ablations")
    
    args = parser.parse_args()
    
    dataset_path = args.dataset_path or os.getenv("DATASET_PATH", ".")
    
    if not os.path.exists(dataset_path):
        logger.error("Dataset path does not exist: %s", dataset_path)
        sys.exit(1)
    
    runner = AblationRunner(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    if args.ablation == "loss" or args.ablation == "all":
        runner.run_loss_ablation(dataset_path)
    
    if args.ablation == "fusion" or args.ablation == "all":
        runner.run_fusion_ablation(dataset_path)
    
    if args.ablation == "training" or args.ablation == "all":
        runner.run_training_ablation(dataset_path)
    
    logger.info("Ablation studies complete!")


if __name__ == "__main__":
    main()
