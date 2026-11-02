"""
AFCL Sensitivity Analysis Runner for Q1 Journal Publication (reviewer follow-up).

Tells the reviewer-requested story about the Asymmetric Focal Contrastive Loss:

    1. Why standardized (symmetric) contrastive loss is insufficient in the
       imbalanced fault-diagnosis case --> `component` sweep contrasts a
       symmetric InfoNCE baseline against the asymmetric variant.
    2. What the asymmetric weight does --> `asymmetric` sweep over
       `abnormal_weight` (1.0 = symmetric, > 1.0 = asymmetric).
    3. Sensitivity to focal scaling / temperature --> `focal` cross-sweep.
    4. Sensitivity to prototype-alignment margin --> `margin` sweep.
    5. Sensitivity to component scaling (lambda_contrast x lambda_align)
       --> `scaling` cross-sweep.
    6. Minority-handling robustness --> `minority` sweep over imbalance ratio,
       hard-negative ratio and hard-negative weight.
    7. Final ablation table (CE -> +Focal -> +SymContrast -> +AsymContrast
       -> +Alignment -> +HardMining) --> `component` sweep markdown table.

The training protocol mirrors `experiments.run_ablation_comparison`:
DINOv2-ViT-B/14 frozen backbone with a 256-dim MLP head, WeightedRandomSampler,
AdamW (lr=1e-4, wd=0.05), CosineAnnealingLR, GradScaler-based AMP.

Usage::

    python -m experiments.run_afcl_sensitivity --sweep all \\
        --dataset_path H:/dataset/MIMII_Dataset/spectrograms \\
        --epochs 30 --batch_size 32 --seed 42

    python -m experiments.run_afcl_sensitivity --sweep component
    python -m experiments.run_afcl_sensitivity --sweep asymmetric
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        ".env",
    )
)

from utils.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core sensitivity runner
# ---------------------------------------------------------------------------


class AFCLSensitivityRunner:
    """Run AFCL hyperparameter sensitivity sweeps on the spectrogram pipeline.

    Each sweep cell trains a fresh `DINOv2Classifier` head from scratch (the
    backbone is frozen, so the head is the only thing that adapts). One cell
    therefore differs from another only in the loss configuration, which is
    exactly what a sensitivity / ablation analysis needs.
    """

    def __init__(
        self,
        output_dir: str,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        seed: int = 42,
        feature_dim: int = 768,
        num_classes: int = 2,
        num_workers: int = 4,
    ) -> None:
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_workers = num_workers

        os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Sensitivity runner using device: %s", self.device)

        self._cached_loaders: Optional[Tuple[DataLoader, DataLoader, List[int], List[float], float]] = None

    # ------------------------------------------------------------------
    # Dataset / model wiring (mirrors run_ablation_comparison)
    # ------------------------------------------------------------------

    def _get_dataloaders(self, dataset_path: str):
        """Return train_loader, val_loader, full label list, class weights and
        the empirical imbalance ratio of the *training* split.

        The split / loader objects are cached so all sweep cells share the
        same data partition (this is what makes cells comparable)."""
        if self._cached_loaders is not None:
            return self._cached_loaders

        from experiments.run_all_models import ImageDataset, _get_dinov2_transforms

        get_transforms_fn = _get_dinov2_transforms()
        transforms_dict = get_transforms_fn(image_size=224)

        train_ds = ImageDataset(dataset_path, transform=transforms_dict["train"])
        val_ds = ImageDataset(dataset_path, transform=transforms_dict["val"])

        labels = train_ds.labels
        n_samples = len(labels)
        if n_samples == 0:
            raise RuntimeError(f"No images found under {dataset_path}")

        train_size = int(0.8 * n_samples)
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(n_samples)
        train_idx, val_idx = indices[:train_size], indices[train_size:]

        train_labels = [labels[i] for i in train_idx]
        class_counts = Counter(train_labels)
        num_classes = self.num_classes
        class_weights = [
            n_samples / (num_classes * class_counts.get(i, 1))
            for i in range(num_classes)
        ]
        sample_weights = torch.tensor(
            [class_weights[lab] for lab in train_labels], dtype=torch.float
        )
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(
            Subset(train_ds, train_idx),
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(val_ds, val_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        normal_n = class_counts.get(0, 1)
        abnormal_n = class_counts.get(1, 1)
        imbalance_ratio = max(normal_n, abnormal_n) / max(1, min(normal_n, abnormal_n))

        logger.info(
            "Train: %d (normal=%d, abnormal=%d, ratio=%.2f), Val: %d",
            len(train_idx),
            normal_n,
            abnormal_n,
            imbalance_ratio,
            len(val_idx),
        )

        self._cached_loaders = (
            train_loader,
            val_loader,
            labels,
            class_weights,
            imbalance_ratio,
        )
        return self._cached_loaders

    def _build_model(self) -> nn.Module:
        """Return a fresh DINOv2 classifier head (backbone frozen)."""
        _set_seed(self.seed)
        from modules.dinov2_encoder import DINOv2Classifier

        model = DINOv2Classifier(
            num_classes=self.num_classes,
            model_name="dinov2_vitb14",
            freeze_backbone=True,
            use_fp16=True,
            hidden_dim=256,
        ).to(self.device)
        return model

    # ------------------------------------------------------------------
    # Shared train/eval loop
    # ------------------------------------------------------------------

    def _train_and_evaluate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        loss_kind: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cell_name: str,
    ) -> Dict[str, Any]:
        """Train `model` for `self.epochs` epochs and return best-by-abnormal-recall
        validation metrics.

        `loss_kind` selects the call signature:
            - "ce"       : criterion(logits, labels)
            - "focal_cb" : criterion(logits, labels) — class-balanced focal alone
            - "afcl"     : criterion(features, logits, labels)
        """
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        use_amp = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_abnormal_recall = -1.0
        best_metrics: Dict[str, Any] = {}
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            model.train()
            if hasattr(criterion, "set_epoch"):
                criterion.set_epoch(epoch)

            running_loss = 0.0
            n_batches = 0

            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    if loss_kind == "afcl":
                        features = model.get_features(x)
                        logits = model.classifier(features)
                        loss = criterion(features.float(), logits.float(), y)
                    else:
                        logits = model(x)
                        loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(loss.detach().cpu())
                n_batches += 1

            scheduler.step()

            metrics = self._evaluate(model, val_loader)
            metrics["epoch"] = epoch
            metrics["train_loss"] = running_loss / max(1, n_batches)

            if hasattr(criterion, "get_last_losses"):
                metrics["loss_components"] = criterion.get_last_losses()

            if metrics["abnormal_recall"] > best_abnormal_recall:
                best_abnormal_recall = metrics["abnormal_recall"]
                best_metrics = dict(metrics)
                best_metrics["best_epoch"] = epoch

            logger.info(
                "[%s] epoch %d/%d train_loss=%.4f abn_rec=%.4f bal_acc=%.4f mcc=%.4f",
                cell_name,
                epoch,
                self.epochs,
                metrics["train_loss"],
                metrics["abnormal_recall"],
                metrics["balanced_accuracy"],
                metrics["mcc"],
            )

        best_metrics["wall_time_s"] = time.time() - t0
        return best_metrics

    def _evaluate(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        from sklearn.metrics import matthews_corrcoef

        model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[float] = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device, non_blocking=True)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                _, predicted = logits.max(1)

                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(y.numpy().tolist())
                all_probs.extend(probs[:, 1].cpu().numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        abnormal_mask = y_true == 1
        normal_mask = y_true == 0
        abnormal_recall = (
            float((y_pred[abnormal_mask] == 1).mean()) if abnormal_mask.sum() > 0 else 0.0
        )
        normal_recall = (
            float((y_pred[normal_mask] == 0).mean()) if normal_mask.sum() > 0 else 0.0
        )
        balanced_acc = (abnormal_recall + normal_recall) / 2
        accuracy = float((y_pred == y_true).mean()) if len(y_true) > 0 else 0.0
        mcc = float(matthews_corrcoef(y_true, y_pred)) if len(set(all_labels)) > 1 else 0.0

        return {
            "abnormal_recall": abnormal_recall,
            "normal_recall": normal_recall,
            "balanced_accuracy": balanced_acc,
            "accuracy": accuracy,
            "mcc": mcc,
        }

    # ------------------------------------------------------------------
    # Criterion factories
    # ------------------------------------------------------------------

    def _make_afcl(
        self,
        *,
        focal_alpha: List[float],
        focal_gamma: float,
        temperature: float,
        abnormal_weight: float,
        lambda_contrast: float,
        lambda_align: float,
        margin: float,
        warmup_epochs: int = 5,
        use_hard_mining: bool = False,
        hard_negative_ratio: float = 0.3,
        hard_weight: float = 2.0,
    ) -> nn.Module:
        """Instantiate the AFCL variant directly (bypassing the factory) so each
        knob can be controlled independently for the sweep."""
        from modules.afcl_loss import (
            AsymmetricFocalContrastiveLoss,
            AFCLWithHardMining,
        )

        kwargs = dict(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            temperature=temperature,
            abnormal_weight=abnormal_weight,
            lambda_contrast=lambda_contrast,
            lambda_align=lambda_align,
            margin=margin,
            warmup_epochs=warmup_epochs,
        )
        if use_hard_mining:
            return AFCLWithHardMining(
                hard_negative_ratio=hard_negative_ratio,
                hard_weight=hard_weight,
                **kwargs,
            ).to(self.device)
        return AsymmetricFocalContrastiveLoss(**kwargs).to(self.device)

    def _default_alpha_gamma(self, imbalance_ratio: float) -> Tuple[List[float], float]:
        """Default class-balanced alpha + gamma derived from imbalance ratio.

        Matches `create_afcl_loss` in modules.afcl_loss."""
        normal_w = 1.0 / (1.0 + imbalance_ratio)
        abnormal_w = imbalance_ratio / (1.0 + imbalance_ratio)
        gamma = min(3.0, 1.5 + 0.3 * math.log(imbalance_ratio + 1))
        return [normal_w, abnormal_w], gamma

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    def _save_json(self, payload: Any, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        with open(path, "w", encoding="utf-8") as f:
            json.dump(_convert(payload), f, indent=2)
        logger.info("Wrote %s", path)
        return path

    def _save_csv(self, rows: List[Dict[str, Any]], filename: str, fieldnames: List[str]) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        logger.info("Wrote %s", path)
        return path

    def _save_markdown_table(
        self,
        rows: List[Dict[str, Any]],
        filename: str,
        title: str,
        columns: List[Tuple[str, str]],
    ) -> str:
        """columns: list of (display_name, row_key)."""
        path = os.path.join(self.output_dir, filename)
        lines = [
            f"# {title}",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "| " + " | ".join(name for name, _ in columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in rows:
            cells: List[str] = []
            for _, key in columns:
                val = row.get(key, "")
                if isinstance(val, float):
                    cells.append(f"{val:.4f}")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Wrote %s", path)
        return path

    # ------------------------------------------------------------------
    # Sweeps
    # ------------------------------------------------------------------

    def run_component_ablation(self, dataset_path: str) -> Dict[str, Any]:
        """Component ablation:
        CE -> +Focal -> +SymContrast -> +AsymContrast -> +Alignment -> +HardMining.

        The Symmetric vs Asymmetric rows differ only in `abnormal_weight`
        (1.0 vs 2.0). The Symmetric row is the experimental hook for the
        reviewer's "what symmetric weight does" point: it shows the
        standardized contrastive loss applied to the imbalanced setting.
        """
        logger.info("=" * 70)
        logger.info("SWEEP: component ablation (CE -> Focal -> ... -> full AFCL)")
        logger.info("=" * 70)

        train_loader, val_loader, _labels, class_weights, imb_ratio = self._get_dataloaders(
            dataset_path
        )
        focal_alpha, focal_gamma = self._default_alpha_gamma(imb_ratio)

        configs: List[Dict[str, Any]] = [
            {
                "name": "CE_baseline",
                "description": "Class-weighted cross-entropy (no focal / contrast / align)",
                "build": lambda: (
                    nn.CrossEntropyLoss(
                        weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device)
                    ),
                    "ce",
                ),
            },
            {
                "name": "+Focal",
                "description": "Class-balanced focal loss (alpha+gamma, no contrastive)",
                "build": lambda: (
                    self._make_afcl(
                        focal_alpha=focal_alpha,
                        focal_gamma=focal_gamma,
                        temperature=0.5,
                        abnormal_weight=1.0,
                        lambda_contrast=0.0,
                        lambda_align=0.0,
                        margin=0.5,
                        warmup_epochs=0,
                        use_hard_mining=False,
                    ),
                    "afcl",
                ),
            },
            {
                "name": "+SymContrast",
                "description": "Focal + symmetric InfoNCE contrastive (abnormal_weight=1.0)",
                "build": lambda: (
                    self._make_afcl(
                        focal_alpha=focal_alpha,
                        focal_gamma=focal_gamma,
                        temperature=0.5,
                        abnormal_weight=1.0,
                        lambda_contrast=0.1,
                        lambda_align=0.0,
                        margin=0.5,
                        warmup_epochs=5,
                        use_hard_mining=False,
                    ),
                    "afcl",
                ),
            },
            {
                "name": "+AsymContrast",
                "description": "Focal + asymmetric contrastive (abnormal_weight=2.0)",
                "build": lambda: (
                    self._make_afcl(
                        focal_alpha=focal_alpha,
                        focal_gamma=focal_gamma,
                        temperature=0.5,
                        abnormal_weight=2.0,
                        lambda_contrast=0.1,
                        lambda_align=0.0,
                        margin=0.5,
                        warmup_epochs=5,
                        use_hard_mining=False,
                    ),
                    "afcl",
                ),
            },
            {
                "name": "+Alignment",
                "description": "+ prototype alignment loss (lambda_align=0.05, margin=0.5)",
                "build": lambda: (
                    self._make_afcl(
                        focal_alpha=focal_alpha,
                        focal_gamma=focal_gamma,
                        temperature=0.5,
                        abnormal_weight=2.0,
                        lambda_contrast=0.1,
                        lambda_align=0.05,
                        margin=0.5,
                        warmup_epochs=5,
                        use_hard_mining=False,
                    ),
                    "afcl",
                ),
            },
            {
                "name": "+HardMining (full AFCL)",
                "description": "Full AFCL with hard-negative mining (hard_weight=2.0, ratio=0.3)",
                "build": lambda: (
                    self._make_afcl(
                        focal_alpha=focal_alpha,
                        focal_gamma=focal_gamma,
                        temperature=0.5,
                        abnormal_weight=2.0,
                        lambda_contrast=0.1,
                        lambda_align=0.05,
                        margin=0.5,
                        warmup_epochs=5,
                        use_hard_mining=True,
                        hard_negative_ratio=0.3,
                        hard_weight=2.0,
                    ),
                    "afcl",
                ),
            },
        ]

        results: List[Dict[str, Any]] = []
        baseline_recall: Optional[float] = None
        for cfg in configs:
            logger.info("--- %s : %s", cfg["name"], cfg["description"])
            model = self._build_model()
            criterion, loss_kind = cfg["build"]()

            metrics = self._train_and_evaluate(
                model, criterion, loss_kind, train_loader, val_loader, cfg["name"]
            )
            if baseline_recall is None:
                baseline_recall = metrics["abnormal_recall"]
            metrics["delta_abnormal_recall_vs_CE"] = metrics["abnormal_recall"] - baseline_recall
            metrics["configuration"] = cfg["name"]
            metrics["description"] = cfg["description"]
            results.append(metrics)

            del model, criterion
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._save_json(
            {"imbalance_ratio_train": imb_ratio, "rows": results},
            "component_ablation.json",
        )
        self._save_csv(
            results,
            "component_ablation.csv",
            fieldnames=[
                "configuration",
                "description",
                "abnormal_recall",
                "normal_recall",
                "balanced_accuracy",
                "accuracy",
                "mcc",
                "delta_abnormal_recall_vs_CE",
                "best_epoch",
                "wall_time_s",
            ],
        )
        self._save_markdown_table(
            results,
            "component_ablation.md",
            title="AFCL Component Ablation",
            columns=[
                ("Configuration", "configuration"),
                ("Abnormal Recall", "abnormal_recall"),
                ("Normal Recall", "normal_recall"),
                ("Balanced Acc", "balanced_accuracy"),
                ("MCC", "mcc"),
                ("Δ vs CE", "delta_abnormal_recall_vs_CE"),
            ],
        )

        return {"sweep": "component", "rows": results, "imbalance_ratio_train": imb_ratio}

    def run_asymmetric_sweep(self, dataset_path: str) -> Dict[str, Any]:
        """Sweep `abnormal_weight` to quantify the asymmetric factor's effect.

        abnormal_weight = 1.0 collapses the loss to a symmetric InfoNCE
        contrastive baseline (the "standardized contrastive" the reviewer
        contrasts with). Larger values increasingly upweight abnormal-abnormal
        positive pairs and any abnormal-involving negatives.
        """
        logger.info("=" * 70)
        logger.info("SWEEP: asymmetric factor (abnormal_weight)")
        logger.info("=" * 70)

        train_loader, val_loader, _labels, _cw, imb_ratio = self._get_dataloaders(dataset_path)
        focal_alpha, focal_gamma = self._default_alpha_gamma(imb_ratio)

        values = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        results: List[Dict[str, Any]] = []
        for w_abn in values:
            label = f"w_abn={w_abn}"
            logger.info("--- %s", label)
            model = self._build_model()
            criterion = self._make_afcl(
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                temperature=0.5,
                abnormal_weight=w_abn,
                lambda_contrast=0.1,
                lambda_align=0.05,
                margin=0.5,
                warmup_epochs=5,
                use_hard_mining=False,
            )
            metrics = self._train_and_evaluate(
                model, criterion, "afcl", train_loader, val_loader, label
            )
            metrics["abnormal_weight"] = w_abn
            metrics["is_symmetric_baseline"] = (w_abn == 1.0)
            results.append(metrics)

            del model, criterion
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._save_json(results, "asymmetric_sweep.json")
        self._save_csv(
            results,
            "asymmetric_sweep.csv",
            fieldnames=[
                "abnormal_weight",
                "is_symmetric_baseline",
                "abnormal_recall",
                "normal_recall",
                "balanced_accuracy",
                "accuracy",
                "mcc",
                "best_epoch",
                "wall_time_s",
            ],
        )
        return {"sweep": "asymmetric", "rows": results}

    def run_focal_sweep(self, dataset_path: str) -> Dict[str, Any]:
        """Cross-sweep over `focal_gamma` x `temperature`."""
        logger.info("=" * 70)
        logger.info("SWEEP: focal_gamma x temperature")
        logger.info("=" * 70)

        train_loader, val_loader, _labels, _cw, imb_ratio = self._get_dataloaders(dataset_path)
        focal_alpha, _default_gamma = self._default_alpha_gamma(imb_ratio)

        gammas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        temperatures = [0.07, 0.1, 0.2, 0.5, 1.0]

        results: List[Dict[str, Any]] = []
        for gamma in gammas:
            for tau in temperatures:
                label = f"gamma={gamma}_tau={tau}"
                logger.info("--- %s", label)
                model = self._build_model()
                criterion = self._make_afcl(
                    focal_alpha=focal_alpha,
                    focal_gamma=gamma,
                    temperature=tau,
                    abnormal_weight=2.0,
                    lambda_contrast=0.1,
                    lambda_align=0.05,
                    margin=0.5,
                    warmup_epochs=5,
                    use_hard_mining=False,
                )
                metrics = self._train_and_evaluate(
                    model, criterion, "afcl", train_loader, val_loader, label
                )
                metrics["focal_gamma"] = gamma
                metrics["temperature"] = tau
                results.append(metrics)

                del model, criterion
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        self._save_json(results, "focal_sweep.json")
        self._save_csv(
            results,
            "focal_sweep.csv",
            fieldnames=[
                "focal_gamma",
                "temperature",
                "abnormal_recall",
                "normal_recall",
                "balanced_accuracy",
                "accuracy",
                "mcc",
                "best_epoch",
                "wall_time_s",
            ],
        )
        return {"sweep": "focal", "rows": results}

    def run_margin_sweep(self, dataset_path: str) -> Dict[str, Any]:
        """Sweep prototype-alignment margin."""
        logger.info("=" * 70)
        logger.info("SWEEP: prototype-alignment margin")
        logger.info("=" * 70)

        train_loader, val_loader, _labels, _cw, imb_ratio = self._get_dataloaders(dataset_path)
        focal_alpha, focal_gamma = self._default_alpha_gamma(imb_ratio)

        margins = [0.1, 0.3, 0.5, 0.7, 1.0]
        results: List[Dict[str, Any]] = []
        for m in margins:
            label = f"margin={m}"
            logger.info("--- %s", label)
            model = self._build_model()
            criterion = self._make_afcl(
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                temperature=0.5,
                abnormal_weight=2.0,
                lambda_contrast=0.1,
                lambda_align=0.05,
                margin=m,
                warmup_epochs=5,
                use_hard_mining=False,
            )
            metrics = self._train_and_evaluate(
                model, criterion, "afcl", train_loader, val_loader, label
            )
            metrics["margin"] = m
            results.append(metrics)

            del model, criterion
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._save_json(results, "margin_sweep.json")
        self._save_csv(
            results,
            "margin_sweep.csv",
            fieldnames=[
                "margin",
                "abnormal_recall",
                "normal_recall",
                "balanced_accuracy",
                "accuracy",
                "mcc",
                "best_epoch",
                "wall_time_s",
            ],
        )
        return {"sweep": "margin", "rows": results}

    def run_scaling_sweep(self, dataset_path: str) -> Dict[str, Any]:
        """Cross-sweep `lambda_contrast` x `lambda_align`."""
        logger.info("=" * 70)
        logger.info("SWEEP: lambda_contrast x lambda_align")
        logger.info("=" * 70)

        train_loader, val_loader, _labels, _cw, imb_ratio = self._get_dataloaders(dataset_path)
        focal_alpha, focal_gamma = self._default_alpha_gamma(imb_ratio)

        lambdas_contrast = [0.0, 0.05, 0.1, 0.2, 0.5]
        lambdas_align = [0.0, 0.05, 0.1, 0.2]

        results: List[Dict[str, Any]] = []
        for lc in lambdas_contrast:
            for la in lambdas_align:
                label = f"lc={lc}_la={la}"
                logger.info("--- %s", label)
                model = self._build_model()
                criterion = self._make_afcl(
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    temperature=0.5,
                    abnormal_weight=2.0,
                    lambda_contrast=lc,
                    lambda_align=la,
                    margin=0.5,
                    warmup_epochs=5,
                    use_hard_mining=False,
                )
                metrics = self._train_and_evaluate(
                    model, criterion, "afcl", train_loader, val_loader, label
                )
                metrics["lambda_contrast"] = lc
                metrics["lambda_align"] = la
                results.append(metrics)

                del model, criterion
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        self._save_json(results, "scaling_sweep.json")
        self._save_csv(
            results,
            "scaling_sweep.csv",
            fieldnames=[
                "lambda_contrast",
                "lambda_align",
                "abnormal_recall",
                "normal_recall",
                "balanced_accuracy",
                "accuracy",
                "mcc",
                "best_epoch",
                "wall_time_s",
            ],
        )
        return {"sweep": "scaling", "rows": results}

    def run_minority_sweep(self, dataset_path: str) -> Dict[str, Any]:
        """Minority-handling sensitivity: `imbalance_ratio` (sets alpha/gamma)
        + `hard_negative_ratio` x `hard_weight`.

        Note: this sweep does not resample the dataset; it changes how the
        loss treats the minority class (focal alpha/gamma and hard-negative
        mining intensity) on the *same* training split.
        """
        logger.info("=" * 70)
        logger.info("SWEEP: minority handling (imbalance_ratio + hard mining)")
        logger.info("=" * 70)

        train_loader, val_loader, _labels, _cw, _imb = self._get_dataloaders(dataset_path)

        imbalance_ratios = [2.0, 4.0, 6.0, 8.0, 10.0]
        hard_neg_ratios = [0.0, 0.1, 0.2, 0.3, 0.5]
        hard_weights = [1.0, 2.0, 3.0]

        results: List[Dict[str, Any]] = []

        for r in imbalance_ratios:
            alpha, gamma = self._default_alpha_gamma(r)
            label = f"imb_ratio={r}"
            logger.info("--- %s (alpha=%s, gamma=%.3f)", label, alpha, gamma)
            model = self._build_model()
            criterion = self._make_afcl(
                focal_alpha=alpha,
                focal_gamma=gamma,
                temperature=0.5,
                abnormal_weight=2.0,
                lambda_contrast=0.1,
                lambda_align=0.05,
                margin=0.5,
                warmup_epochs=5,
                use_hard_mining=False,
            )
            metrics = self._train_and_evaluate(
                model, criterion, "afcl", train_loader, val_loader, label
            )
            metrics.update(
                {
                    "axis": "imbalance_ratio",
                    "imbalance_ratio": r,
                    "alpha_normal": alpha[0],
                    "alpha_abnormal": alpha[1],
                    "focal_gamma": gamma,
                    "hard_negative_ratio": None,
                    "hard_weight": None,
                }
            )
            results.append(metrics)
            del model, criterion
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        default_alpha, default_gamma = self._default_alpha_gamma(4.4)
        for hnr in hard_neg_ratios:
            for hw in hard_weights:
                if hnr == 0.0 and hw != 1.0:
                    continue
                label = f"hnr={hnr}_hw={hw}"
                logger.info("--- %s", label)
                model = self._build_model()
                criterion = self._make_afcl(
                    focal_alpha=default_alpha,
                    focal_gamma=default_gamma,
                    temperature=0.5,
                    abnormal_weight=2.0,
                    lambda_contrast=0.1,
                    lambda_align=0.05,
                    margin=0.5,
                    warmup_epochs=5,
                    use_hard_mining=(hnr > 0.0),
                    hard_negative_ratio=max(hnr, 1e-6),
                    hard_weight=hw,
                )
                metrics = self._train_and_evaluate(
                    model, criterion, "afcl", train_loader, val_loader, label
                )
                metrics.update(
                    {
                        "axis": "hard_mining",
                        "imbalance_ratio": None,
                        "alpha_normal": default_alpha[0],
                        "alpha_abnormal": default_alpha[1],
                        "focal_gamma": default_gamma,
                        "hard_negative_ratio": hnr,
                        "hard_weight": hw,
                    }
                )
                results.append(metrics)
                del model, criterion
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        self._save_json(results, "minority_sweep.json")
        self._save_csv(
            results,
            "minority_sweep.csv",
            fieldnames=[
                "axis",
                "imbalance_ratio",
                "alpha_normal",
                "alpha_abnormal",
                "focal_gamma",
                "hard_negative_ratio",
                "hard_weight",
                "abnormal_recall",
                "normal_recall",
                "balanced_accuracy",
                "accuracy",
                "mcc",
                "best_epoch",
                "wall_time_s",
            ],
        )
        return {"sweep": "minority", "rows": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


SWEEP_CHOICES = ["component", "asymmetric", "focal", "margin", "scaling", "minority", "all"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run AFCL sensitivity / ablation sweeps on the MIMII spectrogram "
            "+ DINOv2 pipeline."
        )
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="all",
        choices=SWEEP_CHOICES,
        help="Which sweep to run. 'all' runs every sweep in sequence.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the spectrogram dataset root. Defaults to DATASET_PATH env var.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/afcl_sensitivity",
        help="Parent directory; a timestamped sub-folder is created for this run.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Override the timestamped run-id sub-folder name.",
    )

    args = parser.parse_args()

    setup_logging(log_dir="logs")

    dataset_path = args.dataset_path or os.getenv("DATASET_PATH")
    if not dataset_path:
        logger.error(
            "Dataset path not provided (use --dataset_path or DATASET_PATH env)."
        )
        sys.exit(1)
    if not os.path.exists(dataset_path):
        logger.error("Dataset path does not exist: %s", dataset_path)
        sys.exit(1)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    _set_seed(args.seed)

    runner = AFCLSensitivityRunner(
        output_dir=run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    sweep_registry: Dict[str, Callable[[str], Dict[str, Any]]] = {
        "component": runner.run_component_ablation,
        "asymmetric": runner.run_asymmetric_sweep,
        "focal": runner.run_focal_sweep,
        "margin": runner.run_margin_sweep,
        "scaling": runner.run_scaling_sweep,
        "minority": runner.run_minority_sweep,
    }

    if args.sweep == "all":
        order = ["component", "asymmetric", "focal", "margin", "scaling", "minority"]
    else:
        order = [args.sweep]

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "args": vars(args),
        "dataset_path": dataset_path,
        "sweeps_planned": order,
        "sweeps_completed": [],
        "sweep_files": {},
        "device": str(runner.device),
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    t0 = time.time()
    for sweep_name in order:
        sweep_t0 = time.time()
        logger.info("===== START sweep '%s' =====", sweep_name)
        try:
            sweep_registry[sweep_name](dataset_path)
            manifest["sweeps_completed"].append(sweep_name)
            manifest["sweep_files"][sweep_name] = {
                "json": f"{sweep_name}_sweep.json" if sweep_name != "component"
                else "component_ablation.json",
                "csv": f"{sweep_name}_sweep.csv" if sweep_name != "component"
                else "component_ablation.csv",
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Sweep '%s' failed: %s", sweep_name, exc)
            manifest.setdefault("errors", {})[sweep_name] = str(exc)
        finally:
            manifest[f"{sweep_name}_wall_time_s"] = time.time() - sweep_t0
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        logger.info("===== END sweep '%s' (%.1fs) =====", sweep_name, time.time() - sweep_t0)

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["total_wall_time_s"] = time.time() - t0
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "All requested sweeps complete. Results in %s (total %.1fs).",
        run_dir,
        manifest["total_wall_time_s"],
    )


if __name__ == "__main__":
    main()
