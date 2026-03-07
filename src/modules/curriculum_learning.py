"""
Curriculum Learning for Imbalanced Anomaly Detection.

Implements progressive difficulty training to handle severe class imbalance:
- Phase 1: Train on balanced subset (easy)
- Phase 2: Gradually introduce hard negatives
- Phase 3: Full dataset with adaptive sampling

This approach helps the model learn meaningful abnormal class representations
before being overwhelmed by the majority normal class.
"""

import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """
    Manages curriculum learning phases and sample selection strategies.

    Phases:
        1. balanced: Equal sampling of normal and abnormal classes
        2. gradual: Progressively introduce more normal samples (hard negatives)
        3. full: Full dataset with class-weighted sampling
    """

    def __init__(
        self,
        total_epochs: int,
        phase1_ratio: float = 0.3,
        phase2_ratio: float = 0.4,
        hard_negative_percentiles: List[float] = None,
    ):
        """
        Args:
            total_epochs: Total training epochs.
            phase1_ratio: Fraction of epochs for phase 1 (balanced).
            phase2_ratio: Fraction of epochs for phase 2 (gradual).
            hard_negative_percentiles: Percentiles for hard negative selection.
        """
        self.total_epochs = total_epochs
        self.phase1_end = int(total_epochs * phase1_ratio)
        self.phase2_end = int(total_epochs * (phase1_ratio + phase2_ratio))
        self.hard_negative_percentiles = hard_negative_percentiles or [25, 50, 75, 100]

    def get_phase(self, epoch: int) -> str:
        """Get current curriculum phase."""
        if epoch <= self.phase1_end:
            return "balanced"
        elif epoch <= self.phase2_end:
            return "gradual"
        else:
            return "full"

    def get_phase_progress(self, epoch: int) -> float:
        """Get progress within current phase (0.0 to 1.0)."""
        phase = self.get_phase(epoch)
        if phase == "balanced":
            return epoch / max(self.phase1_end, 1)
        elif phase == "gradual":
            return (epoch - self.phase1_end) / max(self.phase2_end - self.phase1_end, 1)
        else:
            return (epoch - self.phase2_end) / max(self.total_epochs - self.phase2_end, 1)

    def get_hard_negative_threshold(self, epoch: int) -> float:
        """
        Get threshold for hard negative mining based on epoch.
        Returns percentile of normal samples to include (closest to abnormal cluster).
        """
        phase = self.get_phase(epoch)
        if phase == "balanced":
            return 0.0
        elif phase == "gradual":
            progress = self.get_phase_progress(epoch)
            idx = min(int(progress * len(self.hard_negative_percentiles)),
                      len(self.hard_negative_percentiles) - 1)
            return self.hard_negative_percentiles[idx] / 100.0
        else:
            return 1.0


class CurriculumSampler:
    """
    Creates curriculum-aware data samplers for each training phase.
    """

    def __init__(
        self,
        dataset: Dataset,
        labels: List[int],
        scheduler: CurriculumScheduler,
        abnormal_class: int = 1,
    ):
        self.dataset = dataset
        self.labels = np.array(labels)
        self.scheduler = scheduler
        self.abnormal_class = abnormal_class

        self.normal_indices = np.where(self.labels != abnormal_class)[0]
        self.abnormal_indices = np.where(self.labels == abnormal_class)[0]

        self._embedding_cache = None
        self._hard_negative_order = None

        logger.info("CurriculumSampler initialized:")
        logger.info("  Normal samples: %d", len(self.normal_indices))
        logger.info("  Abnormal samples: %d", len(self.abnormal_indices))

    def update_embeddings(self, embeddings: np.ndarray):
        """
        Update cached embeddings for hard negative mining.
        Call this periodically (e.g., every few epochs) during training.

        Args:
            embeddings: (N, D) array of embeddings for all samples.
        """
        self._embedding_cache = embeddings
        self._compute_hard_negative_order()

    def _compute_hard_negative_order(self):
        """
        Order normal samples by distance to abnormal cluster centroid.
        Samples closer to abnormal cluster are "harder" negatives.
        """
        if self._embedding_cache is None:
            self._hard_negative_order = self.normal_indices
            return

        abnormal_emb = self._embedding_cache[self.abnormal_indices]
        abnormal_centroid = abnormal_emb.mean(axis=0)

        normal_emb = self._embedding_cache[self.normal_indices]
        distances = np.linalg.norm(normal_emb - abnormal_centroid, axis=1)

        sorted_idx = np.argsort(distances)
        self._hard_negative_order = self.normal_indices[sorted_idx]

    def get_sampler(self, epoch: int, batch_size: int) -> Tuple[Subset, Optional[WeightedRandomSampler]]:
        """
        Get dataset subset and sampler for current epoch.

        Returns:
            Tuple of (Subset, Sampler or None)
        """
        phase = self.scheduler.get_phase(epoch)

        if phase == "balanced":
            return self._get_balanced_sampler(batch_size)
        elif phase == "gradual":
            threshold = self.scheduler.get_hard_negative_threshold(epoch)
            return self._get_gradual_sampler(threshold, batch_size)
        else:
            return self._get_full_sampler(batch_size)

    def _get_balanced_sampler(self, batch_size: int) -> Tuple[Subset, WeightedRandomSampler]:
        """Phase 1: Equal number of normal and abnormal samples."""
        n_abnormal = len(self.abnormal_indices)
        n_normal_sample = min(n_abnormal, len(self.normal_indices))

        np.random.shuffle(self.normal_indices)
        selected_normal = self.normal_indices[:n_normal_sample]

        selected_indices = np.concatenate([selected_normal, self.abnormal_indices])
        np.random.shuffle(selected_indices)

        subset = Subset(self.dataset, selected_indices.tolist())

        weights = torch.ones(len(selected_indices))
        sampler = WeightedRandomSampler(weights, len(weights))

        logger.info("Phase 1 (balanced): %d normal, %d abnormal samples",
                    n_normal_sample, n_abnormal)
        return subset, sampler

    def _get_gradual_sampler(
        self, threshold: float, batch_size: int
    ) -> Tuple[Subset, WeightedRandomSampler]:
        """
        Phase 2: Include progressively more hard negatives.

        Args:
            threshold: Fraction of normal samples to include (by hardness).
        """
        if self._hard_negative_order is None:
            self._hard_negative_order = self.normal_indices

        n_normal_include = max(1, int(len(self._hard_negative_order) * threshold))
        selected_normal = self._hard_negative_order[:n_normal_include]

        selected_indices = np.concatenate([selected_normal, self.abnormal_indices])

        n_abnormal = len(self.abnormal_indices)
        n_normal = len(selected_normal)
        total = n_abnormal + n_normal

        weight_normal = total / (2 * n_normal) if n_normal > 0 else 1.0
        weight_abnormal = total / (2 * n_abnormal) if n_abnormal > 0 else 1.0

        labels = np.array([self.labels[i] for i in selected_indices])
        weights = np.where(labels == self.abnormal_class, weight_abnormal, weight_normal)
        weights = torch.tensor(weights, dtype=torch.float)

        sampler = WeightedRandomSampler(weights, len(weights))
        subset = Subset(self.dataset, selected_indices.tolist())

        logger.info("Phase 2 (gradual): %d normal (%.1f%% hardest), %d abnormal",
                    n_normal, threshold * 100, n_abnormal)
        return subset, sampler

    def _get_full_sampler(self, batch_size: int) -> Tuple[Dataset, WeightedRandomSampler]:
        """Phase 3: Full dataset with class-weighted sampling."""
        class_counts = Counter(self.labels)
        total = sum(class_counts.values())
        num_classes = len(class_counts)

        weights_per_class = {
            c: total / (num_classes * count) for c, count in class_counts.items()
        }
        sample_weights = torch.tensor(
            [weights_per_class[l] for l in self.labels], dtype=torch.float
        )
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        logger.info("Phase 3 (full): Using all %d samples with weighted sampling", total)
        return self.dataset, sampler


class CurriculumTrainer:
    """
    Training loop with curriculum learning for anomaly detection.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_loader: DataLoader,
        labels: List[int],
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[Any] = None,
        device: torch.device = None,
        total_epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        use_amp: bool = True,
        embedding_update_interval: int = 5,
        early_stopping_patience: int = 10,
        accumulation_steps: int = 1,
        use_wandb: bool = False,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.labels = labels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.use_amp = use_amp and self.device.type == "cuda"
        self.embedding_update_interval = embedding_update_interval
        self.early_stopping_patience = early_stopping_patience
        self.accumulation_steps = accumulation_steps
        self.use_wandb = use_wandb
        
        if criterion is None:
            from modules.focal_loss import FocalLoss
            criterion = FocalLoss(alpha=[0.1, 0.9], gamma=3.0)
        self.criterion = criterion
        
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=0.05,
            )
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs
            )

        self.curriculum_scheduler = CurriculumScheduler(total_epochs)
        self.curriculum_sampler = CurriculumSampler(
            train_dataset, labels, self.curriculum_scheduler
        )

        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_recall": [],
            "val_f1": [],
            "phase": [],
            "abnormal_recall": [],
            "mcc": [],
        }
        self.best_val_recall = 0.0
        self.best_abnormal_recall = 0.0
        self.best_model_state = None
        self.epochs_no_improve = 0

    @torch.no_grad()
    def _compute_embeddings(self) -> np.ndarray:
        """Compute embeddings for all training samples."""
        self.model.eval()
        embeddings = []

        temp_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=4,
        )

        for x, _ in temp_loader:
            x = x.to(self.device)

            if hasattr(self.model, "encoder"):
                emb = self.model.encoder(x)
            elif hasattr(self.model, "backbone"):
                emb = self.model.backbone(x)
            else:
                with torch.no_grad():
                    _ = self.model(x)
                    if hasattr(self.model, "_last_features"):
                        emb = self.model._last_features
                    else:
                        emb = self.model(x)

            if emb.dim() > 2:
                emb = emb.mean(dim=list(range(1, emb.dim() - 1)))

            embeddings.append(emb.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def _train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        if hasattr(self.criterion, 'train'):
            self.criterion.train()
        
        total_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(x)
                    
                    if hasattr(self.model, 'get_features'):
                        features = self.model.get_features(x)
                        try:
                            loss = self.criterion(features, logits, y)
                        except TypeError:
                            loss = self.criterion(logits, y)
                    else:
                        loss = self.criterion(logits, y)
                    
                    loss = loss / self.accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(x)
                
                if hasattr(self.model, 'get_features'):
                    features = self.model.get_features(x)
                    try:
                        loss = self.criterion(features, logits, y)
                    except TypeError:
                        loss = self.criterion(logits, y)
                else:
                    loss = self.criterion(logits, y)
                
                loss = loss / self.accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate and return comprehensive metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            
            try:
                loss = self.criterion(logits, y)
            except TypeError:
                loss = torch.nn.CrossEntropyLoss()(logits, y)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = (all_preds == all_labels).mean()

        abnormal_mask = all_labels == 1
        normal_mask = all_labels == 0
        
        if abnormal_mask.sum() > 0:
            abnormal_recall = (all_preds[abnormal_mask] == 1).mean()
        else:
            abnormal_recall = 0.0
        
        if normal_mask.sum() > 0:
            normal_recall = (all_preds[normal_mask] == 0).mean()
        else:
            normal_recall = 0.0

        from sklearn.metrics import f1_score, matthews_corrcoef
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        balanced_acc = (abnormal_recall + normal_recall) / 2

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "abnormal_recall": abnormal_recall,
            "normal_recall": normal_recall,
            "balanced_accuracy": balanced_acc,
            "f1": f1,
            "mcc": mcc,
        }

    def train(self) -> Dict[str, Any]:
        """Run full curriculum learning training."""
        logger.info("Starting Curriculum Learning Training")
        logger.info("Total epochs: %d", self.total_epochs)
        logger.info("Phase 1 (balanced): epochs 1-%d", self.curriculum_scheduler.phase1_end)
        logger.info("Phase 2 (gradual): epochs %d-%d",
                    self.curriculum_scheduler.phase1_end + 1,
                    self.curriculum_scheduler.phase2_end)
        logger.info("Phase 3 (full): epochs %d-%d",
                    self.curriculum_scheduler.phase2_end + 1,
                    self.total_epochs)
        
        best_metrics = {}

        for epoch in range(1, self.total_epochs + 1):
            phase = self.curriculum_scheduler.get_phase(epoch)

            if epoch == 1 or (epoch % self.embedding_update_interval == 0 and phase != "balanced"):
                logger.info("Updating embeddings for hard negative mining...")
                try:
                    embeddings = self._compute_embeddings()
                    self.curriculum_sampler.update_embeddings(embeddings)
                except Exception as e:
                    logger.warning("Failed to compute embeddings: %s", e)

            subset, sampler = self.curriculum_sampler.get_sampler(epoch, self.batch_size)
            train_loader = DataLoader(
                subset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
            )

            train_loss, train_acc = self._train_one_epoch(train_loader, epoch)
            val_metrics = self._validate()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_recall"].append(val_metrics["abnormal_recall"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["phase"].append(phase)
            self.history["abnormal_recall"].append(val_metrics["abnormal_recall"])
            self.history["mcc"].append(val_metrics["mcc"])
            
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        "phase": phase,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "abnormal_recall": val_metrics["abnormal_recall"],
                        "normal_recall": val_metrics["normal_recall"],
                        "balanced_accuracy": val_metrics["balanced_accuracy"],
                        "f1": val_metrics["f1"],
                        "mcc": val_metrics["mcc"],
                    })
                except Exception:
                    pass

            logger.info(
                "Epoch %d/%d [%s] | Train Loss: %.4f, Acc: %.4f | "
                "Val Acc: %.4f, Abnormal Recall: %.4f, MCC: %.4f",
                epoch, self.total_epochs, phase.upper(),
                train_loss, train_acc, val_metrics["accuracy"], 
                val_metrics["abnormal_recall"], val_metrics["mcc"]
            )

            abnormal_recall = val_metrics["abnormal_recall"]
            if abnormal_recall > self.best_abnormal_recall:
                self.best_abnormal_recall = abnormal_recall
                self.best_val_recall = abnormal_recall
                best_metrics = val_metrics.copy()
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.epochs_no_improve = 0
                logger.info("  New best abnormal recall: %.4f", abnormal_recall)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            "history": self.history,
            "best_val_recall": self.best_val_recall,
            "best_abnormal_recall": self.best_abnormal_recall,
            "best_metrics": best_metrics,
            "final_epoch": epoch,
        }


def mine_hard_negatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    abnormal_class: int = 1,
) -> np.ndarray:
    """
    Sample hard negatives: normal samples closest to abnormal cluster.

    Args:
        embeddings: (N, D) embeddings for all samples.
        labels: (N,) class labels.
        k: Number of hard negatives to return.
        abnormal_class: Label for abnormal class.

    Returns:
        Indices of k hardest normal samples.
    """
    abnormal_mask = labels == abnormal_class
    normal_mask = ~abnormal_mask

    if abnormal_mask.sum() == 0 or normal_mask.sum() == 0:
        return np.array([])

    abnormal_centroid = embeddings[abnormal_mask].mean(axis=0)
    normal_embeddings = embeddings[normal_mask]
    normal_indices = np.where(normal_mask)[0]

    distances = np.linalg.norm(normal_embeddings - abnormal_centroid, axis=1)
    hard_idx = distances.argsort()[:k]

    return normal_indices[hard_idx]
