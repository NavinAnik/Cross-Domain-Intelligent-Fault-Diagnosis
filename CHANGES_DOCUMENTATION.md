# DINOv2 Enhancement: Implementation Changes Documentation

**Date**: February 7, 2026  
**Version**: 2.0  
**Authors**: AI Assistant  
**Status**: Complete

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Solution Overview](#solution-overview)
4. [Detailed Changes](#detailed-changes)
5. [Technical Implementation](#technical-implementation)
6. [Testing and Validation](#testing-and-validation)
7. [Expected Results](#expected-results)
8. [Usage Instructions](#usage-instructions)
9. [Appendix](#appendix)

---

## Executive Summary

This document describes the enhancements made to the DINOv2 classification training pipeline to address class imbalance issues and add comprehensive per-epoch metric tracking. The implementation adds three new visualization curves (recall, precision, F1 vs epoch) and implements class-weighted loss to significantly improve minority class detection.

### Key Improvements

- **New Metrics Tracking**: Per-epoch precision, recall, and F1 score computation and visualization
- **Class Balancing**: Implemented weighted cross-entropy loss to handle class imbalance
- **Better Organization**: Consolidated all DINOv2 results into `results/dinov2/` directory
- **Enhanced Monitoring**: Added metrics to Weights & Biases logging
- **Backward Compatibility**: Changes work with existing codebase and all model types

### Impact

- **Abnormal Class Recall**: Expected improvement from 23% → 50%+ (2.2x increase)
- **Abnormal Class F1**: Expected improvement from 37% → 60%+ (1.6x increase)
- **Model Usability**: More balanced predictions suitable for production anomaly detection

---

## Problem Statement

### Current State Analysis

Based on the classification report (`results/dinov2/classification_report_epoch_best.csv`):

```
Class 0 (normal):   Precision: 0.85  Recall: 0.995  F1: 0.92  Support: 17,260
Class 1 (abnormal): Precision: 0.91  Recall: 0.23   F1: 0.37  Support: 3,903
Overall Accuracy: 85.4%
```

### Issues Identified

1. **Severe Class Imbalance**: 
   - Normal samples: 17,260 (81.5%)
   - Abnormal samples: 3,903 (18.5%)
   - Ratio: ~4.4:1

2. **Model Bias**:
   - Model predicts "normal" 99.5% of the time for actual normal samples
   - Model only detects 23% of actual abnormal samples
   - Missing 77% of anomalies is unacceptable for anomaly detection

3. **Incomplete Monitoring**:
   - Only accuracy and loss tracked per epoch
   - No visibility into precision/recall trade-offs during training
   - Cannot optimize for F1 or recall without these metrics

4. **Disorganized Results**:
   - Plots scattered across different directories
   - Difficult to find all DINOv2-related artifacts

---

## Solution Overview

### Approach

We implemented a two-pronged approach:

1. **Weighted Loss Function**: Use class-weighted cross-entropy to force the model to pay more attention to minority class (abnormal samples)

2. **Enhanced Metrics**: Track and visualize precision, recall, and F1 per epoch to monitor model behavior during training

### Design Principles

- **Minimal Invasiveness**: Changes localized to necessary functions
- **Backward Compatibility**: All changes work with existing code
- **Reusability**: Same pattern applied to both main script and experiment runner
- **Maintainability**: Clear, well-documented code following existing patterns

---

## Detailed Changes

### File: `src/main_dinov2.py`

#### 1. Import Additions

```python
# Added line 9
import numpy as np

# Added line 18
from sklearn.metrics import precision_recall_fscore_support
```

**Rationale**: Need numpy for array operations and sklearn for computing binary classification metrics.

---

#### 2. Modified `get_dataloaders()` Function

**Before**:
```python
def get_dataloaders(dataset_path, batch_size, val_split=0.2, image_size=224):
    # ... dataset creation code ...
    return train_loader, val_loader
```

**After**:
```python
def get_dataloaders(dataset_path, batch_size, val_split=0.2, image_size=224):
    """
    Create train and validation dataloaders with DINOv2 transforms.
    Returns: train_loader, val_loader, train_labels (for class weighting)
    """
    # ... dataset creation code ...
    
    # Get train labels for class weighting
    train_labels = [train_dataset.labels[i] for i in train_indices]
    
    # ... loader creation code ...
    
    return train_loader, val_loader, train_labels
```

**Changes**:
- Added return value: `train_labels` (list of training set labels)
- Extracts labels corresponding to training indices after train/val split
- Updated docstring to reflect new return signature

**Impact**: Enables computation of class weights from actual training set distribution.

---

#### 3. Class Weight Computation

**Location**: Lines 256-270 (in main() function, after data loading)

**New Code**:
```python
# Compute class weights for balanced loss
from collections import Counter
class_counts = Counter(train_labels)
total = sum(class_counts.values())
class_weights = [total / (len(class_counts) * class_counts[i]) for i in range(num_classes)]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class distribution: {dict(class_counts)}")
print(f"Class weights (balanced): {class_weights.tolist()}\n")
```

**Formula Explanation**:
```
For each class i:
weight_i = total_samples / (num_classes × count_i)

Example with your data:
- total = 21,163
- num_classes = 2
- class_0_count = 17,260
- class_1_count = 3,903

weight_0 = 21,163 / (2 × 17,260) = 0.613
weight_1 = 21,163 / (2 × 3,903) = 2.711

Result: Abnormal class has 4.4× higher weight (matches class imbalance ratio)
```

**Impact**: Loss function now penalizes misclassifying abnormal samples 4.4× more than normal samples.

---

#### 4. Weighted Loss Function

**Before**:
```python
criterion = nn.CrossEntropyLoss()
```

**After**:
```python
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Impact**: 
- During backpropagation, gradients for abnormal samples are amplified
- Model learns to pay more attention to minority class
- Trades some overall accuracy for better class balance

---

#### 5. Enhanced Training Loop Metrics

**Before**:
```python
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
```

**After**:
```python
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
val_precisions, val_recalls, val_f1s = [], [], []
```

**Per-Epoch Computation** (added after validation):
```python
# Compute precision, recall, F1 for validation
y_true_np = y_true.numpy()
y_pred_np = y_pred.numpy()
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true_np, y_pred_np, average='binary', zero_division=0
)

# Track metrics
val_precisions.append(precision)
val_recalls.append(recall)
val_f1s.append(f1)
```

**Metric Details**:
- `average='binary'`: Computes metrics for positive class (abnormal = 1)
- `zero_division=0`: Returns 0 if denominator is 0 (handles edge cases)
- Computed from predicted labels (not probabilities) for consistency with accuracy

---

#### 6. Enhanced Console Output

**Before**:
```python
print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
```

**After**:
```python
print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
print(f"  Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")
```

**Sample Output**:
```
Epoch 10/20
----------------------------------------
  Train Loss: 0.3245 | Train Acc: 0.8912
  Val Loss:   0.4123 | Val Acc:   0.8534
  Val Precision: 0.7234 | Val Recall: 0.5621 | Val F1: 0.6327
  LR: 0.000087
```

---

#### 7. Weights & Biases Logging

**Before**:
```python
log_metrics({
    "train_loss": train_loss,
    "train_acc": train_acc,
    "val_loss": val_loss,
    "val_acc": val_acc,
    "learning_rate": scheduler.get_last_lr()[0],
}, step=epoch)
```

**After**:
```python
log_metrics({
    "train_loss": train_loss,
    "train_acc": train_acc,
    "val_loss": val_loss,
    "val_acc": val_acc,
    "val_precision": precision,
    "val_recall": recall,
    "val_f1": f1,
    "learning_rate": scheduler.get_last_lr()[0],
}, step=epoch)
```

**Impact**: All metrics now visible in WandB dashboard for experiment tracking and comparison.

---

#### 8. New Visualization Plots

**Before**:
```python
plot_graph(
    x=epochs_range,
    y=[train_losses, val_losses],
    labels=["Train Loss", "Validation Loss"],
    title="DINOv2 - Train vs Validation Loss",
    xlabel="Epoch",
    ylabel="Loss",
    filename="dinov2_loss.png"
)
```

**After**:
```python
results_dir = os.path.join("results", "dinov2")
os.makedirs(results_dir, exist_ok=True)

# Loss plot
plot_graph(
    x=epochs_range,
    y=[train_losses, val_losses],
    labels=["Train Loss", "Validation Loss"],
    title="DINOv2 - Train vs Validation Loss",
    xlabel="Epoch",
    ylabel="Loss",
    filename="loss.png",
    output_dir=results_dir
)

# Accuracy plot
plot_graph(
    x=epochs_range,
    y=[train_accuracies, val_accuracies],
    labels=["Train Accuracy", "Validation Accuracy"],
    title="DINOv2 - Train vs Validation Accuracy",
    xlabel="Epoch",
    ylabel="Accuracy",
    filename="accuracy.png",
    output_dir=results_dir
)

# NEW: Recall curve
plot_graph(
    x=epochs_range,
    y=val_recalls,
    labels=["Validation Recall"],
    title="DINOv2 - Validation Recall vs Epoch",
    xlabel="Epoch",
    ylabel="Recall",
    filename="recall_vs_epoch.png",
    output_dir=results_dir
)

# NEW: Precision curve
plot_graph(
    x=epochs_range,
    y=val_precisions,
    labels=["Validation Precision"],
    title="DINOv2 - Validation Precision vs Epoch",
    xlabel="Epoch",
    ylabel="Precision",
    filename="precision_vs_epoch.png",
    output_dir=results_dir
)

# NEW: F1 curve
plot_graph(
    x=epochs_range,
    y=val_f1s,
    labels=["Validation F1"],
    title="DINOv2 - Validation F1 vs Epoch",
    xlabel="Epoch",
    ylabel="F1 Score",
    filename="f1_vs_epoch.png",
    output_dir=results_dir
)
```

**Key Changes**:
- Created dedicated `results/dinov2/` directory
- Updated existing plots to save in new directory
- Added three new plots for per-epoch metrics
- Simplified filenames (removed "dinov2_" prefix since already in dinov2 folder)

---

#### 9. Enhanced Final Summary

**Before**:
```python
print(f"\n{'='*50}")
print(f"Training Complete!")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"{'='*50}")
```

**After**:
```python
print(f"\n{'='*50}")
print(f"Training Complete!")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Final Val Precision: {val_precisions[-1]:.4f}")
print(f"Final Val Recall: {val_recalls[-1]:.4f}")
print(f"Final Val F1: {val_f1s[-1]:.4f}")
print(f"{'='*50}")
```

**Sample Output**:
```
==================================================
Training Complete!
Best Validation Accuracy: 0.8534
Final Val Precision: 0.7234
Final Val Recall: 0.5621
Final Val F1: 0.6327
==================================================
```

---

### File: `src/experiments/run_all_models.py`

This file received nearly identical changes to ensure consistency across both entry points. Key differences:

#### 1. Import Additions (Line 15)

```python
from sklearn.metrics import precision_recall_fscore_support
```

---

#### 2. Modified `_get_image_dataloaders()` Function

**Signature Change**:
```python
# Before
) -> Tuple[DataLoader, DataLoader]:

# After
) -> Tuple[DataLoader, DataLoader, List[int]]:
```

**Added Code**:
```python
# Get train labels for class weighting
train_labels = [train_ds.labels[i] for i in train_idx]

# ... loader creation ...

return train_loader, val_loader, train_labels
```

**Impact**: Image models (DINOv2, MAE, Swin, BEiT, DeiT, CLIP) now return training labels.

---

#### 3. Class Weight Handling in `run_single_model()`

**New Code** (lines 440-476):
```python
# Dataloaders
model_builder, data_getter, transform_getter = _MODEL_REGISTRY[model_name]
train_labels = None  # For class weighting (image models only)

if transform_getter is not None:
    get_transforms = transform_getter()
    train_loader, val_loader, train_labels = data_getter(
        cfg["dataset_path"],
        cfg["batch_size"],
        get_transforms=get_transforms,
        val_split=cfg.get("val_split", 0.2),
        image_size=cfg.get("image_size", 224),
    )
else:
    train_loader, val_loader = data_getter(
        cfg["dataset_path"],
        cfg["batch_size"],
        val_split=cfg.get("val_split", 0.2),
    )

# Compute class weights for image models (helps with class imbalance)
class_weights = None
if train_labels is not None and cfg.get("use_balanced_loss", True):
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    num_classes = cfg.get("num_classes", 2)
    class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Using balanced class weights: {class_weights.tolist()}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Key Features**:
- Handles both image models (with train_labels) and audio models (without)
- Respects config option `use_balanced_loss` (defaults to True)
- Falls back to unweighted loss if train_labels not available
- Backward compatible with AST and future audio models

---

#### 4. Training Loop Enhancements

Same pattern as `main_dinov2.py`:
- Added `val_precisions, val_recalls, val_f1s` lists
- Compute metrics per epoch from y_true/y_pred
- Print and log to WandB
- Added three new plot calls

---

#### 5. Enhanced Return Dictionary

**Before**:
```python
return {
    "model_name": model_name,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accuracies": train_accs,
    "val_accuracies": val_accs,
    "best_val_acc": best_val_acc,
    "final_val_acc": val_accs[-1],
    "evaluation_metrics": evaluation_metrics,
    "config": cfg,
}
```

**After**:
```python
return {
    "model_name": model_name,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accuracies": train_accs,
    "val_accuracies": val_accs,
    "val_precisions": val_precisions,
    "val_recalls": val_recalls,
    "val_f1s": val_f1s,
    "best_val_acc": best_val_acc,
    "final_val_acc": val_accs[-1],
    "final_val_precision": val_precisions[-1],
    "final_val_recall": val_recalls[-1],
    "final_val_f1": val_f1s[-1],
    "evaluation_metrics": evaluation_metrics,
    "config": cfg,
}
```

**Impact**: Enables programmatic access to all metrics for batch experiments and analysis.

---

## Technical Implementation

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   Training Pipeline                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Data Loading                                         │
│     ├── ImageDataset (with labels)                      │
│     ├── Train/Val Split (stratified)                    │
│     └── Extract train_labels → [0,1,0,1,1,...]         │
│                                                          │
│  2. Class Weight Computation                            │
│     ├── Count: {0: 17260, 1: 3903}                     │
│     ├── Total: 21163                                     │
│     ├── weight_0 = 21163/(2×17260) = 0.613             │
│     └── weight_1 = 21163/(2×3903) = 2.711              │
│                                                          │
│  3. Model Setup                                          │
│     ├── DINOv2 Backbone (frozen)                        │
│     ├── MLP Classifier Head (trainable)                 │
│     └── Loss = CrossEntropyLoss(weight=[0.613, 2.711]) │
│                                                          │
│  4. Training Loop (per epoch)                           │
│     ├── Forward Pass                                     │
│     ├── Weighted Loss Computation                       │
│     ├── Backpropagation (amplified for class 1)        │
│     ├── Validation                                       │
│     ├── Compute: Acc, Prec, Recall, F1                 │
│     ├── Log to WandB                                     │
│     └── Save Best Model                                  │
│                                                          │
│  5. Visualization                                        │
│     ├── loss.png (train/val)                            │
│     ├── accuracy.png (train/val)                        │
│     ├── recall_vs_epoch.png                             │
│     ├── precision_vs_epoch.png                          │
│     ├── f1_vs_epoch.png                                 │
│     └── Save to results/dinov2/                         │
│                                                          │
│  6. Final Evaluation                                     │
│     ├── Load Best Model                                  │
│     ├── Generate Confusion Matrix                       │
│     ├── Generate ROC Curve                              │
│     ├── Generate PR Curve                               │
│     └── Save Classification Report                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

### Metric Computation Details

#### Precision (for abnormal class)
```
Precision = True Positives / (True Positives + False Positives)
          = Correctly predicted abnormal / All predicted abnormal
```

#### Recall (for abnormal class)
```
Recall = True Positives / (True Positives + False Negatives)
       = Correctly predicted abnormal / All actual abnormal
```

#### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = Harmonic mean of precision and recall
```

**Example Calculation**:

Given confusion matrix:
```
                Predicted
                Normal  Abnormal
Actual Normal     950      10      = 960 normal samples
     Abnormal     300     100      = 400 abnormal samples
```

Metrics for abnormal class:
```
Precision = 100 / (100 + 10) = 0.909 (91%)
Recall    = 100 / (100 + 300) = 0.250 (25%)
F1        = 2 × (0.909 × 0.250) / (0.909 + 0.250) = 0.392 (39%)
```

---

### Class Weight Impact on Loss

**Without Weighting**:
```python
# Normal sample misclassified
loss_normal = -log(p_correct) = -log(0.1) = 2.303

# Abnormal sample misclassified  
loss_abnormal = -log(p_correct) = -log(0.1) = 2.303

Total_loss = (loss_normal + loss_abnormal) / 2 = 2.303
```

**With Weighting** (w0=0.613, w1=2.711):
```python
# Normal sample misclassified
loss_normal = 0.613 × -log(0.1) = 0.613 × 2.303 = 1.412

# Abnormal sample misclassified
loss_abnormal = 2.711 × -log(0.1) = 2.711 × 2.303 = 6.243

Total_loss = (1.412 + 6.243) / 2 = 3.828
```

**Result**: Misclassifying an abnormal sample contributes 4.4× more to the total loss, forcing the model to learn better abnormal detection.

---

## Testing and Validation

### Pre-Deployment Checks

✅ **Syntax Validation**:
```bash
python -m py_compile src/main_dinov2.py
python -m py_compile src/experiments/run_all_models.py
```
Status: PASSED (no syntax errors)

✅ **Linter Validation**:
```
ReadLints on both files
```
Status: PASSED (no linter errors)

✅ **Import Validation**:
```python
# All required modules accessible
from sklearn.metrics import precision_recall_fscore_support  # ✓
import numpy as np  # ✓
```

✅ **Type Compatibility**:
- Return types match function signatures
- Torch tensors properly converted to numpy for sklearn
- Class weights properly cast to torch.tensor

---

### Test Plan

#### Unit Test: Class Weight Computation
```python
def test_class_weights():
    train_labels = [0]*17260 + [1]*3903  # Your actual distribution
    
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())  # 21163
    num_classes = 2
    
    weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]
    
    assert abs(weights[0] - 0.613) < 0.001
    assert abs(weights[1] - 2.711) < 0.001
    assert abs(weights[1] / weights[0] - 4.42) < 0.01  # Ratio check
```

#### Integration Test: End-to-End Training (5 epochs)
```bash
cd src
python -m experiments.run_all_models --model dinov2 --epochs 5
```

Expected outputs:
1. Console: Class distribution and weights printed
2. Console: Precision, recall, F1 printed each epoch
3. Files: 5 new PNGs in `results/dinov2/`
4. Files: Best model checkpoint saved

#### Validation Test: Check Improvements
```python
# After training completes
import pandas as pd

# Load new results
report = pd.read_csv("results/dinov2/classification_report_epoch_best.csv")
abnormal_recall = report.loc[1, "recall"]
abnormal_f1 = report.loc[1, "f1-score"]

# Verify improvements
assert abnormal_recall > 0.45, f"Recall {abnormal_recall} below threshold"
assert abnormal_f1 > 0.55, f"F1 {abnormal_f1} below threshold"
```

---

## Expected Results

### Before vs After Comparison

#### Quantitative Metrics

| Metric | Before | After (Expected) | Change |
|--------|--------|------------------|--------|
| **Overall Accuracy** | 85.4% | 80-83% | -2 to -5% |
| **Normal Precision** | 85.1% | 82-85% | Slight decrease |
| **Normal Recall** | 99.5% | 92-95% | -4 to -7% |
| **Normal F1** | 91.7% | 87-90% | -2 to -4% |
| **Abnormal Precision** | 91.4% | 75-85% | May decrease |
| **Abnormal Recall** | **23.0%** | **50-65%** | **+27 to +42%** ⭐ |
| **Abnormal F1** | **36.7%** | **60-72%** | **+23 to +35%** ⭐ |
| **MCC** | 0.415 | 0.50-0.60 | +0.09 to +0.19 |

⭐ = Primary improvement targets

#### Confusion Matrix Evolution

**Before**:
```
                Predicted
                Normal  Abnormal
Actual Normal   17,175    85
     Abnormal    3,006    897
     
→ Missing 77% of abnormal samples
```

**After (Expected)**:
```
                Predicted
                Normal  Abnormal
Actual Normal   16,200  1,060
     Abnormal    1,560  2,343
     
→ Missing only ~40% of abnormal samples
```

#### Visualization Examples

**Recall Curve** (recall_vs_epoch.png):
```
1.0 ┤
0.9 ┤
0.8 ┤
0.7 ┤
0.6 ┤                    ╭────────────
0.5 ┤              ╭─────╯
0.4 ┤         ╭────╯
0.3 ┤    ╭────╯
0.2 ┤────╯
    └─────────────────────────────────
    0    5    10   15   20  (epochs)
```
Expected pattern: Steady increase from ~0.23 to ~0.55-0.65

**Precision Curve** (precision_vs_epoch.png):
```
1.0 ┤
0.9 ┤────╮
0.8 ┤    ╰────╮
0.7 ┤         ╰────╮
0.6 ┤              ╰────────────────
    └─────────────────────────────────
    0    5    10   15   20  (epochs)
```
Expected pattern: Gradual decrease from ~0.91 to ~0.75-0.85 (acceptable trade-off)

---

### Success Criteria

| Criterion | Target | Priority |
|-----------|--------|----------|
| Abnormal Recall | >50% | **CRITICAL** |
| Abnormal F1 | >60% | **HIGH** |
| Overall Accuracy | >78% | MEDIUM |
| Normal Recall | >90% | MEDIUM |
| Training Convergence | Stable curves | HIGH |
| All plots generated | 5 new PNGs | HIGH |

---

## Usage Instructions

### Basic Usage

#### Option 1: Direct Script
```bash
cd src
python main_dinov2.py
```

#### Option 2: Experiment Runner (Recommended)
```bash
cd src
python -m experiments.run_all_models --model dinov2
```

#### Option 3: Custom Configuration
```bash
cd src
python -m experiments.run_all_models \
    --model dinov2 \
    --epochs 30 \
    --batch_size 24 \
    --lr 5e-5
```

---

### Configuration Options

Edit `src/configs/.env`:

```bash
# Dataset
DATASET_PATH=H:/dataset/MIMII_Dataset/spectrograms
NUM_CLASSES=2

# Training
BATCH_SIZE=16           # Increase if GPU allows (24-32)
EPOCHS=20              # Increase to 30 for better convergence
LEARNING_RATE=1e-4     # Decrease to 5e-5 if unstable

# Model
DINOV2_MODEL=dinov2_vitl14    # or dinov2_vitb14 for smaller
IMAGE_SIZE=224
FREEZE_BACKBONE=true

# Optimization
USE_AMP=true                         # Mixed precision
GRADIENT_ACCUMULATION_STEPS=1        # Increase if batch_size limited
```

---

### Monitoring Training

#### Real-time Console Output
```
Epoch 15/20
----------------------------------------
  Epoch 15/20 | Batch 100/500 | Loss: 0.4231 | Acc: 0.8234
  
  Train Loss: 0.3876 | Train Acc: 0.8456
  Val Loss:   0.4234 | Val Acc:   0.8198
  Val Precision: 0.7621 | Val Recall: 0.5432 | Val F1: 0.6341
  LR: 0.000045
  ✓ Saved best model (Val Acc: 0.8198)
```

#### Weights & Biases Dashboard
- Navigate to: https://wandb.ai/your-project/mimii-anomaly-detection
- View real-time curves for all metrics
- Compare multiple runs
- Download data as CSV

#### Results Files
```
results/dinov2/
├── loss.png                              # Loss curves
├── accuracy.png                          # Accuracy curves
├── recall_vs_epoch.png                   # Recall progression ⭐
├── precision_vs_epoch.png                # Precision progression ⭐
├── f1_vs_epoch.png                       # F1 progression ⭐
├── conf_matrix_epoch_best.png            # Final confusion matrix
├── roc_epoch_best.png                    # ROC curve
├── pr_curve_epoch_best.png               # Precision-Recall curve
├── classification_report_epoch_best.png  # Visual report
└── classification_report_epoch_best.csv  # Metric table
```

---

### Post-Training Analysis

#### Load and Analyze Results
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load classification report
report = pd.read_csv("results/dinov2/classification_report_epoch_best.csv", index_col=0)

# Extract metrics
normal_metrics = report.loc["0"]
abnormal_metrics = report.loc["1"]

print(f"Abnormal Class Performance:")
print(f"  Precision: {abnormal_metrics['precision']:.3f}")
print(f"  Recall:    {abnormal_metrics['recall']:.3f}")
print(f"  F1 Score:  {abnormal_metrics['f1-score']:.3f}")

# View curves
from PIL import Image
recall_curve = Image.open("results/dinov2/recall_vs_epoch.png")
recall_curve.show()
```

#### Compare Before/After
```python
# Before
before = {
    'precision': 0.914,
    'recall': 0.230,
    'f1': 0.367
}

# After (from new report)
after = {
    'precision': abnormal_metrics['precision'],
    'recall': abnormal_metrics['recall'],
    'f1': abnormal_metrics['f1-score']
}

# Calculate improvements
improvements = {k: after[k] - before[k] for k in before.keys()}
print(f"\nImprovements:")
for metric, delta in improvements.items():
    print(f"  {metric.capitalize()}: {delta:+.3f} ({delta/before[metric]*100:+.1f}%)")
```

---

## Appendix

### A. Complete File List

#### Modified Files
1. `src/main_dinov2.py` (441 lines)
2. `src/experiments/run_all_models.py` (742 lines)

#### Created Documentation
1. `CHANGES_DOCUMENTATION.md` (this file)
2. `IMPLEMENTATION_SUMMARY.md`
3. `VALIDATION_CHECKLIST.md`

#### Unchanged Files
- `src/utils/plot_graph.py` (supports output_dir, no changes needed)
- `src/evaluate_models.py` (used for final evaluation, no changes needed)
- `src/modules/evaluation.py` (metric utilities, no changes needed)
- `src/modules/dinov2_encoder.py` (model definition, no changes needed)

---

### B. Troubleshooting

#### Issue: Import Error for sklearn
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**:
```bash
pip install scikit-learn
```

#### Issue: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**:
```bash
# Reduce batch size in .env
BATCH_SIZE=8  # or even 4

# Or use gradient accumulation
GRADIENT_ACCUMULATION_STEPS=2
```

#### Issue: Abnormal Recall Still Low (<40%)
**Possible Causes**:
1. Need more epochs (try 30)
2. Learning rate too high (try 5e-5)
3. Need stronger class weights (multiply abnormal weight by 1.5)

**Solution**:
```python
# In main_dinov2.py or run_all_models.py, modify class weight computation:
class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]
class_weights[1] *= 1.5  # Amplify abnormal weight by 50%
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
```

#### Issue: Plots Not Generated
**Check**:
```bash
# Verify directory was created
ls -la results/dinov2/

# Check for errors in console output
grep -i "error" training.log
```

---

### C. Future Enhancements

#### Short-term (Next Version)
1. **Focal Loss**: Alternative to weighted cross-entropy
   ```python
   from torch_loss import FocalLoss
   criterion = FocalLoss(alpha=0.25, gamma=2.0)
   ```

2. **Threshold Tuning**: Find optimal decision threshold
   ```python
   from sklearn.metrics import precision_recall_curve
   precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
   # Find threshold where F1 is maximized
   ```

3. **Data Augmentation**: Stronger augmentation for minority class
   ```python
   from torchvision import transforms
   abnormal_augment = transforms.Compose([
       transforms.RandomRotation(30),
       transforms.ColorJitter(0.3, 0.3, 0.3),
       transforms.RandomHorizontalFlip(),
   ])
   ```

#### Long-term (Future Research)
1. Semi-supervised learning with pseudo-labeling
2. Ensemble methods (combine multiple models)
3. Active learning to identify hard examples
4. Anomaly detection without normal/abnormal labels

---

### D. References

#### Documentation
- [PyTorch CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Sklearn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Weights & Biases Guide](https://docs.wandb.ai/guides)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)

#### Related Work
- He et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
- Chawla et al. "SMOTE: Synthetic Minority Over-sampling Technique" (JAIR 2002)
- Huang et al. "Learning Deep Representations for Imbalanced Classification" (CVPR 2016)

---

### E. Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-02-07 | 2.0 | Initial implementation: class weighting + metric curves | AI Assistant |
| 2026-02-07 | 2.0 | Documentation created | AI Assistant |

---

### F. Contact & Support

For questions or issues:
1. Check this documentation first
2. Review IMPLEMENTATION_SUMMARY.md for quick reference
3. Check VALIDATION_CHECKLIST.md for common problems
4. Create an issue in the project repository

---

**Document Status**: ✅ Complete  
**Last Updated**: February 7, 2026  
**Next Review**: After first training run with new code

