# Implementation Summary: DINOv2 Results Enhancement

## Changes Made

### 1. `src/main_dinov2.py`

#### Added imports:
- `numpy as np` - for numerical operations
- `precision_recall_fscore_support` from sklearn.metrics - for computing precision, recall, F1

#### Modified `get_dataloaders()` function:
- Now returns `train_loader, val_loader, train_labels` (added train_labels for class weighting)
- Extracts training set labels from train_indices for computing class weights

#### Added class-weighted loss:
- Computes balanced class weights from training set class distribution
- Uses `nn.CrossEntropyLoss(weight=class_weights)` to handle class imbalance
- Prints class distribution and computed weights

#### Enhanced training loop:
- Added three new metric lists: `val_precisions`, `val_recalls`, `val_f1s`
- Computes precision, recall, F1 for each validation epoch using sklearn
- Logs these metrics to Weights & Biases
- Prints precision, recall, F1 alongside accuracy and loss

#### Added new plots in results/dinov2/:
- `recall_vs_epoch.png` - Validation recall over epochs
- `precision_vs_epoch.png` - Validation precision over epochs
- `f1_vs_epoch.png` - Validation F1 score over epochs
- Updated existing plots (`loss.png`, `accuracy.png`) to save in `results/dinov2/` subdirectory

#### Enhanced final output:
- Prints final precision, recall, F1 alongside best validation accuracy

### 2. `src/experiments/run_all_models.py`

#### Added imports:
- `precision_recall_fscore_support` from sklearn.metrics

#### Modified `_get_image_dataloaders()` function:
- Now returns `train_loader, val_loader, train_labels` (added train_labels)
- Return type annotation updated to `Tuple[DataLoader, DataLoader, List[int]]`

#### Added class-weighted loss for image models:
- Extracts `train_labels` from dataloader when available (image models)
- Computes balanced class weights if `use_balanced_loss` is True (default)
- Uses `nn.CrossEntropyLoss(weight=class_weights)` for balanced training
- Falls back to unweighted loss for audio models (AST) that don't return train_labels

#### Enhanced training loop:
- Added three new metric lists: `val_precisions`, `val_recalls`, `val_f1s`
- Computes binary precision, recall, F1 for each validation epoch
- Logs these metrics to Weights & Biases
- Prints precision, recall, F1 alongside accuracy and loss

#### Added new plots per model:
- `recall_vs_epoch.png` - Validation recall over epochs
- `precision_vs_epoch.png` - Validation precision over epochs  
- `f1_vs_epoch.png` - Validation F1 score over epochs
- All plots save to model-specific subdirectory (e.g., `results/dinov2/`)

#### Enhanced return value and output:
- Returns precision, recall, F1 lists and final values in result dictionary
- Prints final precision, recall, F1 in training summary

## Expected Improvements

### Before (from classification_report_epoch_best.csv):
- **Accuracy**: ~85.4%
- **Class 0 (normal)**: precision 0.85, recall 0.995, F1 0.92
- **Class 1 (abnormal)**: precision 0.91, recall **0.23**, F1 **0.37**
- **Issue**: Model strongly biased toward predicting "normal" class

### After (expected with class-weighted loss):
- **Class 1 (abnormal) recall**: Should improve significantly (target >0.5)
- **Class 1 (abnormal) F1**: Should improve to >0.6
- **Overall accuracy**: May decrease slightly (85% → 80-83%) but this is acceptable
- **Balance**: Much better balance between normal and abnormal class performance

## New Artifacts Generated

When running DINOv2 training, the following files will be created in `results/dinov2/`:

1. **loss.png** - Training and validation loss curves
2. **accuracy.png** - Training and validation accuracy curves
3. **recall_vs_epoch.png** ⭐ NEW - Validation recall over training
4. **precision_vs_epoch.png** ⭐ NEW - Validation precision over training
5. **f1_vs_epoch.png** ⭐ NEW - Validation F1 score over training
6. **conf_matrix_epoch_best.png** - Confusion matrix for best epoch
7. **roc_epoch_best.png** - ROC curve for best epoch
8. **pr_curve_epoch_best.png** - Precision-Recall curve for best epoch
9. **classification_report_epoch_best.png** - Visual classification report
10. **classification_report_epoch_best.csv** - CSV classification report with metrics

## How to Run

### Option 1: Using main_dinov2.py
```bash
cd src
python main_dinov2.py
```

### Option 2: Using run_all_models.py (recommended for experiments)
```bash
cd src
python -m experiments.run_all_models --model dinov2
```

### Optional: Adjust config
Edit `src/configs/.env` to modify:
- `EPOCHS=30` (increase from 20 if needed)
- `LEARNING_RATE=5e-5` (decrease if needed for stability)
- `BATCH_SIZE=16` (adjust based on GPU memory)

## Technical Details

### Class Weighting Formula
For binary classification with class counts [c0, c1]:
```python
total = c0 + c1
weight_0 = total / (2 * c0)
weight_1 = total / (2 * c1)
```

This gives higher weight to the minority class (abnormal), forcing the model to pay more attention to it during training.

### Metric Computation
```python
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='binary', zero_division=0
)
```
- Uses `average='binary'` to focus on positive class (abnormal = 1)
- `zero_division=0` handles edge cases gracefully

## Verification

To verify the implementation works:
1. Check that `results/dinov2/` contains all 10 PNG/CSV files after training
2. Verify class weights are printed at training start
3. Confirm precision, recall, F1 are logged each epoch
4. Compare new classification report with old one - abnormal recall should be >0.5
