# Spectrogram-image fault classification: methodology, results, and reproducibility

This document aligns the **supervised spectrogram fault** track in this repository with a rigorous experimental write-up style (clear datasets, splits, metrics, limitations, and reproducibility), without borrowing unrelated technical content from other domains.

**Anchor experiment:** run id `20260407_193339` under `src/results/spectrogram_fault/` (see also [`src/paper_assets/ANCHOR_RUN_20260407_193339.md`](../src/paper_assets/ANCHOR_RUN_20260407_193339.md)).

---

## 1. Problem and task

We treat **industrial fault detection** as **image-level classification** on precomputed **spectrogram PNGs**. Each image is assigned a discrete label (binary or multi-class) according to the folder layout consumed by `ImageDataset` in [`src/experiments/run_all_models.py`](../src/experiments/run_all_models.py). The training entry point [`src/experiments/train_spectrogram_fault.py`](../src/experiments/train_spectrogram_fault.py) infers `num_classes` as `max(labels) + 1` over the training index set.

**Anchor run:** binary classification (`num_classes`: 2) on the MIMII spectrogram tree at `H:/dataset/MIMII_Dataset/spectrograms`, all SNR folders combined (`dataset_subdir`: null).

---

## 2. Data protocol

### 2.1 Preprocessing

- Resize to `image_size × image_size` (default **224**).
- **ImageNet** normalization (mean/std) as in [`src/modules/spectrogram_fault_data.py`](../src/modules/spectrogram_fault_data.py) (`get_fault_transforms`).

### 2.2 Augmentation (training only)

Regimes: `off`, `weak`, `medium`, `strong`. For non-`off` settings, training applies light **geometric / photometric** jitter in PIL space, then **tensor-stage** SpecAugment-style operators from [`src/modules/spectrogram_augment.py`](../src/modules/spectrogram_augment.py): **frequency mask**, **time mask**, optional **time shift**, **Gaussian noise**. Validation uses resize, to tensor, normalize only.

**Anchor run:** `augment` = `strong` (largest masks and jitter among the presets).

### 2.3 Train / validation split and leakage control

- Default **20%** validation (`val_split` = 0.2), **stratified** by label.
- When `ImageDataset` provides multiple **groups** (e.g. machine or session keys), [`get_train_val_indices`](../src/modules/spectrogram_fault_data.py) uses the **first fold** of **StratifiedGroupKFold** so that **no group appears in both** train and validation, reducing optimistic bias from correlated clips.
- Ablate leakage control with CLI `--no_group_split` (disables group-aware folding).

**Anchor run:** `group_split` = true.

### 2.4 Optional SNR or sub-corpus restriction

[`resolve_image_dataset_root`](../src/modules/spectrogram_fault_data.py) supports restricting to a child folder (e.g. single SNR) via `--dataset_subdir` or `DATASET_SUBDIR`.

**Anchor run:** no restriction (`dataset_subdir`: null).

---

## 3. Models

Models are built via [`build_model` in `src/modules/spectrogram_cnn_models.py`](../src/modules/spectrogram_cnn_models.py).

| CLI `model` | Description |
|---------------|---------------|
| `resnet50`, `efficientnet_b0`, … | **CNNBaseline:** timm backbone, `global_pool=""`, **global average pooling**, linear classifier. |
| `cnn_se` | **CNNSpatialAttention:** last feature map modulated by **CBAM-style spatial attention** (channel-wise average and max, `7×7` conv, sigmoid) before GAP. |
| `cnn_transformer` | **CNNTransformerHybrid:** feature map reshaped to a token grid, linear projection to `d_model`, **2D sinusoidal positional encodings**, optional **CLS** token, `nn.TransformerEncoder` (GELU, pre-norm), layer norm, linear head. |

**Anchor run:** `model` = `resnet50`, `pretrained` = true — i.e. **CNNBaseline** with ImageNet weights.

Schematic PNGs for the pipeline and for each architecture family are generated under `src/paper_assets/figures/` by `scripts/paper_figures/generate_spectrogram_fault_figures.py`.

### 3.1 Figure inventory (methodology and architectures)

| File | Role |
|------|------|
| [`methodology_pipeline.png`](../src/paper_assets/figures/methodology_pipeline.png) | End-to-end data and training flow. |
| [`architecture_cnn_baseline.png`](../src/paper_assets/figures/architecture_cnn_baseline.png) | **CNNBaseline** (anchor: ResNet50). |
| [`architecture_cnn_spatial_attention.png`](../src/paper_assets/figures/architecture_cnn_spatial_attention.png) | **CNNSpatialAttention** (`cnn_se`). |
| [`architecture_cnn_transformer_hybrid.png`](../src/paper_assets/figures/architecture_cnn_transformer_hybrid.png) | **CNNTransformerHybrid** (`cnn_transformer`). |

Regenerate with: `python scripts/paper_figures/generate_spectrogram_fault_figures.py` from the repository root.

---

## 4. Training objective and optimization

### 4.1 Loss

- **`ce_weighted`:** class weights inversely proportional to frequency on the **training subset** only ([`_make_criterion`](../src/experiments/train_spectrogram_fault.py)).
- **`focal`:** focal loss with class-balanced `alpha` ([`modules/focal_loss.py`](../src/modules/focal_loss.py)).

**Anchor run:** `ce_weighted`.

### 4.2 Optimizer, schedule, and stability

- **AdamW** with weight decay (default 0.05), learning rate default **1e-4**.
- **Linear warmup** for `warmup_epochs` (default 3), then **cosine** decay to a small minimum LR ([`LinearWarmupCosineLR`](../src/experiments/train_spectrogram_fault.py)).
- **Mixed precision** on CUDA when `--amp` (default on); gradient **clipping** (`max_grad_norm` default 1.0).
- **Early stopping** when validation **balanced accuracy** does not improve for `patience` epochs (default 15).
- **Checkpoint:** weights with best validation balanced accuracy saved as `best.pt`.

**Anchor run:** `batch_size` 32, `epochs` 50 (may stop earlier), `seed` 42, `amp` true.

### 4.3 Differentiated elements (honest contribution framing)

1. **Group-aware stratified validation** for grouped industrial spectrograms.
2. **Hybrid architectures** (CNN + spatial attention; CNN + Transformer with 2D sinusoidal PE) in one training stack, alongside standard timm CNNs.
3. **Built-in interpretability exports** after training: Grad-CAM for models exposing `.backbone`; attention heatmaps for `cnn_se` and last-layer attention for `cnn_transformer` ([`train_spectrogram_fault.py`](../src/experiments/train_spectrogram_fault.py)).

The **spectrogram_fault** path is **supervised** end-to-end; contrastive pretraining in other modules is a **separate** line of work unless you explicitly run and cite it.

---

## 5. Evaluation metrics

Final metrics use [`compute_classification_metrics`](../src/modules/evaluation.py) on the **validation** split, plus **balanced accuracy**. Full numbers are written to `metrics.json` per run.

---

## 6. Results (anchor run `20260407_193339`)

### 6.1 Metrics table

| Metric | Value |
|--------|------:|
| Validation balanced accuracy | 0.9108 |
| Accuracy | 0.9482 |
| Macro precision | 0.9240 |
| Macro recall | 0.9108 |
| Macro F1 | 0.9172 |
| MCC | 0.8347 |
| Cohen's kappa | 0.8344 |
| Sensitivity | 0.8488 |
| Specificity | 0.9728 |
| ROC AUC (binary positive class) | 0.9758 |
| PR AUC (positive class) | 0.9409 |
| Brier score | 0.0403 |
| ECE | 0.0089 |
| Train time (s) | 30666.3 |

*Source: `src/results/spectrogram_fault/20260407_193339/metrics.json` on the training machine.*

### 6.2 Figures (training script output)

Mirrored for version control and manuscript insertion under:

**[`src/paper_assets/figures/from_anchor_run_20260407_193339/`](../src/paper_assets/figures/from_anchor_run_20260407_193339/)**

| File | Description |
|------|-------------|
| `training_curves.png` | Training loss and validation balanced accuracy per epoch. |
| `metric_curves.png` | Validation macro precision, recall, and F1 per epoch. |
| `roc_curve.png` | ROC curve with AUC. |
| `confusion_matrix.png` | Confusion matrix on validation. |
| `grad_cam_0.png` … `grad_cam_3.png` | Example Grad-CAM overlays (ResNet backbone). |
| `config.json`, `metrics.json` | Snapshots of the anchor run for version control alongside figures. |

### 6.3 Broader comparisons

If you ran [`src/experiments/run_spectrogram_ablation.py`](../src/experiments/run_spectrogram_ablation.py), add bar plots or tables from `ablation_results_*.csv` / `.json` under your chosen `output_dir`. No ablation CSV is bundled for the anchor run alone.

---

## 7. Limitations (template in the spirit of thorough empirical papers)

1. **Domain shift:** Combining all SNRs and machines improves data volume but may **hide** failure modes on a single operating condition; report per-SNR or per-machine splits when the research question requires it.
2. **Group definition:** `StratifiedGroupKFold` is only as sound as the **group** labels from `ImageDataset`; incorrect grouping can still leak information or over-fragment training.
3. **Supervision cost:** Labels follow the folder structure; **noisy labels** or **ambiguous** normal/abnormal boundaries are not modeled here.
4. **Interpretability:** Grad-CAM explains **CNN backbone** evidence; it does not replace causal validation on the physical process.
5. **Baselines:** Strong pure CNN baselines (e.g. the same ResNet50 setup) may match or beat hybrids on some corpora—report **honest negatives** from your ablation grid.

---

## 8. Reproducibility checklist

1. **Environment:** Python dependencies in [`src/requirements.txt`](../src/requirements.txt); **`timm`** required for spectrogram CNN models ([`README_SPECTROGRAM_FAULT.md`](../src/experiments/README_SPECTROGRAM_FAULT.md)).
2. **Data path:** Set `DATASET_PATH` or pass `--dataset_path` (see anchor `config.json`).
3. **Command (anchor configuration):**

```bash
cd src
python -m experiments.train_spectrogram_fault ^
  --dataset_path H:/dataset/MIMII_Dataset/spectrograms ^
  --model resnet50 ^
  --augment strong ^
  --loss ce_weighted ^
  --batch_size 32 ^
  --epochs 50 ^
  --lr 0.0001 ^
  --weight_decay 0.05 ^
  --warmup_epochs 3 ^
  --patience 15 ^
  --val_split 0.2 ^
  --seed 42 ^
  --run_id 20260407_193339_replica
```

Omit line continuations on Unix; use `\` instead of `^`. Add `--no_group_split` only if reproducing the non-group ablation.

4. **Windows dataloader:** `--num_workers -1` selects **0** workers by default on Windows to avoid multiprocessing hangs ([`default_num_workers`](../src/modules/spectrogram_fault_data.py)).
5. **Artifacts:** Each run writes `config.json`, `metrics.json`, plots, checkpoints, and `interpretability/` as documented in [`README_SPECTROGRAM_FAULT.md`](../src/experiments/README_SPECTROGRAM_FAULT.md).

---

## 9. What changed vs a minimal ImageNet-transfer baseline

- **SpecAugment-style** tensor augmentations and staged strength presets.
- **Stratified group** validation option wired to the same `ImageDataset` groups used elsewhere in the repo.
- **Class-imbalance-aware** losses (weighted CE, focal).
- **Extended model zoo** (spatial attention, CNN+Transformer hybrid) with a single `build_model` factory.
- **Automatic** ROC, confusion matrix, training curves, and optional Grad-CAM / attention figures for the paper supplement.
