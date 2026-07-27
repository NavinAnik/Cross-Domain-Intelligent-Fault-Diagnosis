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

#### MIMII spectrogram PNGs (anchor data source)

The anchor run consumes **precomputed spectrogram PNGs**, not raw WAV at train time. Expected folder layout (see [`ImageDataset`](../src/experiments/run_all_models.py)):

```
<MIMII_spectrograms_root>/
  <-6dB|0dB|6dB>/
    <fan|pump|slider|valve>/
      normal/*.png
      abnormal/*.png
```

**Anchor:** `H:/dataset/MIMII_Dataset/spectrograms`, all SNR folders (`dataset_subdir`: null). Groups for leakage-aware splitting are SNR/machine-type keys (e.g. `0dB/fan`).

PNG generation is performed **before** training (no MIMII export script ships in this repo). The repository’s MIMII **audio** mel convention in [`modules/data_loader.py`](../src/modules/data_loader.py) uses **16 kHz** sampling and **128** mel bands via `librosa.feature.melspectrogram` + `power_to_db`. The CWRU export script [`export_cwru_spectrogram_images.py`](../src/experiments/export_cwru_spectrogram_images.py) documents the same log-mel → min–max → bilinear RGB PNG pattern (`n_fft=512`, `hop_length=128`, resize to 224) used for other corpora in this project.

#### Training-time preprocessing

- Resize to `image_size × image_size` (default **224**).
- **ImageNet** normalization (mean/std) as in [`src/modules/spectrogram_fault_data.py`](../src/modules/spectrogram_fault_data.py) (`get_fault_transforms`):
  - mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.

### 2.2 Augmentation (training only)

Regimes: `off`, `weak`, `medium`, `strong` (CLI `--augment`, default `strong`). Defined in [`get_fault_transforms`](../src/modules/spectrogram_fault_data.py). For non-`off` settings, training applies light **geometric / photometric** jitter in PIL space, then **tensor-stage** SpecAugment-style operators from [`src/modules/spectrogram_augment.py`](../src/modules/spectrogram_augment.py). Validation uses resize, to tensor, normalize only.

**Anchor run:** `augment` = `strong`.

We **omit** horizontal flip and rotation (physically invalid for spectrograms; see module docstring in `spectrogram_augment.py`).

#### Application rule (what “probability” means here)

| Operator | Application |
|----------|-------------|
| `FrequencyMask`, `TimeMask`, `TimeShift`, `GaussianNoiseTensor` | **Always applied** on every training sample (no `RandomApply` wrapper) |
| `RandomAffine`, `ColorJitter` | **Stochastic** — torchvision samples new parameters each call |
| Mask width / shift magnitude | Uniform random within `max_mask_pct` / `max_shift_pct` |

#### `strong` preset (anchor)

**PIL stage:**

| Transform | Parameters |
|-----------|------------|
| `RandomAffine` | `degrees=0`, `translate=(0.04, 0.02)`, `scale=(0.95, 1.05)` |
| `ColorJitter` | `brightness=0.3`, `contrast=0.3` |

**Tensor stage (after normalize):**

| Transform | Parameters |
|-----------|------------|
| `FrequencyMask` | `max_mask_pct=0.20`, `num_masks=2` |
| `TimeMask` | `max_mask_pct=0.20`, `num_masks=2` |
| `TimeShift` | `max_shift_pct=0.08` |
| `GaussianNoiseTensor` | `std=0.01` |

#### Other presets (supplementary)

| Preset | PIL | Tensor-stage |
|--------|-----|--------------|
| `off` | resize only | none |
| `weak` | `RandomAffine(translate=(0.02,0.01), scale=(0.98,1.02))` | `FrequencyMask(0.10,1)`, `TimeMask(0.10,1)`, `GaussianNoiseTensor(0.005)` |
| `medium` | `RandomAffine(translate=(0.03,0.01), scale=(0.97,1.03))`, `ColorJitter(0.2,0.2)` | `FrequencyMask(0.15,2)`, `TimeMask(0.15,2)`, `TimeShift(0.05)`, `GaussianNoiseTensor(0.005)` |

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
- **Learning-rate schedule:** custom [`LinearWarmupCosineLR`](../src/experiments/train_spectrogram_fault.py) (not PyTorch `CosineAnnealingLR`). For base LR \(\eta_0\), warmup length \(W\) (default **3**), total epochs \(E\) (default **50**), minimum LR \(\eta_{\min} = 10^{-7}\):
  - Epochs \(1 \ldots W\): \(\eta(e) = \eta_0 \cdot (e / W)\) (linear warmup from 0).
  - Epochs \(W+1 \ldots E\): cosine decay from \(\eta_0\) to \(\eta_{\min}\) via
    \(t = (e - W)/(E - W)\), \(\eta(e) = \eta_{\min} + (\eta_0 - \eta_{\min}) \cdot \tfrac{1}{2}(1 + \cos(\pi t))\).
  - `scheduler.step()` is called **once at the start of each epoch** (before the training loop).
- **Mixed precision** on CUDA when `--amp` (default on); gradient **clipping** (`max_grad_norm` default 1.0).
- **Early stopping** when validation **balanced accuracy** does not improve for `patience` epochs (default 15).
- **Checkpoint:** weights with best validation balanced accuracy saved as `best.pt`.

**Anchor run:** `batch_size` 32, `epochs` 50 (best checkpoint at epoch 18), `seed` 42, `amp` true.

### 4.4 Consolidated training hyperparameters (anchor run `20260407_193339`)

Full snapshot: [`src/paper_assets/figures/from_anchor_run_20260407_193339/config.json`](../src/paper_assets/figures/from_anchor_run_20260407_193339/config.json).

| Setting | Value |
|---------|-------|
| Model | `resnet50` (ImageNet-pretrained `CNNBaseline`) |
| Input size | 224×224 RGB |
| Optimizer | AdamW, lr **1×10⁻⁴**, weight decay **0.05** |
| Batch size | **32** |
| Max epochs | **50** (early stopping; best at epoch 18) |
| Warmup epochs | **3** |
| Min LR (cosine floor) | **1×10⁻⁷** |
| Loss | `ce_weighted` (\(w_c = n / (C \cdot n_c)\) on train labels) |
| Augmentation | `strong` (see §2.2) |
| AMP | enabled |
| Gradient clipping | `max_grad_norm = 1.0` |
| Early stopping | patience **15** on val **balanced accuracy** |
| Val split | **20%**, stratified, group-aware (`StratifiedGroupKFold`, first fold) |
| Seed | **42** |
| Train / val counts | 86,304 / 19,510 |
| DataLoader workers | 0 on Windows, 2 elsewhere (`--num_workers -1`) |

### 4.5 Random seeds and reproducibility limits

| Component | Controlled by `seed=42` |
|-----------|---------------------------|
| PyTorch / NumPy RNG | `torch.manual_seed(42)`, `np.random.seed(42)` at startup ([`main()`](../src/experiments/train_spectrogram_fault.py)) |
| Train/val index split | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` — first fold as validation ([`get_train_val_indices`](../src/modules/spectrogram_fault_data.py)) |
| DataLoader shuffle | PyTorch RNG after `manual_seed` |
| Augmentation | Python `random` and PyTorch in transform operators |

**Caveat:** The anchor trainer does not set `random.seed`, `cuda.manual_seed_all`, or `cudnn.deterministic=True` (those are used in the CWRU reviewer suite via [`set_global_seeds`](../src/modules/cwru_experiment_utils.py)). The data split and primary RNGs are fixed, but **bitwise-identical GPU runs are not guaranteed** across hardware or driver versions.

A copy-paste reviewer response with the same content is in [`docs/reviewer_response_reproducibility.md`](reviewer_response_reproducibility.md).

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
6. **Reviewer response:** [`docs/reviewer_response_reproducibility.md`](reviewer_response_reproducibility.md) — copy-paste implementation details for the reproducibility comment.

---

## 9. What changed vs a minimal ImageNet-transfer baseline

- **SpecAugment-style** tensor augmentations and staged strength presets.
- **Stratified group** validation option wired to the same `ImageDataset` groups used elsewhere in the repo.
- **Class-imbalance-aware** losses (weighted CE, focal).
- **Extended model zoo** (spatial attention, CNN+Transformer hybrid) with a single `build_model` factory.
- **Automatic** ROC, confusion matrix, training curves, and optional Grad-CAM / attention figures for the paper supplement.
