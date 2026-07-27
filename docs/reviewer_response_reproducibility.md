# Reviewer Response: Implementation Details and Reproducibility

**Comment:** *The paper does not provide enough implementation details for reproducibility. Important settings such as augmentation probabilities, random seeds, scheduler strategies, and training configurations should be included more clearly.*

**Response:** We thank the reviewer for this suggestion. We have added an explicit **Implementation Details** subsection to the manuscript (mirrored in [`docs/spectrogram_fault_paper.md`](spectrogram_fault_paper.md) §2.1, §2.2, §4.2–§4.5) and provide the anchor-run settings below. All values are taken from run `20260407_193339`; the full CLI snapshot is version-controlled in [`src/paper_assets/figures/from_anchor_run_20260407_193339/config.json`](../src/paper_assets/figures/from_anchor_run_20260407_193339/config.json).

---

## Training configuration (anchor run)

| Setting | Value |
|---------|-------|
| Model | ResNet50 (`CNNBaseline`, ImageNet-pretrained via timm) |
| Input size | 224×224 RGB |
| Optimizer | AdamW, learning rate **1×10⁻⁴**, weight decay **0.05** |
| Batch size | **32** |
| Max epochs | **50** (best checkpoint at epoch 18; early stopping active) |
| Loss | Weighted cross-entropy: class weight \(w_c = n / (C \cdot n_c)\) on the **training** subset only |
| AMP | Enabled on CUDA |
| Gradient clipping | `max_grad_norm = 1.0` |
| Early stopping | Patience **15** epochs on validation **balanced accuracy** |
| Checkpoint | Best validation balanced accuracy → `best.pt` |
| Train/validation split | **20%** validation, stratified; **group-aware** (`StratifiedGroupKFold`, first fold) |
| Groups | SNR/machine keys from `ImageDataset` (e.g. `0dB/fan`) |
| Normalization | ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |
| Random seed | **42** (see seed protocol below) |
| Train / val counts | 86,304 / 19,510 samples |

**Reproduction command:**

```bash
cd src
python -m experiments.train_spectrogram_fault \
  --dataset_path <MIMII_spectrograms_root> \
  --model resnet50 --augment strong --loss ce_weighted \
  --batch_size 32 --epochs 50 --lr 0.0001 --weight_decay 0.05 \
  --warmup_epochs 3 --patience 15 --val_split 0.2 --seed 42
```

Each run writes `config.json` and `metrics.json` under `src/results/spectrogram_fault/<run_id>/`.

---

## Learning-rate scheduler

We use a custom **`LinearWarmupCosineLR`** scheduler ([`train_spectrogram_fault.py`](../src/experiments/train_spectrogram_fault.py)), not PyTorch’s built-in `CosineAnnealingLR`.

For each parameter group with base learning rate \(\eta_0\):

1. **Warmup** (epochs \(e = 1, \ldots, W\), anchor \(W = 3\)):
   \[
   \eta(e) = \eta_0 \cdot \frac{e}{W}
   \]
2. **Cosine decay** (epochs \(e = W+1, \ldots, E\), anchor \(E = 50\)):
   \[
   t = \frac{e - W}{E - W}, \quad
   \eta(e) = \eta_{\min} + (\eta_0 - \eta_{\min}) \cdot \tfrac{1}{2}\left(1 + \cos(\pi t)\right)
   \]
   with \(\eta_{\min} = 1 \times 10^{-7}\).

The scheduler is stepped **once at the start of each epoch** (before the training loop for that epoch). Anchor: \(\eta_0 = 10^{-4}\), \(W = 3\), \(E = 50\).

---

## Data augmentation (`augment = strong`)

Augmentations are defined in [`get_fault_transforms`](../src/modules/spectrogram_fault_data.py). We deliberately **omit** horizontal flip and rotation because they break time–frequency structure on spectrograms ([`spectrogram_augment.py`](../src/modules/spectrogram_augment.py)).

### Application rule (probabilities)

| Operator | When applied |
|----------|----------------|
| `FrequencyMask`, `TimeMask`, `TimeShift`, `GaussianNoiseTensor` | **Always** on every training sample (no `RandomApply` wrapper) |
| `RandomAffine`, `ColorJitter` | **Stochastic**: torchvision draws new transform parameters on each call (transform always runs; parameters vary) |
| Mask width / shift magnitude | Uniform random within `max_mask_pct` / `max_shift_pct` |

Validation: resize → to tensor → ImageNet normalize only.

### `strong` preset (anchor run)

**PIL stage (before normalize):**

| Transform | Parameters |
|-----------|------------|
| `RandomAffine` | `degrees=0`, `translate=(0.04, 0.02)`, `scale=(0.95, 1.05)` |
| `ColorJitter` | `brightness=0.3`, `contrast=0.3` |

**Tensor stage (after normalize):**

| Transform | Parameters |
|-----------|------------|
| `FrequencyMask` | `max_mask_pct=0.20`, `num_masks=2` |
| `TimeMask` | `max_mask_pct=0.20`, `num_masks=2` |
| `TimeShift` | `max_shift_pct=0.08` (circular roll along time) |
| `GaussianNoiseTensor` | `std=0.01` |

### Other presets (supplementary)

| Preset | PIL | Tensor-stage |
|--------|-----|--------------|
| `off` | resize only | none |
| `weak` | `RandomAffine(translate=(0.02,0.01), scale=(0.98,1.02))` | `FrequencyMask(0.10,1)`, `TimeMask(0.10,1)`, `GaussianNoiseTensor(0.005)` |
| `medium` | `RandomAffine(translate=(0.03,0.01), scale=(0.97,1.03))`, `ColorJitter(0.2,0.2)` | `FrequencyMask(0.15,2)`, `TimeMask(0.15,2)`, `TimeShift(0.05)`, `GaussianNoiseTensor(0.005)` |

---

## Random seeds and split protocol

| Component | Seed usage |
|-----------|------------|
| PyTorch / NumPy | `torch.manual_seed(42)`, `np.random.seed(42)` at process start |
| Train/val index split | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` — **first fold** held out as validation |
| DataLoader shuffle | PyTorch RNG after `manual_seed` |
| Augmentation randomness | Python `random` and PyTorch RNG in transform operators |

**Limitation:** The anchor trainer does not set `random.seed`, `cuda.manual_seed_all`, or `cudnn.deterministic=True`. We therefore fix the data split and primary RNGs with `seed=42`, but **bitwise-identical GPU results are not guaranteed** across hardware or driver versions.

---

## MIMII spectrogram inputs

The anchor experiment trains on **precomputed spectrogram PNGs** (not raw WAV at train time). Folder layout:

```
<MIMII_spectrograms_root>/
  <-6dB|0dB|6dB>/
    <fan|pump|slider|valve>/
      normal/*.png
      abnormal/*.png
```

Groups for leakage-aware splitting are SNR/machine-type keys (e.g. `0dB/fan`). The anchor run loads **all SNR folders** (`dataset_subdir`: null), yielding 105,814 images before the 80/20 grouped split.

PNG generation is external to the training script. The repository’s MIMII **audio** mel pipeline ([`modules/data_loader.py`](../src/modules/data_loader.py)) uses **16 kHz** sampling, **128** mel bands, `librosa.feature.melspectrogram` followed by `power_to_db`, consistent with the log-mel → min–max → RGB PNG export pattern documented for CWRU in [`export_cwru_spectrogram_images.py`](../src/experiments/export_cwru_spectrogram_images.py) (`n_fft=512`, `hop_length=128`, bilinear resize to 224). Training-time preprocessing is resize to 224 and ImageNet normalization only.

---

## Artifacts for independent verification

| Artifact | Location |
|----------|----------|
| Full hyperparameter snapshot | `src/paper_assets/figures/from_anchor_run_20260407_193339/config.json` |
| Metrics | `src/paper_assets/figures/from_anchor_run_20260407_193339/metrics.json` |
| Methodology write-up | `docs/spectrogram_fault_paper.md` |
| Training entry point | `src/experiments/train_spectrogram_fault.py` |
