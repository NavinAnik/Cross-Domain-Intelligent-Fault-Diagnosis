# Spectrogram fault classification (CNN / attention / hybrid)

End-to-end training for **folder-based spectrogram images** using `ImageDataset` from `run_all_models.py`, SpecAugment-style tensor augmentations, timm CNN backbones, optional spatial attention and CNN+Transformer hybrids, weighted CE or focal loss, and publication-style metrics and figures.

## Environment

- Set **`DATASET_PATH`** to the root folder that `ImageDataset` expects (same layout as your other experiments). You can put it in `src/configs/.env` as `DATASET_PATH=...` or pass `--dataset_path` on the CLI.
- **MIMII spectrograms** (example): `H:/dataset/MIMII_Dataset/spectrograms` with folders `-6dB`, `0dB`, `6dB`, then `fan` / `pump` / `slider` / `valve`, then `normal` / `abnormal` PNGs. With the repo default `.env`, all SNRs load together (groups like `0dB/fan` for grouped splits). To use **one SNR only**, set `DATASET_SUBDIR=0dB` in `.env` or pass `--dataset_subdir 0dB`.
- Install dependencies from `src/requirements.txt` (includes **`timm`**).
- On Windows, dataloader workers default to **`0`** when `--num_workers -1` to avoid multiprocessing hangs.

## Train a single run

From the `src/` directory:

```bash
python -m experiments.train_spectrogram_fault --dataset_path %DATASET_PATH% --model resnet50 --augment strong --loss ce_weighted
```

### Useful flags

| Flag | Description |
|------|-------------|
| `--model` | `resnet50`, `efficientnet_b0`, `cnn_baseline`, `cnn_se`, `cnn_transformer`, or other timm names for pure CNN baseline |
| `--backbone` | timm backbone name when using `cnn_baseline`, `cnn_se`, or `cnn_transformer` (default `resnet50`) |
| `--augment` | `off`, `weak`, `medium`, `strong` (see `modules/spectrogram_fault_data.py`) |
| `--loss` | `ce_weighted` (inverse-frequency weights on train) or `focal` |
| `--no_group_split` | Disable StratifiedGroupKFold when `ImageDataset.groups` is set |
| `--output_dir` | Base directory for runs (default `results/spectrogram_fault`) |
| `--run_id` | Optional subdirectory name; default is a timestamp |

## Ablation grid (model × augmentation × loss)

Writes **`ablation_results_<timestamp>.csv`** and **`.json`** under `--output_dir`.

```bash
python -m experiments.run_spectrogram_ablation --dataset_path %DATASET_PATH% --quick
```

Full grid defaults: models `resnet50`, `efficientnet_b0`, `cnn_se`, `cnn_transformer`; augments `off`, `strong`; losses `ce_weighted`, `focal`. Override with `--models`, `--augments`, `--losses` (comma-separated).

## Output layout per run

Under `results/spectrogram_fault/<run_id>/` (or your `--output_dir` + `run_id`):

| Artifact | Purpose |
|----------|---------|
| `best.pt` | Best validation balanced-accuracy checkpoint |
| `last.pt` | Final weights + metrics blob |
| `config.json` | Resolved CLI args + class counts |
| `metrics.json` | `compute_classification_metrics` + val balanced accuracy + train time (includes precision, recall, F1, AUC) |
| `confusion_matrix.png` | Counts confusion matrix |
| `roc_curve.png` | ROC (binary or per-class for multiclass) |
| `training_curves.png` | Train loss and val balanced accuracy |
| `metric_curves.png` | Per-epoch validation precision, recall, and F1 curves |
| `interpretability/` | Spatial / transformer attention samples; **Grad-CAM** (`grad_cam_*.png`) for models with a `.backbone` |

## Code map

- `modules/spectrogram_cnn_models.py` — `build_model`, baselines and hybrids
- `modules/spectrogram_fault_data.py` — transforms + `get_train_val_indices` (group-aware when possible)
- `modules/spectrogram_augment.py` — time/freq mask, noise, etc.
- `modules/evaluation.py` — `compute_classification_metrics`
- `modules/visualization.py` — `plot_grad_cam_cnn`, `compute_grad_cam_cnn` (ViT path: `plot_grad_cam`)

## CWRU reviewer-response validation suite

Rigorous, leakage-aware evaluation for the Case Western Reserve University (CWRU) 12 kHz drive-end binary task (normal vs abnormal). Uses the same **ResNet50** architecture as `train_spectrogram_fault`; no model changes.

### 1. Download and export (from repo root)

```bash
python scripts/download_cwru_mat.py --preset full_12k_de --out_dir data/cwru_mat

cd src
python -m experiments.export_cwru_spectrogram_images ^
  --mat_dir ../data/cwru_mat --out_dir ../data/cwru_spectrograms_full_png
```

Export writes `normal/` / `abnormal/` PNGs plus **`manifest.csv`** (stem, load_hp, fault metadata per window).

### 2. Full orchestrator (recommended)

```bash
python -m experiments.run_cwru_reviewer_response ^
  --dataset_path ../data/cwru_spectrograms_full_png ^
  --k 5 --epochs 25 --do_kfold --do_cross_load --do_gradcam_physics
```

Publication figures are copied to **`results/manuscript_figures/`** (cross-load heatmap, localization fidelity, fault-frequency overlays, Grad-CAM physics barplots, `manuscript_summary.md`).

### Grad-CAM physics (standalone)

```bash
python -m experiments.evaluate_gradcam_physics ^
  --dataset_path ../data/cwru_spectrograms_full_png ^
  --mat_dir ../data/cwru_mat ^
  --checkpoint results/cwru_reviewer_response/<run>/kfold/grouped/fold_0/best.pt
```

Outputs under `results/cwru_reviewer_response/<timestamp>/`:

| Artifact | Purpose |
|----------|---------|
| `audit_report.md` | Protocol audit + live split verification |
| `manuscript_text.md` | Auto-filled sentences for the paper |
| `run_config.json` | Reproducibility config (seed, k, epochs) |
| `kfold/` | Stratified + grouped K-fold CV (CSV, LaTeX, figures, checkpoints) |
| `cross_load/` | Train/test across motor loads 1–3 HP |
| `tables/`, `figures/` | Aggregated publication assets |

### 3. Individual experiments

```bash
python -m experiments.run_cwru_kfold_cv --dataset_path ../data/cwru_spectrograms_full_png --cv_mode both --k 5 --epochs 25
python -m experiments.run_cwru_cross_load --dataset_path ../data/cwru_spectrograms_full_png --epochs 25
```

- **`--cv_mode grouped`**: recording-wise `StratifiedGroupKFold` (leakage-free; primary evidence).
- **`--cv_mode stratified`**: window-level split (legacy protocol; expect higher scores if leakage was present).
- **`--init_checkpoint`**: optional warm-start from a MIMII or prior CWRU `best.pt`.

### Code map (CWRU)

- `modules/cwru_metadata.py` — official 12k DE stem → load / fault mapping
- `modules/cwru_split_checks.py` — overlap, fold disjointness, load disjointness reports
- `modules/cwru_experiment_utils.py` — shared dataset / training helpers
- `utils/manuscript_io.py` — CSV + LaTeX tables, PNG + PDF figures
- `experiments/run_cwru_kfold_cv.py`, `run_cwru_cross_load.py`, `run_cwru_reviewer_response.py`
- `experiments/evaluate_gradcam_physics.py` — localization fidelity + BPFI/BPFO/BSF overlays
- `modules/cwru_bearing_physics.py` — fault frequency table and mel-axis region masks

