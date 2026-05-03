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
