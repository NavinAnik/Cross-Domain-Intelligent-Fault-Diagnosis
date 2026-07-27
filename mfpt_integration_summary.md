# MFPT Integration Summary

MFPT is integrated as a first-class dataset using the **same architecture as CWRU** (CLI scripts, `src/modules/`, `src/experiments/`, `manifest.csv`, `manuscript_io` reporting). No separate framework or YAML config layer was added.

## Files added

| File | Purpose |
|------|---------|
| [scripts/download_mfpt.py](scripts/download_mfpt.py) | Download & extract MFPT `.mat` files |
| [scripts/generate_mfpt_spectrograms.py](scripts/generate_mfpt_spectrograms.py) | CLI wrapper for spectrogram export |
| [src/modules/mfpt_metadata.py](src/modules/mfpt_metadata.py) | Stem → label/load/fault_type/sample_rate |
| [src/modules/mfpt_experiment_utils.py](src/modules/mfpt_experiment_utils.py) | Manifest load, dataset prep, few-shot splits |
| [src/modules/source_model_locator.py](src/modules/source_model_locator.py) | Auto-locate best CWRU ResNet50 checkpoint |
| [src/experiments/export_mfpt_spectrogram_images.py](src/experiments/export_mfpt_spectrogram_images.py) | Log-Mel PNG export + `manifest.csv` |
| [src/experiments/run_mfpt_kfold_cv.py](src/experiments/run_mfpt_kfold_cv.py) | In-domain K-fold CV (grouped + stratified) |
| [src/experiments/run_mfpt_domain_adaptation.py](src/experiments/run_mfpt_domain_adaptation.py) | CWRU→MFPT zero-shot, few-shot, full fine-tune |
| [src/build_mfpt_reviewer_package.py](src/build_mfpt_reviewer_package.py) | Domain-adaptation reviewer package builder |
| [results/domain_adaptation/cwru_pipeline_analysis.md](results/domain_adaptation/cwru_pipeline_analysis.md) | CWRU pipeline reference for MFPT mirror |

## Files modified

None (MFPT reuses existing `cwru_experiment_utils`, `spectrogram_fault_data`, `manuscript_io`, `cwru_split_checks`, and `export_cwru_spectrogram_images` Mel helpers without changing CWRU code).

## Data layout

```
data/
  mfpt/                    # flat .mat files (from download script)
  mfpt_spectrograms/       # normal/ + abnormal/ PNGs + manifest.csv
results/
  domain_adaptation/       # CWRU→MFPT experiments + reviewer package
  mfpt/                    # in-domain MFPT metrics/figures/reports/manifests
```

## Download instructions

From repo root (use project **venv** for training: `venv\Scripts\python.exe` on Windows):

```bash
# Full known-condition MathWorks split (20 files)
python scripts/download_mfpt.py

# Force re-download
python scripts/download_mfpt.py --force

# Small subset for smoke tests
python scripts/download_mfpt.py --preset mathworks_train_only
```

Output: `data/mfpt/*.mat`

## Preprocessing instructions

```bash
# Option A: wrapper script (repo root)
python scripts/generate_mfpt_spectrograms.py

# Option B: module (from src/)
cd src
python -m experiments.export_mfpt_spectrogram_images \
  --mat_dir ../data/mfpt \
  --out_dir ../data/mfpt_spectrograms
```

Outputs:
- `data/mfpt_spectrograms/normal/*.png`
- `data/mfpt_spectrograms/abnormal/*.png`
- `data/mfpt_spectrograms/manifest.csv`
- `data/mfpt_spectrograms/export_meta.json`

Uses the **same log-Mel parameters as CWRU** (window=2048, hop=2048, n_fft=512, hop_length=128, n_mels=128, image_size=224). Sample rate is per-file from MFPT metadata or MAT struct.

## Training instructions (in-domain MFPT)

```bash
cd src
..\venv\Scripts\python.exe -m experiments.run_mfpt_kfold_cv \
  --dataset_path ../data/mfpt_spectrograms \
  --cv_mode both --k 5 --epochs 25
```

Results: `src/results/mfpt/kfold_cv/<timestamp>/`

Single-model training also works via existing entry point:

```bash
..\venv\Scripts\python.exe -m experiments.train_spectrogram_fault \
  --dataset_path ../data/mfpt_spectrograms --model resnet50 --infer_mat_stem_groups
```

## Evaluation instructions

Source model (auto-located from CWRU grouped k-fold):

```bash
cd src
..\venv\Scripts\python.exe -c "from modules.source_model_locator import write_source_model_summary; write_source_model_summary('../results/domain_adaptation/source_model_summary.md')"
```

See [results/domain_adaptation/source_model_summary.md](results/domain_adaptation/source_model_summary.md) for checkpoint path, accuracy, F1, ROC-AUC.

## Domain adaptation experiment instructions

CWRU-trained ResNet50 → MFPT evaluation:

```bash
cd src
..\venv\Scripts\python.exe -m experiments.run_mfpt_domain_adaptation \
  --dataset_path ../data/mfpt_spectrograms \
  --epochs 25 \
  --do_zero_shot --do_few_shot --do_full_finetune
```

Protocols:
| Protocol | Description |
|----------|-------------|
| Zero-shot | CWRU checkpoint evaluated on MFPT, no fine-tuning |
| Few-shot | Fine-tune on 1%, 5%, 10%, 20% of MFPT labels (group-aware) |
| Full fine-tune | Pretrained CWRU init vs training from scratch on MFPT |

Outputs per run (`src/results/domain_adaptation/<timestamp>/`):
- `domain_adaptation.csv` / `.tex`
- `figures/few_shot_accuracy_curve.png`, `domain_adaptation_bars.png`
- `audit_report.md`, `manuscript_text.md`, `summary.json`
- Per-protocol `config.json`, `test_metrics.json`

## Reviewer package

```bash
cd src
..\venv\Scripts\python.exe build_mfpt_reviewer_package.py
```

Output: `src/results/domain_adaptation/reviewer_package/`

## Smoke test results (verified)

| Step | Status |
|------|--------|
| Download (`mathworks_train_only`) | 4 `.mat` files |
| Spectrogram export | 714 PNG windows + manifest |
| Zero-shot domain adaptation | Completed |
| Few-shot + full fine-tune (2 epochs) | Completed |
| Grouped K-fold (k=2, 2 epochs) | Completed |
| Reviewer package build | 0 missing artifacts |

**Note:** Use the project `venv` Python (`venv\Scripts\python.exe`). System Python 3.14 has a torch/torchvision compatibility issue.

## Label mapping (MFPT)

| Class | Stems | Label |
|-------|-------|-------|
| Normal | `baseline_1`, `baseline_2`, `baseline_3` | 0 |
| Abnormal (IR) | `InnerRaceFault_vload_*` | 1 |
| Abnormal (OR) | `OuterRaceFault_*`, `OuterRaceFault_vload_*` | 1 |

Binary task: `normal/` vs `abnormal/` folders (same as CWRU).
