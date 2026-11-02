# Cross-Domain Intelligent Fault Diagnosis

[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41598--026--61771--5-blue)](https://doi.org/10.1038/s41598-026-61771-5)

Official code for the paper published in **Scientific Reports** (2026):

> **Cross-domain intelligent fault diagnosis with superlet spectrograms and a parameter-efficient CNN-Transformer hybrid**
> S. M. Navin Nayer Anik, Md. Ehsanul Haque, Fahmid Al Farid, Mahe Zabin, Jia Uddin, Hezerul Bin Abdul Karim
> *Scientific Reports* (2026). https://www.nature.com/articles/s41598-026-61771-5

## 🧠 Method Overview

Industrial acoustic monitoring is framed as supervised image classification on **superlet time-frequency spectrograms**.
The model is a **CNN-Transformer hybrid**: an ImageNet-pretrained ResNet-50 backbone, a lightweight Transformer encoder over the spatial feature tokens, and **spatial attention gating**.
An **Asymmetric Focal Contrastive Learning (AFCL)** objective handles severe class imbalance.
Main evaluation is on **MIMII** under a group-aware (leakage-free) split; cross-domain transfer is evaluated on **CWRU** (zero-shot and fine-tuning).

## 📊 Key Results (from the paper)

| Setting | Metric | Value |
|---|---|---|
| MIMII | Accuracy | 94.82% |
| MIMII | Balanced accuracy | 91.08% |
| MIMII | F1-score | 91.72% |
| MIMII | ROC-AUC | 0.9758 |
| CWRU (zero-shot) | AUC | 0.8886 |
| CWRU (fine-tuned, 3 epochs) | AUC | 1.0000 |
| Efficiency | Params / FLOPs / latency | 25.6M / 4.1G / 4.5 ms per sample |

## 🗺️ Paper → Code

| Paper component | Code |
|---|---|
| CNN-Transformer hybrid, spatial attention | `src/modules/spectrogram_cnn_models.py` (`CNNTransformerHybrid`, `SpatialAttention2D`, `build_model`) |
| AFCL loss | `src/modules/afcl_loss.py` (`AsymmetricFocalContrastiveLoss`, `create_afcl_loss`) |
| Spectrogram training / evaluation | `src/experiments/train_spectrogram_fault.py`, `src/experiments/eval_spectrogram_checkpoint.py` |
| Ablations & AFCL sensitivity | `src/experiments/run_spectrogram_ablation.py`, `src/experiments/run_afcl_sensitivity.py` |
| Cross-validation & cross-domain | `src/experiments/run_cwru_kfold_cv.py`, `run_cwru_cross_load.py`, `run_mfpt_kfold_cv.py`, `run_mfpt_domain_adaptation.py` |
| Statistical significance | `src/experiments/run_wilcoxon_significance.py` |
| Grad-CAM / physics check | `src/experiments/evaluate_gradcam_physics.py` |
| Efficiency (params, FLOPs, latency, ONNX, quantization, pruning) | `scripts/run_model_efficiency.py`, `scripts/analyze_model_complexity.py`, `scripts/benchmark_runtime.py` |

## 📁 Project Structure

```
Fault-Diagnosis/
├── Dockerfile, docker-compose.yml
├── scripts/            # Efficiency: complexity, runtime, ONNX export, FP16/INT8 quantization, pruning
├── results/            # Generated plots and tables
└── src/
    ├── configs/.env    # Local config (create it; not committed)
    ├── modules/        # Models (spectrogram CNNs, transformer encoders), losses (AFCL, focal, contrastive), data loaders
    ├── fusion/         # Multimodal fusion (early, late, cross-attention, ConFu, ProMoBal, HTSF)
    ├── experiments/    # Experiment runners (training, CV, ablations, cross-domain, figures, tables)
    ├── training/       # Optimization helpers
    ├── utils/          # Logging, tracking, plotting
    └── main_*.py       # Standalone single-model scripts
```

## 🚀 Getting Started

### 1. Install Dependencies

**Run all commands from the `src` directory** (or set `PYTHONPATH` to include `src` when running from repo root).

```bash
cd src
pip install -r requirements.txt
```

Optional: for transformer models only, use `pip install -r requirements_transformers.txt` instead.

### 2. Configuration

Create `src/configs/.env` with at least:

- `DATASET_PATH` – path to your spectrogram / MIMII-style dataset
- Optional: `DATASET_SUBDIR` (e.g. `0dB` for MIMII), `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `NUM_CLASSES`, `USE_AMP`, `GRADIENT_ACCUMULATION_STEPS`, `USE_GRADIENT_CHECKPOINTING`

Command-line flags override these values.

## 🔁 Reproducing the Paper

```bash
cd src

# Train the CNN-Transformer hybrid (ResNet-50 backbone) on spectrograms
python -m experiments.train_spectrogram_fault --dataset_path $DATASET_PATH --model cnn_transformer --backbone resnet50

# Ablations and AFCL sensitivity
python -m experiments.run_spectrogram_ablation --dataset_path $DATASET_PATH
python -m experiments.run_afcl_sensitivity --sweep all

# CWRU: k-fold CV (window-level and leakage-free recording-level) and cross-load generalization
python -m experiments.run_cwru_kfold_cv --dataset_path ../data/cwru_spectrograms_full_png --cv_mode both --k 5 --epochs 25
python -m experiments.run_cwru_cross_load --dataset_path ../data/cwru_spectrograms_full_png --epochs 25

# MFPT: k-fold CV
python -m experiments.run_mfpt_kfold_cv --dataset_path ../data/mfpt_spectrograms --cv_mode both --k 5 --epochs 25

# Efficiency report (params, FLOPs, latency)
python ../scripts/run_model_efficiency.py --help
```

Use `--help` on any runner for the full option list.

---

## Additional Experiments (transformer encoders and multimodal fusion)

### Single-model training (DINOv2, MAE, Swin, BEiT, DeiT, CLIP, AST)

**One model:**

```bash
cd src
python -m experiments.run_all_models --model dinov2
python -m experiments.run_all_models --model mae --epochs 20 --batch_size 16
python -m experiments.run_all_models --model ast --dataset_path C:\data\mimii
```

Options: `--model` (required), `--epochs`, `--batch_size`, `--lr`, `--dataset_path`, `--output_dir`, `--results_dir`, `--name`.  
**Run all models in sequence:**

```bash
python -m experiments.run_all_models --run_all --epochs 20 --batch_size 16
```

**Alternative runner (all 7 models with summary JSON):**

```bash
python -m experiments.run_single_model_experiments --epochs 20 --output_dir model --results_dir results
python -m experiments.run_single_model_experiments --models dinov2,mae,ast --dataset_path C:\data\mimii
```

### Fusion experiments (early, late, cross-attention, ConFu, ProMoBal)

**All fusion strategies:**

```bash
python -m experiments.run_fusion_experiments --fusion all
```

**Specific fusions:**

```bash
python -m experiments.run_fusion_experiments --fusion early late confu --epochs 20 --dataset_path C:\data\mimii
```

Options: `--fusion` (space-separated or `all`), `--dataset_path`, `--epochs`, `--batch_size`, `--lr`, `--output_dir`, `--results_dir`, `--name`.

### Ablation studies

```bash
# Fusion ablations (need audio + image paths)
python -m experiments.run_ablations --mode fusion --audio_path C:\data\audio --image_path C:\data\images --results_dir results

# Contrastive loss ablations
python -m experiments.run_ablations --mode contrastive --dataset_path C:\data\mimii --results_dir results

# Modality ablations (audio-only vs vision-only vs multimodal)
python -m experiments.run_ablations --mode modality --audio_path C:\data\audio --image_path C:\data\images --results_dir results

# Model-architecture ablations
python -m experiments.run_ablations --mode models --dataset_path C:\data\mimii --results_dir results

# Full ablations (fusion + contrastive + modality + models)
python -m experiments.run_ablations --mode all --audio_path C:\data\audio --image_path C:\data\images --dataset_path C:\data\mimii --results_dir results
```

Optional: `--epochs`, `--batch_size`, `--models dinov2,mae,ast`, `--no_models`, `--no_fusion`, `--no_contrastive`, `--no_modality`.

### Cross-validation and statistical analysis

**Analyze existing CV results (e.g. from a JSON):**

```bash
python -m experiments.run_cv_and_statistical_analysis analyze --input path/to/cv_results.json --output_dir results --confidence 0.95
```

**CV mode** (requires integrating your train/eval function with the script’s CV helpers):

```bash
python -m experiments.run_cv_and_statistical_analysis cv --dataset_path C:\data\mimii --n_splits 5
```

### Publication tables and figures

**Generate comparison tables (Markdown/CSV):**

```bash
python -m experiments.generate_tables --results_dir results --output_dir tables_out
python -m experiments.generate_tables --demo
```

**Generate publication figures (plan Figures 1–9):**

```bash
python -m experiments.generate_publication_figures --results_dir results --output_dir publication_figures
python -m experiments.generate_publication_figures --demo
```

### Other entry points

- **Hyperparameter search:**  
  `python -m experiments.hyperparameter_search --help`
- **Comparative analysis:**  
  `python -m experiments.comparative_analysis --help`
- **Standalone model scripts (single-model training):**  
  `main_dinov2.py`, `main_mae.py`, `main_clip.py`, `main_ast.py` (run from `src`; they use `configs/.env` and `DATASET_PATH`).


## 📝 Citation

If you use this code, please cite:

> Anik, S.M.N.N., Haque, M.E., Al Farid, F. et al. Cross-domain intelligent fault diagnosis with superlet spectrograms and a parameter-efficient CNN-Transformer hybrid. *Sci Rep* (2026). https://doi.org/10.1038/s41598-026-61771-5

Received 30 April 2026 · Accepted 07 July 2026 · Published 13 July 2026

```bibtex
@article{anik2026crossdomain,
  title   = {Cross-domain intelligent fault diagnosis with superlet spectrograms and a parameter-efficient CNN-Transformer hybrid},
  author  = {Anik, S. M. Navin Nayer and Haque, Md. Ehsanul and Al Farid, Fahmid and Zabin, Mahe and Uddin, Jia and Karim, Hezerul Bin Abdul},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {10.1038/s41598-026-61771-5},
  url     = {https://www.nature.com/articles/s41598-026-61771-5}
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
