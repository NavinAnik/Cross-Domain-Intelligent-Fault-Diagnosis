# Contrastive Self-Supervised Learning

This repository implements a contrastive self-supervised learning framework using PyTorch. It allows for self-supervised pretraining of an encoder network followed by supervised fine-tuning for classification tasks.

## 📁 Project Structure

```
contrastive_self-supervised_learning-main/
│
├── main_pretrain.py            # Script for contrastive self-supervised pretraining
├── main_finetune.py            # Script for supervised fine-tuning
├── evaluate_models.py          # Evaluation script
├── train.py                    # General training pipeline
├── train.bat                   # Windows batch script for training
├── requirements.txt            # Python dependencies
├── encoder_pretrained.pth      # Pretrained encoder weights
├── encoder_epoch_1.pth         # Encoder checkpoint
├── model_epoch_1.pth           # Model checkpoint
│
├── modules/                    # Core model components
│   ├── encoder.py              # Encoder architecture
│   ├── classifier.py           # Classifier head
│   ├── contrastive_loss.py     # Contrastive loss function
│   ├── data_loader.py          # Data loading utilities
│   └── evaluation.py           # Evaluation metrics
│
└── utils/                      # Utility functions
    ├── plot_graph.py           # Graph plotting (e.g., loss curves)
    └── utils.py                # Miscellaneous utilities
```

## 🚀 Getting Started

### 1. Install Dependencies

**Run all commands from the `src` directory** (or set `PYTHONPATH` to include `src` when running from repo root).

Create a virtual environment and install required packages:

```bash
cd src
pip install -r requirements.txt
```

Optional: for transformer models only, you can use `pip install -r requirements_transformers.txt` instead of the full `requirements.txt`.

### 2. Configuration

Create `src/configs/.env` with at least:

- `DATASET_PATH` – path to your MIMII-style dataset (folder with `normal/` and `abnormal/` or machine-type subfolders)
- Optional: `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `NUM_CLASSES`, `USE_AMP`, `GRADIENT_ACCUMULATION_STEPS`, `USE_GRADIENT_CHECKPOINTING`

Main scripts and experiment runners read this file. You can override paths from the command line (see below).

### 3. Basic Pipeline (pretrain → finetune → evaluate)

```bash
cd src
python main_pretrain.py
python main_finetune.py
python evaluate_models.py
```

Or on Windows, from repo root:

```bat
train.bat
```

(Ensure `train.bat` runs from a directory where `requirements.txt` and `main_pretrain.py` are found, or edit it to `cd src` first.)

---

## How to Run (SOTA Transformers & Q1 Publication Plan)

The following commands implement the **SOTA Transformers Q1 Publication** pipeline (see your Cursor plan for full methodology). Run them from **`src`** so that `configs/.env` and imports resolve correctly.

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

## 🧠 Method Overview

The self-supervised phase trains an encoder using a contrastive loss function, encouraging the model to learn useful representations without labeled data. These representations are then fine-tuned using a classification head with labeled data.

## 📊 Visualization

The training process includes graph plotting utilities located in `utils/plot_graph.py`.

## 📦 Checkpoints

Pretrained weights are included:
- `encoder_pretrained.pth`
- `encoder_epoch_1.pth`
- `model_epoch_1.pth`

## 📄 License

MIT License (or replace with your actual license)
