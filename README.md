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

Create a virtual environment and install required packages:

```bash
pip install -r requirements.txt
```

### 2. Pretrain Encoder

```bash
python main_pretrain.py
```

### 3. Fine-Tune Model

```bash
python main_finetune.py
```

### 4. Evaluate

```bash
python evaluate_models.py
```

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
