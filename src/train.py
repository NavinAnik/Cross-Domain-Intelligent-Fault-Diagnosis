import os
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    auc,
)
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

try:
    from training.optimization import (
        enable_gradient_checkpointing,
        autocast_context,
        TrainingStepHelper,
    )
    _HAS_OPTIMIZATION = True
except ImportError:
    _HAS_OPTIMIZATION = False

from utils.logging_config import setup_logging
try:
    from utils.experiment_tracking import init_wandb, log_metrics, finish_wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
    def init_wandb(*args, **kwargs):
        return False
    def log_metrics(*args, **kwargs):
        pass
    def finish_wandb():
        pass

# ----------- Config ------------
SAMPLE_RATE = 16000
N_MELS = 64
MAX_FRAMES = 128
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
TEMPERATURE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Memory optimization (used when training.optimization is available)
USE_AMP = os.environ.get("USE_AMP", "true").lower() == "true" and DEVICE.type == "cuda"
GRADIENT_ACCUMULATION_STEPS = max(1, int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1")))
USE_GRADIENT_CHECKPOINTING = os.environ.get("USE_GRADIENT_CHECKPOINTING", "false").lower() == "true"
logger.info("Using device: %s", DEVICE)
if _HAS_OPTIMIZATION:
    logger.info("Optimization: USE_AMP=%s, GRADIENT_ACCUMULATION_STEPS=%s, USE_GRADIENT_CHECKPOINTING=%s", USE_AMP, GRADIENT_ACCUMULATION_STEPS, USE_GRADIENT_CHECKPOINTING)
SEED = 42
# ------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -------- Dataset for contrastive training -----------
class ContrastiveMIMIIDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        self.samples = []
        self.augment = augment

        for machine_type in os.listdir(root_dir):
            machine_type_path = os.path.join(root_dir, machine_type)
            if not os.path.isdir(machine_type_path):
                continue
            for machine_id in os.listdir(machine_type_path):
                machine_id_path = os.path.join(machine_type_path, machine_id)
                if not os.path.isdir(machine_id_path):
                    continue
                for label in ['normal', 'abnormal']:
                    label_path = os.path.join(machine_id_path, label)
                    if os.path.exists(label_path):
                        wav_files = glob.glob(os.path.join(label_path, "*.wav"))
                        self.samples.extend(wav_files)

        self.augment_fn = AudioAugmentation()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path = self.samples[idx]
        waveform, _ = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0)  # mono
        waveform = waveform[:SAMPLE_RATE * 10]

        x1 = self.process(waveform.clone())
        x2 = self.process(waveform.clone())
        return x1, x2

    def process(self, waveform):
        if self.augment:
            waveform = self.augment_fn(waveform)
        mel_spec = librosa.feature.melspectrogram(
            y=waveform.numpy(),
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=1024,
            hop_length=512,
        )
        log_mel = librosa.power_to_db(mel_spec).T

        if log_mel.shape[0] > MAX_FRAMES:
            log_mel = log_mel[:MAX_FRAMES, :]
        elif log_mel.shape[0] < MAX_FRAMES:
            pad = MAX_FRAMES - log_mel.shape[0]
            log_mel = np.pad(log_mel, ((0, pad), (0, 0)), mode='constant')

        return torch.tensor(log_mel, dtype=torch.float32)


# -------- Labeled validation dataset for evaluation ----------
class ValMIMIIDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        label_map = {'normal': 0, 'abnormal': 1}

        for machine_type in os.listdir(root_dir):
            machine_type_path = os.path.join(root_dir, machine_type)
            if not os.path.isdir(machine_type_path):
                continue
            for machine_id in os.listdir(machine_type_path):
                machine_id_path = os.path.join(machine_type_path, machine_id)
                if not os.path.isdir(machine_id_path):
                    continue
                for label_name in ['normal', 'abnormal']:
                    label_path = os.path.join(machine_id_path, label_name)
                    if os.path.exists(label_path):
                        for file in os.listdir(label_path):
                            if file.endswith('.wav'):
                                self.samples.append(os.path.join(label_path, file))
                                self.labels.append(label_map[label_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path = self.samples[idx]
        waveform, _ = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0)  # mono
        waveform = waveform[:SAMPLE_RATE * 10]

        mel_spec = librosa.feature.melspectrogram(
            y=waveform.numpy(),
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=1024,
            hop_length=512,
        )
        log_mel = librosa.power_to_db(mel_spec).T

        if log_mel.shape[0] > MAX_FRAMES:
            log_mel = log_mel[:MAX_FRAMES, :]
        elif log_mel.shape[0] < MAX_FRAMES:
            pad = MAX_FRAMES - log_mel.shape[0]
            log_mel = np.pad(log_mel, ((0, pad), (0, 0)), mode='constant')

        return torch.tensor(log_mel, dtype=torch.float32), self.labels[idx]


# ---------- Transformer Encoder ----------
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=N_MELS, embed_dim=128, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.input_proj(x)  # (B, T, embed_dim)
        x = self.transformer(x)  # (B, T, embed_dim)
        x = x.permute(0, 2, 1)  # (B, embed_dim, T)
        x = self.pool(x).squeeze(-1)  # (B, embed_dim)
        x = self.output_proj(x)
        return x


# ---------- Projection Head ----------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------- NT-Xent Loss ----------
def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(DEVICE)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    positives = torch.cat(
        [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
    ).to(DEVICE)

    logits = sim_matrix / temperature
    labels = positives

    loss = F.cross_entropy(logits, labels)
    return loss


# ---------- Evaluation metrics and plotting ----------
def evaluate(model, val_loader):
    model.eval()
    preds = []
    targets = []
    embeddings = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            emb = model(x)
            embeddings.append(emb.cpu())
            preds.append(emb.cpu())  # Use embeddings for scoring (e.g. anomaly detection)
            targets.append(y.cpu())

    embeddings = torch.cat(embeddings)
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    # Here we treat embedding norm as anomaly score (example)
    scores = torch.norm(preds, dim=1)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    pred_labels = (scores < 0.5).long()  # threshold, tune as needed

    # Confusion matrix
    cm = confusion_matrix(targets, pred_labels)
    logger.info("Confusion Matrix:\n%s", cm)

    # AUC
    auc_score = roc_auc_score(targets, scores)
    logger.info("AUC: %.4f", auc_score)

    # Precision, Recall, F1 Curve data
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    max_f1 = f1_scores.max()
    logger.info("Max F1-score: %.4f", max_f1)

    # Plot ROC curve, Precision-Recall curve, F1 vs threshold
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(thresholds, f1_scores[:-1], label="F1-score")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title("F1-score vs Threshold")
    plt.legend()

    plt.subplot(1, 3, 3)
    # Simple confusion matrix heatmap
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("evaluation_plots.png")

    return auc_score, max_f1


def plot_train_val_curves(train_losses, val_losses):
    epochs = len(train_losses)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, epochs + 1), train_acc, label="Train Accuracy")
    # plt.plot(range(1, epochs + 1), val_acc, label="Val Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.title("Accuracy Curve")

    plt.tight_layout()
    plt.show()


# -------- Audio augmentation helper -------- #
class AudioAugmentation:
    def __init__(self):
        self.augmentations = [
            lambda x: x + 0.005 * torch.randn_like(x),
            lambda x: x * (0.8 + 0.4 * torch.rand(1)),
        ]

    def __call__(self, x):
        aug = random.choice(self.augmentations)
        return aug(x)


# -------- Training loop with validation evaluation -------- #
def train_contrastive_with_eval(train_root, val_root):
    # Experiment tracking (Weights & Biases)
    if _HAS_WANDB:
        init_wandb(
            project="mimii-anomaly-detection",
            config={
                "model": "contrastive_transformer",
                "train_root": train_root,
                "val_root": val_root,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "temperature": TEMPERATURE,
                "use_amp": USE_AMP,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "use_gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
                "seed": SEED,
            },
            name="contrastive-mimii",
        )

    train_dataset = ContrastiveMIMIIDataset(train_root, augment=True)
    val_dataset = ValMIMIIDataset(val_root)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    encoder = TransformerEncoder(input_dim=N_MELS).to(DEVICE)
    projector = ProjectionHead().to(DEVICE)
    if _HAS_OPTIMIZATION and USE_GRADIENT_CHECKPOINTING:
        enable_gradient_checkpointing(encoder)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=LEARNING_RATE
    )
    step_helper = (
        TrainingStepHelper(
            optimizer,
            use_amp=USE_AMP,
            device=DEVICE,
            accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        )
        if _HAS_OPTIMIZATION
        else None
    )

    train_losses = []
    val_losses = []
    train_acc = []  # optional
    val_acc = []  # optional

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        projector.train()
        total_loss = 0
        if step_helper:
            step_helper.zero_grad()
            step_helper.reset_step_count()

        for x1, x2 in train_loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)

            if step_helper and USE_AMP:
                with autocast_context(USE_AMP, DEVICE):
                    z1 = projector(encoder(x1))
                    z2 = projector(encoder(x2))
                    loss = nt_xent_loss(z1, z2, temperature=TEMPERATURE)
                step_helper.step(loss)
            else:
                z1 = projector(encoder(x1))
                z2 = projector(encoder(x2))
                loss = nt_xent_loss(z1, z2, temperature=TEMPERATURE)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x1.size(0)
            logger.info("[Epoch %d] Processed %d samples, Loss: %.4f", epoch, len(train_loader.dataset), loss.item())
        if step_helper:
            step_helper.finalize_step()

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        logger.info("[Epoch %d] Train Contrastive Loss: %.4f", epoch, avg_train_loss)

        # Evaluate on validation set embeddings to get some metric
        auc_score, max_f1 = evaluate(encoder, val_loader)
        log_metrics({
            "train_loss": avg_train_loss,
            "val_auc": auc_score,
            "val_max_f1": max_f1,
        }, step=epoch)

    plot_train_val_curves(train_losses, val_losses)
    finish_wandb()


if __name__ == "__main__":
    setup_logging(log_dir="logs")
    train_contrastive_with_eval("F:\dataset\MIMII_Dataset\\train\\6dB", "F:\dataset\MIMII_Dataset\\val\\-6dB")
