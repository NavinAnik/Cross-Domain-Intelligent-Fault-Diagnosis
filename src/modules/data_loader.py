
import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms  # Not used here but retained for consistency
from sklearn.model_selection import train_test_split, StratifiedKFold

load_dotenv("../configs/.env")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
N_MELS = int(os.getenv("N_MELS", 128))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", 128))


# import os
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Subset
#
# class MIMIIDataset(Dataset):
#     def __init__(self, root_dir, transform=None, mode='train'):
#         self.samples = []
#         self.labels = []
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((224, 224)),  # Resize for CNNs like ResNet
#             transforms.ToTensor(),
#         ])
#         label_map = {'normal': 0, 'abnormal': 1}
#
#         for snr_folder in os.listdir(root_dir):
#             snr_path = os.path.join(root_dir, snr_folder)
#             if not os.path.isdir(snr_path):
#                 continue
#             for machine_type in os.listdir(snr_path):
#                 machine_path = os.path.join(snr_path, machine_type)
#                 for label_name in ['normal', 'abnormal']:
#                     label_path = os.path.join(machine_path, label_name)
#                     if os.path.exists(label_path):
#                         for file in os.listdir(label_path):
#                             if file.endswith('.png'):
#                                 self.samples.append(os.path.join(label_path, file))
#                                 self.labels.append(label_map[label_name])
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         img_path = self.samples[idx]
#         label = self.labels[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, label
#
# def get_dataloaders(dataset_path, batch_size, val_split=0.2):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])
#
#     full_dataset = MIMIIDataset(dataset_path, transform=transform)
#     indices = list(range(len(full_dataset)))
#     train_indices, val_indices = train_test_split(indices, test_size=val_split, stratify=full_dataset.labels)
#
#     train_subset = Subset(full_dataset, train_indices)
#     val_subset = Subset(full_dataset, val_indices)
#
#     train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, val_loader
#
# def get_validation_loader(dataset_path, batch_size):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
#
#     val_dataset = MIMIIDataset(dataset_path, transform=transform, mode='val')
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#     return val_loader


class MIMIIAudioDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        label_map = {'normal': 0, 'abnormal': 1}

        for machine_type in os.listdir(root_dir):
            type_path = os.path.join(root_dir, machine_type)
            if not os.path.isdir(type_path):
                continue

            for machine_id in os.listdir(type_path):
                id_path = os.path.join(type_path, machine_id)
                if not os.path.isdir(id_path):
                    continue

                for label_name in ['normal', 'abnormal']:
                    label_path = os.path.join(id_path, label_name)
                    if os.path.exists(label_path):
                        for file in glob.glob(os.path.join(label_path, "*.wav")):
                            self.samples.append(file)
                            self.labels.append(label_map[label_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]

        # Load audio
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel).T  # Shape: (time, mel)

        # Pad/crop to MAX_FRAMES
        if log_mel.shape[0] > MAX_FRAMES:
            log_mel = log_mel[:MAX_FRAMES]
        elif log_mel.shape[0] < MAX_FRAMES:
            pad = MAX_FRAMES - log_mel.shape[0]
            log_mel = np.pad(log_mel, ((0, pad), (0, 0)))

        return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0), label

def get_audio_dataloaders(root_dir, batch_size=32, val_split=0.2):
    dataset = MIMIIAudioDataset(root_dir)
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=dataset.labels,
        random_state=42
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_validation_audio_loader(root_dir, batch_size=32):
    val_dataset = MIMIIAudioDataset(root_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader


def get_cv_fold_indices(root_dir, n_splits=5, random_state=42):
    """
    Return stratified K-fold train/validation indices for cross-validation.

    Args:
        root_dir: Dataset root (passed to MIMIIAudioDataset).
        n_splits: Number of folds (default 5 for 5-fold CV).
        random_state: Random seed for reproducibility.

    Returns:
        List of (train_indices, val_indices) tuples, one per fold.
        train_indices and val_indices are numpy arrays of dataset indices.
    """
    dataset = MIMIIAudioDataset(root_dir)
    labels = np.array(dataset.labels)
    indices = np.arange(len(dataset))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, val_idx in skf.split(indices, labels):
        folds.append((train_idx, val_idx))
    return folds


def get_audio_dataloaders_cv(
    root_dir,
    fold_index,
    n_splits=5,
    batch_size=32,
    random_state=42,
):
    """
    Get train and validation DataLoaders for one fold of K-fold cross-validation.

    Args:
        root_dir: Dataset root directory.
        fold_index: Which fold (0 to n_splits - 1).
        n_splits: Total number of folds (default 5).
        batch_size: Batch size for both loaders.
        random_state: Seed for StratifiedKFold.

    Returns:
        (train_loader, val_loader) for the requested fold.
    """
    folds = get_cv_fold_indices(root_dir, n_splits=n_splits, random_state=random_state)
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"fold_index must be in [0, {n_splits - 1}], got {fold_index}")
    train_idx, val_idx = folds[fold_index]
    dataset = MIMIIAudioDataset(root_dir)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader