"""
Multi-Dataset Loader for Industrial Anomaly Detection.

Supports:
- MIMII (Malfunctioning Industrial Machine Investigation and Inspection)
- ToyADMOS (Toy Anomaly Detection with Machine Operating Sounds)
- DCASE 2020/2021/2022 Task 2 (Anomalous Sound Detection)

This enables comprehensive evaluation across multiple benchmarks for Q1 publication.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Dataset Class
# ---------------------------------------------------------------------------

class AnomalyDetectionDataset(Dataset):
    """
    Base class for anomaly detection audio datasets.
    
    Handles:
    - Audio file loading and spectrogram conversion
    - Consistent preprocessing across datasets
    - Label encoding (0=normal, 1=abnormal)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        target_sr: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        max_duration: float = 10.0,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_samples = int(max_duration * target_sr)
        
        self.samples: List[Tuple[str, int]] = []
        self.dataset_name = "base"
        
        self._setup_mel_transform()

    def _setup_mel_transform(self):
        """Setup mel spectrogram transform."""
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio file and convert to mono."""
        waveform, sr = torchaudio.load(path)
        
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            padding = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform

    def _to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram."""
        mel = self.mel_transform(waveform)
        mel_db = self.amplitude_to_db(mel)
        
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        
        return mel_db

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        
        waveform = self._load_audio(path)
        spectrogram = self._to_spectrogram(waveform)
        
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, label

    def get_labels(self) -> List[int]:
        """Return all labels for stratified splitting."""
        return [label for _, label in self.samples]


# ---------------------------------------------------------------------------
# MIMII Dataset
# ---------------------------------------------------------------------------

class MIMIIDataset(AnomalyDetectionDataset):
    """
    MIMII Dataset: Malfunctioning Industrial Machine Investigation and Inspection.
    
    Structure:
        MIMII_Dataset/
        ├── fan/
        │   ├── id_00/
        │   │   ├── normal/
        │   │   └── abnormal/
        │   ├── id_02/
        │   └── ...
        ├── pump/
        ├── slider/
        └── valve/
    """

    MACHINE_TYPES = ["fan", "pump", "slider", "valve"]

    def __init__(
        self,
        root_dir: str,
        machine_type: Optional[str] = None,
        machine_ids: Optional[List[str]] = None,
        split: str = "train",
        val_ratio: float = 0.2,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(root_dir, split, **kwargs)
        self.dataset_name = "MIMII"
        self.machine_type = machine_type
        self.machine_ids = machine_ids
        
        self._scan_files()
        
        if split in ("train", "val"):
            self._split_data(val_ratio, random_state)

    def _scan_files(self):
        """Scan directory for audio files."""
        audio_extensions = {".wav", ".WAV"}
        
        machine_types = [self.machine_type] if self.machine_type else self.MACHINE_TYPES
        
        for m_type in machine_types:
            m_path = self.root_dir / m_type
            if not m_path.exists():
                continue
            
            for m_id in os.listdir(m_path):
                if self.machine_ids and m_id not in self.machine_ids:
                    continue
                
                id_path = m_path / m_id
                if not id_path.is_dir():
                    continue
                
                for label_name, label in [("normal", 0), ("abnormal", 1)]:
                    label_path = id_path / label_name
                    if not label_path.exists():
                        continue
                    
                    for f in label_path.iterdir():
                        if f.suffix in audio_extensions:
                            self.samples.append((str(f), label))
        
        logger.info("MIMII: Found %d samples", len(self.samples))

    def _split_data(self, val_ratio: float, random_state: int):
        """Split into train/val sets."""
        labels = self.get_labels()
        indices = list(range(len(self.samples)))
        
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, stratify=labels, random_state=random_state
        )
        
        if self.split == "train":
            self.samples = [self.samples[i] for i in train_idx]
        else:
            self.samples = [self.samples[i] for i in val_idx]


# ---------------------------------------------------------------------------
# ToyADMOS Dataset
# ---------------------------------------------------------------------------

class ToyADMOSDataset(AnomalyDetectionDataset):
    """
    ToyADMOS Dataset: Toy Anomaly Detection with Machine Operating Sounds.
    
    Structure:
        ToyADMOS/
        ├── ToyConveyor/
        │   ├── case1/
        │   │   ├── NormalSound_IND/
        │   │   │   └── *.wav
        │   │   └── AnomalousSound_IND/
        │   │       └── *.wav
        │   └── case2/
        └── ToyCar/
            └── ...
    
    or
        ToyADMOS/
        ├── train/
        │   └── *.wav (normal only)
        └── test/
            └── *.wav (mixed normal/anomaly, with anomaly_ prefix)
    """

    MACHINE_TYPES = ["ToyConveyor", "ToyCar"]

    def __init__(
        self,
        root_dir: str,
        machine_type: Optional[str] = None,
        split: str = "train",
        val_ratio: float = 0.2,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(root_dir, split, **kwargs)
        self.dataset_name = "ToyADMOS"
        self.machine_type = machine_type
        
        self._scan_files()
        
        if split in ("train", "val") and len(self.samples) > 0:
            self._split_data(val_ratio, random_state)

    def _scan_files(self):
        """Scan for ToyADMOS audio files."""
        audio_extensions = {".wav", ".WAV"}
        
        if (self.root_dir / "train").exists():
            self._scan_standard_structure()
        else:
            self._scan_case_structure()

    def _scan_standard_structure(self):
        """Scan train/test directory structure."""
        train_dir = self.root_dir / "train"
        test_dir = self.root_dir / "test"
        
        for f in train_dir.iterdir():
            if f.suffix.lower() == ".wav":
                self.samples.append((str(f), 0))
        
        for f in test_dir.iterdir():
            if f.suffix.lower() == ".wav":
                label = 1 if "anomaly" in f.name.lower() else 0
                self.samples.append((str(f), label))

    def _scan_case_structure(self):
        """Scan ToyConveyor/case1/NormalSound structure."""
        machine_types = [self.machine_type] if self.machine_type else self.MACHINE_TYPES
        
        for m_type in machine_types:
            m_path = self.root_dir / m_type
            if not m_path.exists():
                continue
            
            for case_dir in m_path.iterdir():
                if not case_dir.is_dir():
                    continue
                
                normal_dirs = ["NormalSound_IND", "normal", "Normal"]
                for nd in normal_dirs:
                    np = case_dir / nd
                    if np.exists():
                        for f in np.iterdir():
                            if f.suffix.lower() == ".wav":
                                self.samples.append((str(f), 0))
                
                anomaly_dirs = ["AnomalousSound_IND", "abnormal", "Anomalous"]
                for ad in anomaly_dirs:
                    ap = case_dir / ad
                    if ap.exists():
                        for f in ap.iterdir():
                            if f.suffix.lower() == ".wav":
                                self.samples.append((str(f), 1))
        
        logger.info("ToyADMOS: Found %d samples", len(self.samples))

    def _split_data(self, val_ratio: float, random_state: int):
        """Split into train/val sets."""
        labels = self.get_labels()
        if len(set(labels)) < 2:
            return
        
        indices = list(range(len(self.samples)))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, stratify=labels, random_state=random_state
        )
        
        if self.split == "train":
            self.samples = [self.samples[i] for i in train_idx]
        else:
            self.samples = [self.samples[i] for i in val_idx]


# ---------------------------------------------------------------------------
# DCASE Dataset
# ---------------------------------------------------------------------------

class DCASEDataset(AnomalyDetectionDataset):
    """
    DCASE Challenge Task 2 Dataset: Anomalous Sound Detection.
    
    Supports DCASE 2020, 2021, 2022 formats.
    
    Structure (DCASE 2020/2021):
        DCASE2020/
        ├── dev_data/
        │   ├── fan/
        │   │   ├── train/
        │   │   │   └── normal_id_00_*.wav
        │   │   └── test/
        │   │       └── anomaly_id_00_*.wav
        │   └── ...
        └── eval_data/
    
    Structure (DCASE 2022):
        DCASE2022/
        ├── dev_data/
        │   ├── bearing/
        │   │   ├── train/
        │   │   │   └── section_00_source_train_normal_*.wav
        │   │   └── test/
        │   │       └── section_00_source_test_*.wav
    """

    MACHINE_TYPES_2020 = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
    MACHINE_TYPES_2021 = ["fan", "gearbox", "pump", "slider", "ToyCar", "ToyTrain", "valve"]
    MACHINE_TYPES_2022 = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]

    def __init__(
        self,
        root_dir: str,
        year: int = 2020,
        machine_type: Optional[str] = None,
        data_type: str = "dev",
        split: str = "train",
        val_ratio: float = 0.2,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(root_dir, split, **kwargs)
        self.dataset_name = f"DCASE{year}"
        self.year = year
        self.machine_type = machine_type
        self.data_type = data_type
        
        if year == 2020:
            self.machine_types = self.MACHINE_TYPES_2020
        elif year == 2021:
            self.machine_types = self.MACHINE_TYPES_2021
        else:
            self.machine_types = self.MACHINE_TYPES_2022
        
        self._scan_files()
        
        if split in ("train", "val") and len(self.samples) > 0:
            labels = self.get_labels()
            if len(set(labels)) >= 2:
                self._split_data(val_ratio, random_state)

    def _scan_files(self):
        """Scan DCASE directory structure."""
        data_dir = self.root_dir / f"{self.data_type}_data"
        if not data_dir.exists():
            data_dir = self.root_dir
        
        machine_types = [self.machine_type] if self.machine_type else self.machine_types
        
        for m_type in machine_types:
            m_path = data_dir / m_type
            if not m_path.exists():
                continue
            
            train_path = m_path / "train"
            if train_path.exists():
                for f in train_path.iterdir():
                    if f.suffix.lower() == ".wav":
                        self.samples.append((str(f), 0))
            
            test_path = m_path / "test"
            if test_path.exists():
                for f in test_path.iterdir():
                    if f.suffix.lower() == ".wav":
                        label = 1 if "anomaly" in f.name.lower() else 0
                        self.samples.append((str(f), label))
        
        logger.info("DCASE%d: Found %d samples", self.year, len(self.samples))

    def _split_data(self, val_ratio: float, random_state: int):
        """Split into train/val sets."""
        labels = self.get_labels()
        indices = list(range(len(self.samples)))
        
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, stratify=labels, random_state=random_state
        )
        
        if self.split == "train":
            self.samples = [self.samples[i] for i in train_idx]
        else:
            self.samples = [self.samples[i] for i in val_idx]


# ---------------------------------------------------------------------------
# Multi-Dataset Loader
# ---------------------------------------------------------------------------

class MultiDatasetLoader:
    """
    Unified loader for multiple anomaly detection datasets.
    
    Enables:
    - Cross-dataset evaluation
    - Combined training across datasets
    - Consistent preprocessing
    """

    DATASET_CLASSES = {
        "mimii": MIMIIDataset,
        "toyadmos": ToyADMOSDataset,
        "dcase2020": lambda **kw: DCASEDataset(year=2020, **kw),
        "dcase2021": lambda **kw: DCASEDataset(year=2021, **kw),
        "dcase2022": lambda **kw: DCASEDataset(year=2022, **kw),
    }

    def __init__(
        self,
        dataset_configs: Dict[str, Dict[str, Any]],
        batch_size: int = 32,
        num_workers: int = 4,
        use_weighted_sampler: bool = True,
    ):
        """
        Args:
            dataset_configs: Dict of dataset_name -> {root_dir: ..., **kwargs}
            batch_size: Batch size for dataloaders.
            num_workers: Number of workers.
            use_weighted_sampler: Whether to use weighted sampling for imbalanced data.
        """
        self.dataset_configs = dataset_configs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_weighted_sampler = use_weighted_sampler
        
        self.datasets: Dict[str, Dict[str, AnomalyDetectionDataset]] = {}

    def load_datasets(self) -> Dict[str, Dict[str, AnomalyDetectionDataset]]:
        """Load all configured datasets."""
        for name, config in self.dataset_configs.items():
            dataset_type = name.lower().replace("-", "").replace("_", "")
            
            if dataset_type.startswith("dcase"):
                year = int(dataset_type[-4:])
                dataset_cls = lambda **kw: DCASEDataset(year=year, **kw)
            elif dataset_type in self.DATASET_CLASSES:
                dataset_cls = self.DATASET_CLASSES[dataset_type]
            else:
                logger.warning("Unknown dataset type: %s", name)
                continue
            
            train_ds = dataset_cls(split="train", **config)
            val_ds = dataset_cls(split="val", **config)
            
            self.datasets[name] = {"train": train_ds, "val": val_ds}
            
            train_labels = train_ds.get_labels()
            val_labels = val_ds.get_labels()
            logger.info(
                "%s: train=%d (normal=%d, abnormal=%d), val=%d",
                name, len(train_ds),
                train_labels.count(0), train_labels.count(1),
                len(val_ds)
            )
        
        return self.datasets

    def get_combined_loaders(
        self, split: str = "train"
    ) -> Tuple[DataLoader, List[str]]:
        """
        Get combined dataloader across all datasets.
        
        Returns:
            DataLoader, list of dataset names (for tracking).
        """
        datasets_list = []
        dataset_names = []
        
        for name, ds_dict in self.datasets.items():
            if split in ds_dict:
                datasets_list.append(ds_dict[split])
                dataset_names.append(name)
        
        if not datasets_list:
            raise ValueError(f"No datasets found for split: {split}")
        
        combined = ConcatDataset(datasets_list)
        
        sampler = None
        shuffle = True
        
        if self.use_weighted_sampler and split == "train":
            all_labels = []
            for ds in datasets_list:
                all_labels.extend(ds.get_labels())
            
            class_counts = Counter(all_labels)
            total = sum(class_counts.values())
            weights = [total / (2 * class_counts[l]) for l in all_labels]
            sampler = WeightedRandomSampler(weights, len(weights))
            shuffle = False
        
        loader = DataLoader(
            combined,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        return loader, dataset_names

    def get_per_dataset_loaders(self) -> Dict[str, Dict[str, DataLoader]]:
        """Get separate dataloaders for each dataset."""
        loaders = {}
        
        for name, ds_dict in self.datasets.items():
            loaders[name] = {}
            
            for split, ds in ds_dict.items():
                sampler = None
                shuffle = split == "train"
                
                if self.use_weighted_sampler and split == "train":
                    labels = ds.get_labels()
                    if len(set(labels)) >= 2:
                        class_counts = Counter(labels)
                        total = sum(class_counts.values())
                        weights = [total / (2 * class_counts[l]) for l in labels]
                        sampler = WeightedRandomSampler(weights, len(weights))
                        shuffle = False
                
                loaders[name][split] = DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    shuffle=shuffle if sampler is None else False,
                    sampler=sampler,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
        
        return loaders


def get_multi_dataset_loaders(
    mimii_path: Optional[str] = None,
    toyadmos_path: Optional[str] = None,
    dcase_path: Optional[str] = None,
    dcase_year: int = 2020,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Convenience function to create multi-dataset loaders.
    
    Returns:
        train_loader, val_loader, dataset_info
    """
    configs = {}
    
    if mimii_path and os.path.exists(mimii_path):
        configs["mimii"] = {"root_dir": mimii_path}
    
    if toyadmos_path and os.path.exists(toyadmos_path):
        configs["toyadmos"] = {"root_dir": toyadmos_path}
    
    if dcase_path and os.path.exists(dcase_path):
        configs[f"dcase{dcase_year}"] = {"root_dir": dcase_path}
    
    if not configs:
        raise ValueError("No valid dataset paths provided")
    
    loader = MultiDatasetLoader(
        dataset_configs=configs,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    loader.load_datasets()
    
    train_loader, train_datasets = loader.get_combined_loaders("train")
    val_loader, val_datasets = loader.get_combined_loaders("val")
    
    info = {
        "datasets": list(configs.keys()),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
    }
    
    return train_loader, val_loader, info
