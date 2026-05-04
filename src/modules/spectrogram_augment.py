"""
Spectrogram-specific augmentation transforms for mel-spectrogram images.

Standard image augmentations (horizontal flip, rotation) are inappropriate for
spectrograms because they break time-frequency structure. This module provides
augmentations that respect spectrogram semantics:

- SpecAugment (frequency masking + time masking) -- Park et al., Interspeech 2019
- Time shifting (circular shift along time axis)
- Frequency shifting (circular shift along frequency axis)
- Spectrogram-aware mixup

These operate on PIL Images or Tensors representing mel-spectrogram visualizations.
"""

import random
import torch
import numpy as np
from torchvision import transforms


class FrequencyMask:
    """Mask a random contiguous band of frequency bins (vertical stripe on spectrogram).

    Equivalent to SpecAugment frequency masking applied to image-format spectrograms.

    Args:
        max_mask_pct: Maximum fraction of frequency axis to mask (0.0 to 1.0).
        num_masks: Number of frequency masks to apply.
        fill_value: Value to fill masked region (0 = black).
    """

    def __init__(self, max_mask_pct: float = 0.15, num_masks: int = 2,
                 fill_value: float = 0.0):
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.fill_value = fill_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, freq_bins, _ = x.shape
        for _ in range(self.num_masks):
            mask_width = random.randint(1, max(1, int(freq_bins * self.max_mask_pct)))
            start = random.randint(0, max(0, freq_bins - mask_width))
            x[:, start:start + mask_width, :] = self.fill_value
        return x


class TimeMask:
    """Mask a random contiguous span of time steps (horizontal stripe on spectrogram).

    Equivalent to SpecAugment time masking applied to image-format spectrograms.

    Args:
        max_mask_pct: Maximum fraction of time axis to mask (0.0 to 1.0).
        num_masks: Number of time masks to apply.
        fill_value: Value to fill masked region (0 = black).
    """

    def __init__(self, max_mask_pct: float = 0.15, num_masks: int = 2,
                 fill_value: float = 0.0):
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.fill_value = fill_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, _, time_steps = x.shape
        for _ in range(self.num_masks):
            mask_width = random.randint(1, max(1, int(time_steps * self.max_mask_pct)))
            start = random.randint(0, max(0, time_steps - mask_width))
            x[:, :, start:start + mask_width] = self.fill_value
        return x


class TimeShift:
    """Circular shift along the time axis.

    Args:
        max_shift_pct: Maximum shift as fraction of time axis (0.0 to 0.5).
    """

    def __init__(self, max_shift_pct: float = 0.1):
        self.max_shift_pct = max_shift_pct

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        time_steps = x.shape[-1]
        shift = random.randint(
            -int(time_steps * self.max_shift_pct),
            int(time_steps * self.max_shift_pct))
        return torch.roll(x, shifts=shift, dims=-1)


class FrequencyShift:
    """Circular shift along the frequency axis.

    Args:
        max_shift_pct: Maximum shift as fraction of freq axis (0.0 to 0.1).
    """

    def __init__(self, max_shift_pct: float = 0.05):
        self.max_shift_pct = max_shift_pct

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        freq_bins = x.shape[-2]
        shift = random.randint(
            -int(freq_bins * self.max_shift_pct),
            int(freq_bins * self.max_shift_pct))
        return torch.roll(x, shifts=shift, dims=-2)


class GaussianNoiseTensor:
    """Add Gaussian noise to a tensor.

    Args:
        std: Standard deviation of noise.
    """

    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


def get_spectrogram_transforms(image_size: int = 224,
                               strength: str = "medium"):
    """Get spectrogram-aware train/val transforms.

    Unlike generic image transforms, these avoid:
    - Horizontal flip (reverses time axis semantics)
    - Large rotations (breaks freq-time alignment)
    - Hue jitter (irrelevant for spectrograms)

    Instead applies SpecAugment-style masking and spectrogram-specific shifts.

    Args:
        image_size: Target size for resize.
        strength: "weak", "medium", or "strong".

    Returns:
        Dict with 'train' and 'val' transforms.
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if strength == "weak":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            FrequencyMask(max_mask_pct=0.10, num_masks=1),
            TimeMask(max_mask_pct=0.10, num_masks=1),
        ])
    elif strength == "strong":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.02),
                                    scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            FrequencyMask(max_mask_pct=0.20, num_masks=2),
            TimeMask(max_mask_pct=0.20, num_masks=2),
            TimeShift(max_shift_pct=0.1),
            FrequencyShift(max_shift_pct=0.05),
            GaussianNoiseTensor(std=0.01),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.10)),
        ])
    else:  # medium (default)
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.01),
                                    scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            FrequencyMask(max_mask_pct=0.15, num_masks=2),
            TimeMask(max_mask_pct=0.15, num_masks=2),
            TimeShift(max_shift_pct=0.05),
            GaussianNoiseTensor(std=0.005),
        ])

    return {"train": train_transform, "val": val_transform}
