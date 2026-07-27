"""
CWRU 12 kHz drive-end bearing fault frequency physics (SKF 6205-2RS JEM).

Used for Grad-CAM localization fidelity and fault-frequency overlay figures.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.cwru_metadata import LOAD_TO_RPM

# CWRU Bearing Data Center drive-end geometry (6205-2RS JEM)
N_BALLS = 9
BALL_DIA_IN = 0.3126
PITCH_DIA_IN = 1.537
CONTACT_ANGLE_DEG = 0.0

# Validated multipliers vs published CWRU tables (shaft frequency fr = rpm/60)
MULT_BPFI = 5.4152
MULT_BPFO = 3.5848
MULT_BSF = 2.3567
MULT_FTF = 0.3983
MULT_BSF2 = 2.0 * MULT_BSF  # 4.7135, commonly observed ball defect line

DEFAULT_SR = 12000
DEFAULT_N_FFT = 512
DEFAULT_N_MELS = 128
DEFAULT_HOP_LENGTH = 128
DEFAULT_IMAGE_SIZE = 224


def shaft_frequency_hz(rpm: float) -> float:
    return float(rpm) / 60.0


def fault_frequencies(rpm: float) -> Dict[str, float]:
    """Return BPFI, BPFO, BSF, BSF2, FTF in Hz for a given shaft speed."""
    fr = shaft_frequency_hz(rpm)
    return {
        "fr": fr,
        "BPFI": fr * MULT_BPFI,
        "BPFO": fr * MULT_BPFO,
        "BSF": fr * MULT_BSF,
        "BSF2": fr * MULT_BSF2,
        "FTF": fr * MULT_FTF,
    }


def fault_frequencies_table(loads: Sequence[int] = (0, 1, 2, 3)) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for load_hp in loads:
        rpm = LOAD_TO_RPM[int(load_hp)]
        freqs = fault_frequencies(rpm)
        rows.append(
            {
                "load_hp": int(load_hp),
                "rpm": float(rpm),
                "fr": freqs["fr"],
                "BPFI": freqs["BPFI"],
                "BPFO": freqs["BPFO"],
                "BSF": freqs["BSF"],
                "BSF2": freqs["BSF2"],
                "FTF": freqs["FTF"],
            }
        )
    return rows


def save_fault_frequencies_csv(path: str, loads: Sequence[int] = (0, 1, 2, 3)) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = fault_frequencies_table(loads)
    headers = ["load_hp", "rpm", "fr", "BPFI", "BPFO", "BSF", "BSF2", "FTF"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow({k: round(row[k], 4) if k not in ("load_hp", "rpm") else row[k] for k in headers})
    return path


def class_to_fault_lines(fault_type: str) -> List[str]:
    ft = str(fault_type).upper().strip()
    if ft == "IR":
        return ["BPFI"]
    if ft == "OR":
        return ["BPFO"]
    if ft == "B":
        return ["BSF2", "BSF"]
    return []


def mel_bin_centers_hz(
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
    fmax: Optional[float] = None,
) -> np.ndarray:
    import librosa

    fmax = fmax or sr / 2.0
    return librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fmax)


def freq_to_col(
    f_hz: float,
    width: int,
    *,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
    fmax: Optional[float] = None,
) -> float:
    """Map frequency (Hz) to horizontal column index in exported mel PNG (width = mel axis)."""
    centers = mel_bin_centers_hz(sr=sr, n_mels=n_mels, fmax=fmax)
    idx = int(np.argmin(np.abs(centers - float(f_hz))))
    return float(idx) / max(n_mels - 1, 1) * (width - 1)


def _cols_for_freq_band(
    f_hz: float,
    width: int,
    tol_frac: float,
    *,
    sr: int,
    n_mels: int,
) -> np.ndarray:
    col = freq_to_col(f_hz, width, sr=sr, n_mels=n_mels)
    half = max(1.0, tol_frac * width)
    cols = np.arange(width, dtype=np.float32)
    return np.abs(cols - col) <= half


def fault_region_mask(
    width: int,
    rpm: float,
    fault_type: str,
    *,
    tol_frac: float = 0.05,
    n_harmonics: int = 3,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
) -> np.ndarray:
    """
    Boolean mask (width,) for mel-frequency columns corresponding to fault lines
    (fundamental + harmonics within +/- tol_frac of image width).
    """
    mask = np.zeros(width, dtype=bool)
    freqs = fault_frequencies(rpm)
    lines = class_to_fault_lines(fault_type)
    if not lines:
        return mask
    nyquist = sr / 2.0
    for line in lines:
        f0 = freqs.get(line, 0.0)
        for h in range(1, n_harmonics + 1):
            fh = f0 * h
            if fh >= nyquist:
                continue
            mask |= _cols_for_freq_band(fh, width, tol_frac, sr=sr, n_mels=n_mels)
    return mask


def fault_region_mask_2d(
    height: int,
    width: int,
    rpm: float,
    fault_type: str,
    **kwargs,
) -> np.ndarray:
    """2D mask (H, W): fault columns span full time axis."""
    col_mask = fault_region_mask(width, rpm, fault_type, **kwargs)
    return np.tile(col_mask[np.newaxis, :], (height, 1))


def energy_in_freq_bands(
    cam: np.ndarray,
    rpm: float,
    bands: Sequence[str] = ("BPFI", "BPFO", "BSF"),
    tol_frac: float = 0.05,
    *,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
) -> Dict[str, float]:
    """Fraction of normalized CAM energy within +/- tol around each fault line."""
    cam = np.asarray(cam, dtype=np.float64)
    if cam.ndim == 3:
        cam = cam.mean(axis=-1)
    cam = cam - cam.min()
    total = cam.sum()
    if total <= 1e-12:
        return {b: 0.0 for b in bands}

    h, w = cam.shape
    freqs = fault_frequencies(rpm)
    out: Dict[str, float] = {}
    for band in bands:
        f_hz = freqs.get(band, 0.0)
        col_mask = _cols_for_freq_band(f_hz, w, tol_frac, sr=sr, n_mels=n_mels)
        region = np.tile(col_mask[np.newaxis, :], (h, 1))
        out[band] = float(cam[region].sum() / total)
    return out


def localization_fidelity(
    cam: np.ndarray,
    rpm: float,
    fault_type: str,
    *,
    top_percent: float = 0.20,
    tol_frac: float = 0.05,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
) -> float:
    """
    Localization Fidelity = activated pixels inside fault region / total activated pixels.
    Activated = top ``top_percent`` of normalized CAM values.
    """
    cam = np.asarray(cam, dtype=np.float64)
    if cam.ndim == 3:
        cam = cam.mean(axis=-1)
    cam_n = cam - cam.min()
    mx = cam_n.max()
    if mx <= 1e-12:
        return 0.0
    cam_n = cam_n / mx
    thresh = np.quantile(cam_n, 1.0 - top_percent)
    activated = cam_n >= thresh
    n_act = int(activated.sum())
    if n_act == 0:
        return 0.0
    h, w = cam.shape
    region = fault_region_mask_2d(h, w, rpm, fault_type, tol_frac=tol_frac, sr=sr, n_mels=n_mels)
    if not region.any():
        return 0.0
    return float((activated & region).sum() / n_act)
