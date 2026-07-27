"""
MFPT bearing fault metadata keyed by .mat filename stem.

Mappings follow the MathWorks known-condition MFPT splits and MFPT Society
documentation: baseline files are normal; InnerRaceFault / OuterRaceFault are abnormal.

https://www.mfpt.org/fault-data-sets/
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Default sample rates from MFPT documentation.
SR_HIGH = 97656  # baseline + constant-load outer race @ 270 lbs
SR_VLOAD = 48828  # variable-load inner/outer race sets

STEM_TO_META: Dict[str, Dict[str, Any]] = {}


def _meta(
    stem: str,
    *,
    label: int,
    load_lbs: Optional[float],
    shaft_hz: float = 25.0,
    fault_type: str = "normal",
    sample_rate: int = SR_VLOAD,
) -> Dict[str, Any]:
    return {
        "stem": stem,
        "label": int(label),
        "load_lbs": load_lbs,
        "shaft_hz": float(shaft_hz),
        "fault_type": fault_type,
        "sample_rate": int(sample_rate),
    }


def _register_baseline(stems: List[str], *, load_lbs: float = 270.0, sample_rate: int = SR_HIGH) -> None:
    for stem in stems:
        STEM_TO_META[str(stem)] = _meta(
            str(stem),
            label=0,
            load_lbs=load_lbs,
            fault_type="normal",
            sample_rate=sample_rate,
        )


def _register_or_constant(stems: List[str], *, load_lbs: float = 270.0) -> None:
    for stem in stems:
        STEM_TO_META[str(stem)] = _meta(
            str(stem),
            label=1,
            load_lbs=load_lbs,
            fault_type="OR",
            sample_rate=SR_HIGH,
        )


def _register_vload_ir(stems: List[str], loads: List[Optional[float]]) -> None:
    if len(stems) != len(loads):
        raise ValueError(f"IR vload stems/loads length mismatch: {stems} vs {loads}")
    for stem, load in zip(stems, loads):
        STEM_TO_META[str(stem)] = _meta(
            str(stem),
            label=1,
            load_lbs=load,
            fault_type="IR",
            sample_rate=SR_VLOAD,
        )


def _register_vload_or(stems: List[str], loads: List[Optional[float]]) -> None:
    if len(stems) != len(loads):
        raise ValueError(f"OR vload stems/loads length mismatch: {stems} vs {loads}")
    for stem, load in zip(stems, loads):
        STEM_TO_META[str(stem)] = _meta(
            str(stem),
            label=1,
            load_lbs=load,
            fault_type="OR",
            sample_rate=SR_VLOAD,
        )


def _build_registry() -> None:
    if STEM_TO_META:
        return

    _register_baseline(["baseline_1", "baseline_2", "baseline_3"], load_lbs=270.0, sample_rate=SR_HIGH)
    _register_or_constant(["OuterRaceFault_1", "OuterRaceFault_2", "OuterRaceFault_3"], load_lbs=270.0)

    _register_vload_ir(
        [f"InnerRaceFault_vload_{i}" for i in range(1, 8)],
        [0.0, 25.0, 50.0, 100.0, 150.0, 200.0, 250.0],
    )
    _register_vload_or(
        [f"OuterRaceFault_vload_{i}" for i in range(1, 8)],
        [25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
    )


def _infer_from_stem(stem: str) -> Optional[Dict[str, Any]]:
    """Heuristic metadata for unknown MFPT stems."""
    s = str(stem).strip()
    sl = s.lower()
    if sl.startswith("baseline"):
        return _meta(s, label=0, load_lbs=270.0, fault_type="normal", sample_rate=SR_HIGH)
    if "innerracefault" in sl.replace("_", ""):
        m = re.search(r"vload[_]?(\d+)", sl)
        load = float(m.group(1)) * 25.0 if m else None
        return _meta(s, label=1, load_lbs=load, fault_type="IR", sample_rate=SR_VLOAD)
    if "outerracefault" in sl.replace("_", ""):
        if "vload" in sl:
            m = re.search(r"vload[_]?(\d+)", sl)
            load = float(m.group(1)) * 25.0 if m else None
            return _meta(s, label=1, load_lbs=load, fault_type="OR", sample_rate=SR_VLOAD)
        return _meta(s, label=1, load_lbs=270.0, fault_type="OR", sample_rate=SR_HIGH)
    return None


def meta_for_stem(stem: str) -> Optional[Dict[str, Any]]:
    _build_registry()
    s = str(stem).strip()
    return STEM_TO_META.get(s) or _infer_from_stem(s)


def load_lbs_for_stem(stem: str) -> Optional[float]:
    m = meta_for_stem(stem)
    return None if m is None else m.get("load_lbs")


def label_for_stem(stem: str) -> Optional[int]:
    m = meta_for_stem(stem)
    return None if m is None else int(m["label"])


def sample_rate_for_stem(stem: str) -> Optional[int]:
    m = meta_for_stem(stem)
    return None if m is None else int(m["sample_rate"])


def stem_from_png_basename(basename: str) -> Optional[str]:
    m = re.match(r"^(.+?)_w\d+\.(?:png|jpg|jpeg|bmp|webp)$", basename, re.I)
    return m.group(1) if m else None


def all_known_stems() -> List[str]:
    _build_registry()
    return sorted(STEM_TO_META.keys())


_build_registry()
