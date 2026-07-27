"""
CWRU 12 kHz drive-end bearing metadata keyed by .mat filename stem.

Mappings follow the Case Western Reserve University Bearing Data Center tables:
https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

LOAD_TO_RPM: Dict[int, int] = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

STEM_TO_META: Dict[str, Dict[str, Any]] = {}


def _meta(
    stem: str,
    *,
    label: int,
    load_hp: int,
    fault_type: str,
    diameter_in: Optional[str] = None,
    or_position: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "stem": stem,
        "label": int(label),
        "load_hp": int(load_hp),
        "rpm": LOAD_TO_RPM[int(load_hp)],
        "fault_type": fault_type,
        "diameter_in": diameter_in,
        "or_position": or_position,
    }


def _register_fault_group(
    stems: List[str],
    fault_type: str,
    diameter_in: str,
    or_position: Optional[str] = None,
) -> None:
    if len(stems) != 4:
        raise ValueError(f"Expected 4 stems per fault group, got {stems}")
    for load_hp, stem in enumerate(stems):
        STEM_TO_META[str(stem)] = _meta(
            str(stem),
            label=1,
            load_hp=load_hp,
            fault_type=fault_type,
            diameter_in=diameter_in,
            or_position=or_position,
        )


def _build_registry() -> None:
    if STEM_TO_META:
        return
    for stem, load_hp in [("97", 0), ("98", 1), ("99", 2), ("100", 3)]:
        STEM_TO_META[stem] = _meta(stem, label=0, load_hp=load_hp, fault_type="normal")

    _register_fault_group(["105", "106", "107", "108"], "IR", "0.007")
    _register_fault_group(["118", "119", "120", "121"], "B", "0.007")
    _register_fault_group(["130", "131", "132", "133"], "OR", "0.007", "6")
    _register_fault_group(["144", "145", "146", "147"], "OR", "0.007", "3")
    _register_fault_group(["156", "158", "159", "160"], "OR", "0.007", "12")

    _register_fault_group(["169", "170", "171", "172"], "IR", "0.014")
    _register_fault_group(["185", "186", "187", "188"], "B", "0.014")
    _register_fault_group(["197", "198", "199", "200"], "OR", "0.014", "6")

    _register_fault_group(["209", "210", "211", "212"], "IR", "0.021")
    _register_fault_group(["222", "223", "224", "225"], "B", "0.021")
    _register_fault_group(["234", "235", "236", "237"], "OR", "0.021", "6")
    _register_fault_group(["246", "247", "248", "249"], "OR", "0.021", "3")
    _register_fault_group(["258", "259", "260", "261"], "OR", "0.021", "12")

    _register_fault_group(["3001", "3002", "3003", "3004"], "IR", "0.028")
    _register_fault_group(["3005", "3006", "3007", "3008"], "B", "0.028")


def meta_for_stem(stem: str) -> Optional[Dict[str, Any]]:
    _build_registry()
    s = str(stem).strip().lstrip("Xx")
    return STEM_TO_META.get(s)


def load_for_stem(stem: str) -> Optional[int]:
    m = meta_for_stem(stem)
    return None if m is None else int(m["load_hp"])


def label_for_stem(stem: str) -> Optional[int]:
    m = meta_for_stem(stem)
    return None if m is None else int(m["label"])


def stem_from_png_basename(basename: str) -> Optional[str]:
    import re

    m = re.match(r"^(.+?)_w\d+\.(?:png|jpg|jpeg|bmp|webp)$", basename, re.I)
    return m.group(1) if m else None


def all_known_stems() -> List[str]:
    _build_registry()
    return sorted(STEM_TO_META.keys(), key=lambda s: int(s))


_build_registry()
