"""
Automated split / leakage verification for CWRU reviewer-response experiments.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

from modules.cwru_metadata import stem_from_png_basename


def stems_from_paths(paths: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for p in paths:
        base = os.path.basename(p)
        stem = stem_from_png_basename(base)
        if stem:
            out.add(stem)
        else:
            m = re.match(r"^(\d+)_w", base)
            if m:
                out.add(m.group(1))
    return out


def assert_no_file_overlap(train_paths: Sequence[str], val_paths: Sequence[str]) -> Dict[str, Any]:
    tr = stems_from_paths(train_paths)
    va = stems_from_paths(val_paths)
    overlap = tr & va
    return {
        "check": "no_file_overlap",
        "passed": len(overlap) == 0,
        "train_stems_n": len(tr),
        "val_stems_n": len(va),
        "overlap_stems": sorted(overlap),
    }


def assert_folds_disjoint(val_index_sets: Sequence[Sequence[int]]) -> Dict[str, Any]:
    seen: Dict[int, int] = {}
    duplicates: List[int] = []
    for fold_i, idxs in enumerate(val_index_sets):
        for i in idxs:
            if i in seen:
                duplicates.append(int(i))
            else:
                seen[i] = fold_i
    return {
        "check": "folds_disjoint",
        "passed": len(duplicates) == 0,
        "n_folds": len(val_index_sets),
        "duplicate_indices": sorted(set(duplicates))[:50],
    }


def assert_load_disjoint(
    train_stems: Sequence[str],
    test_stems: Sequence[str],
    manifest: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    if manifest is not None and "stem" in manifest.columns and "load_hp" in manifest.columns:
        stem_load = manifest.drop_duplicates("stem").set_index("stem")["load_hp"].to_dict()
        tr_loads = {stem_load.get(s) for s in train_stems if s in stem_load}
        te_loads = {stem_load.get(s) for s in test_stems if s in stem_load}
    else:
        tr_loads = set()
        te_loads = set()
    stem_overlap = set(train_stems) & set(test_stems)
    load_overlap = tr_loads & te_loads
    return {
        "check": "load_disjoint",
        "passed": len(stem_overlap) == 0 and (not load_overlap or load_overlap == set()),
        "stem_overlap": sorted(stem_overlap),
        "train_loads": sorted(x for x in tr_loads if x is not None),
        "test_loads": sorted(x for x in te_loads if x is not None),
        "load_overlap": sorted(x for x in load_overlap if x is not None),
    }


def write_verification_report(
    path: str,
    results: List[Dict[str, Any]],
    *,
    title: str = "CWRU split verification report",
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    md_path = path if path.endswith(".md") else f"{path}.md"
    json_path = md_path.replace(".md", ".json")

    lines = [f"# {title}", ""]
    all_pass = True
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        if not r.get("passed"):
            all_pass = False
        lines.append(f"## {r.get('check', 'check')} — **{status}**")
        for k, v in r.items():
            if k in ("check", "passed"):
                continue
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    lines.append(f"**Overall:** {'PASS' if all_pass else 'FAIL'}")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"passed": all_pass, "results": results}, f, indent=2)
