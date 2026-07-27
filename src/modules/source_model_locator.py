"""
Locate the best CWRU ResNet50 checkpoint for MFPT domain-adaptation experiments.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _results_roots() -> List[Path]:
    root = _repo_root()
    return [
        root / "src" / "results",
        root / "results",
    ]


def find_cwru_kfold_runs() -> List[Path]:
    """Return directories containing grouped k-fold folds_grouped.csv."""
    out: List[Path] = []
    for base in _results_roots():
        if not base.is_dir():
            continue
        for csv_path in base.glob("cwru_reviewer_response/*/kfold/grouped/folds_grouped.csv"):
            out.append(csv_path.parent)
    return sorted(set(out), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def pick_best_fold_from_csv(folds_csv: Path) -> Tuple[int, Dict[str, Any]]:
    df = pd.read_csv(folds_csv)
    if "accuracy" not in df.columns:
        raise ValueError(f"No accuracy column in {folds_csv}")
    acc = pd.to_numeric(df["accuracy"], errors="coerce")
    best_i = int(acc.idxmax())
    row = df.iloc[best_i].to_dict()
    fold = int(row.get("fold", best_i))
    return fold, row


def find_best_cwru_checkpoint(
    *,
    prefer_run: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pick highest-accuracy grouped k-fold checkpoint from completed CWRU runs.

    Returns dict with checkpoint path, fold, metrics, folds_csv, run_dir.
    """
    candidates: List[Tuple[float, Path, int, Dict[str, Any]]] = []

    for run_dir in find_cwru_kfold_runs():
        if prefer_run and prefer_run not in str(run_dir):
            continue
        folds_csv = run_dir / "folds_grouped.csv"
        if not folds_csv.is_file():
            continue
        fold, row = pick_best_fold_from_csv(folds_csv)
        ck = run_dir / f"fold_{fold}" / "best.pt"
        if not ck.is_file():
            continue
        acc = float(row.get("accuracy", 0) or 0)
        candidates.append((acc, run_dir, fold, {**row, "checkpoint": str(ck.resolve())}))

    if not candidates:
        # Fallback: any grouped fold_0 best.pt
        for base in _results_roots():
            pattern = str(base / "cwru_reviewer_response" / "*" / "kfold" / "grouped" / "fold_0" / "best.pt")
            for ck in sorted(glob.glob(pattern), reverse=True):
                if os.path.isfile(ck):
                    return {
                        "checkpoint": os.path.normpath(ck),
                        "fold": 0,
                        "accuracy": None,
                        "f1": None,
                        "auc_roc": None,
                        "run_dir": os.path.dirname(os.path.dirname(ck)),
                        "folds_csv": None,
                        "note": "fallback fold_0 (no folds_grouped.csv metrics)",
                    }
        raise FileNotFoundError(
            "No CWRU grouped k-fold checkpoint found under results/cwru_reviewer_response/*/kfold/grouped/"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    acc, run_dir, fold, row = candidates[0]
    return {
        "checkpoint": row["checkpoint"],
        "fold": fold,
        "accuracy": float(row.get("accuracy")) if row.get("accuracy") is not None else None,
        "f1": float(row.get("f1")) if row.get("f1") is not None else None,
        "auc_roc": float(row.get("auc_roc")) if row.get("auc_roc") is not None else None,
        "run_dir": str(run_dir.resolve()),
        "folds_csv": str((run_dir / "folds_grouped.csv").resolve()),
        "dataset": "CWRU 12 kHz drive-end (grouped k-fold)",
        "model": "resnet50",
    }


def write_source_model_summary(out_path: str, info: Optional[Dict[str, Any]] = None) -> str:
    info = info or find_best_cwru_checkpoint()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def _fmt(x: Any) -> str:
        if x is None:
            return "N/A"
        try:
            v = float(x)
            return f"{v * 100:.2f}%" if v <= 1 else f"{v:.4f}"
        except (TypeError, ValueError):
            return str(x)

    lines = [
        "# Source model summary (CWRU → MFPT domain adaptation)",
        "",
        "Auto-located from completed CWRU grouped k-fold experiment artifacts.",
        "",
        f"- **Checkpoint path:** `{info.get('checkpoint', 'N/A')}`",
        f"- **Dataset:** {info.get('dataset', 'CWRU')}",
        f"- **Model:** {info.get('model', 'resnet50')}",
        f"- **Best fold:** {info.get('fold', 'N/A')}",
        f"- **Accuracy:** {_fmt(info.get('accuracy'))}",
        f"- **F1:** {_fmt(info.get('f1'))}",
        f"- **ROC-AUC:** {_fmt(info.get('auc_roc'))}",
        "",
    ]
    if info.get("folds_csv"):
        lines.append(f"- **Metrics source:** `{info['folds_csv']}`")
    if info.get("note"):
        lines.append(f"- **Note:** {info['note']}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(out_path.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return out_path
