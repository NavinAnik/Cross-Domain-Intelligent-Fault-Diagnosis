"""
Build MFPT domain-adaptation reviewer package from completed experiment outputs.

Mirrors build_reviewer_package.py for CWRU→MFPT artifacts.

Run from ``src/``:
  python build_mfpt_reviewer_package.py
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / "results"
PKG = RESULTS / "domain_adaptation" / "reviewer_package"

SUBDIRS = ["figures", "metrics", "tables", "manuscript", "reviewer_response", "raw_results", "docs"]

MISSING: List[str] = []


def _note_missing(label: str, hint: str) -> None:
    MISSING.append(f"{label} -- expected: {hint}")


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _latest_domain_adapt_run() -> Optional[Path]:
    base = RESULTS / "domain_adaptation"
    if not base.is_dir():
        return None
    cands = [
        p for p in base.iterdir()
        if p.is_dir() and p.name not in ("reviewer_package",) and (p / "domain_adaptation.csv").exists()
    ]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _mfpt_kfold_run() -> Optional[Path]:
    base = RESULTS / "mfpt" / "kfold_cv"
    if not base.is_dir():
        return None
    cands = [p for p in base.iterdir() if p.is_dir() and (p / "grouped" / "folds_grouped.csv").exists()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def load_domain_adaptation(run_dir: Path) -> Dict[str, Any]:
    csv_path = run_dir / "domain_adaptation.csv"
    if not csv_path.exists():
        _note_missing("domain_adaptation.csv", str(csv_path))
        return {}
    df = pd.read_csv(csv_path)
    rows = df.to_dict("records")
    return {"rows": rows, "source": os.path.relpath(csv_path, REPO).replace("\\", "/")}


def load_source_summary() -> Dict[str, Any]:
    jp = REPO / "results" / "domain_adaptation" / "source_model_summary.json"
    if not jp.exists():
        jp = RESULTS / "domain_adaptation" / "source_model_summary.json"
    if jp.exists():
        return json.loads(jp.read_text(encoding="utf-8"))
    _note_missing("source_model_summary", str(jp))
    return {}


def write_master_metrics(ctx: Dict[str, Any]) -> None:
    long_rows: List[Dict[str, Any]] = []

    def add(category, group, metric, value, source=""):
        long_rows.append({"category": category, "group": group, "metric": metric, "value": value, "source": source})

    src = ctx.get("source", {})
    if src:
        s = "source_model_summary.json"
        for k in ("accuracy", "f1", "auc_roc"):
            if src.get(k) is not None:
                add("source_cwru", "grouped_kfold", k, round(float(src[k]), 4), s)

    da = ctx.get("domain_adaptation", {})
    if da.get("rows"):
        s = da.get("source", "")
        for r in da["rows"]:
            g = f"{r.get('Protocol', '')}_{r.get('Setting', '')}"
            for mk in ("Accuracy", "Precision", "Recall", "F1", "AUC-ROC"):
                if mk in r and r[mk] == r[mk]:
                    add("domain_adaptation", g, mk.lower().replace("-", "_"), r[mk], s)

    metrics_dir = PKG / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "master_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category", "group", "metric", "value", "source"])
        w.writeheader()
        w.writerows(long_rows)
    with open(metrics_dir / "master_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2)


def copy_artifacts(da_run: Optional[Path], kf_run: Optional[Path]) -> None:
    fig = PKG / "figures"
    tab = PKG / "tables"
    raw = PKG / "raw_results"
    for d in (fig, tab, raw):
        d.mkdir(parents=True, exist_ok=True)

    if da_run:
        for name in ("domain_adaptation.csv", "domain_adaptation.tex", "audit_report.md", "manuscript_text.md", "summary.json"):
            src = da_run / name
            if src.exists():
                shutil.copy2(src, tab / name if name.endswith(".csv") or name.endswith(".tex") else raw / name)
        fig_src = da_run / "figures"
        if fig_src.is_dir():
            for f in fig_src.glob("*"):
                if f.suffix.lower() in (".png", ".pdf"):
                    shutil.copy2(f, fig / f.name)

    src_summary = _first_existing([
        REPO / "results" / "domain_adaptation" / "source_model_summary.md",
        RESULTS / "domain_adaptation" / "source_model_summary.md",
    ])
    if src_summary:
        shutil.copy2(src_summary, raw / "source_model_summary.md")

    if kf_run:
        for mode in ("grouped", "stratified"):
            fp = kf_run / mode / f"folds_{mode}.csv"
            if fp.exists():
                shutil.copy2(fp, tab / f"mfpt_folds_{mode}.csv")


def write_reviewer_response(ctx: Dict[str, Any]) -> None:
    da = ctx.get("domain_adaptation", {})
    src = ctx.get("source", {})
    lines = [
        "# MFPT Domain Adaptation — Reviewer Response",
        "",
        "We evaluated cross-domain transfer from CWRU (source) to MFPT (target) using the same "
        "log-Mel spectrogram ResNet50 pipeline.",
        "",
        f"**Source checkpoint:** `{src.get('checkpoint', 'N/A')}`",
        f"**Source CWRU grouped k-fold accuracy:** { _pct(src.get('accuracy')) }",
        "",
        "## Domain adaptation results",
        "",
    ]
    if da.get("rows"):
        lines += [
            "| Protocol | Setting | Accuracy | F1 |",
            "|---|---|---|---|",
        ]
        for r in da["rows"]:
            lines.append(
                f"| {r.get('Protocol')} | {r.get('Setting')} | "
                f"{_pct(r.get('Accuracy'))} | {_pct(r.get('F1'))} |"
            )
    else:
        lines.append("_Domain adaptation results not found._")
    lines.append("")

    out = PKG / "reviewer_response" / "reviewer_response.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def write_manuscript(ctx: Dict[str, Any]) -> None:
    da = ctx.get("domain_adaptation", {})
    lines = [
        "# Results — MFPT domain adaptation",
        "",
        "The MFPT bearing fault dataset was integrated using the identical preprocessing and "
        "evaluation protocol as CWRU (log-Mel spectrograms, ResNet50, grouped stem splits).",
        "",
    ]
    if da.get("rows"):
        zs = [r for r in da["rows"] if r.get("Protocol") == "zero_shot"]
        if zs:
            lines.append(
                f"Zero-shot evaluation (CWRU-trained model on MFPT without fine-tuning) achieved "
                f"{_pct(zs[0].get('Accuracy'))} accuracy."
            )
        fs = [r for r in da["rows"] if r.get("Protocol") == "few_shot"]
        if fs:
            best = max(fs, key=lambda r: float(r.get("Accuracy", 0) or 0))
            lines.append(
                f"Few-shot fine-tuning peaked at {best.get('Setting')} labels "
                f"({_pct(best.get('Accuracy'))} accuracy)."
            )
    (PKG / "manuscript" / "results_section.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "N/A"
    try:
        v = float(x)
        return f"{v * 100:.1f}%" if v <= 1 else f"{v:.2f}"
    except (TypeError, ValueError):
        return str(x)


def write_validation() -> None:
    lines = [
        "# MFPT reviewer package validation",
        "",
        f"Generated {_dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Missing artifacts",
        "",
    ]
    if MISSING:
        lines += [f"- {m}" for m in MISSING]
    else:
        lines.append("None.")
    (PKG / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for d in SUBDIRS:
        (PKG / d).mkdir(parents=True, exist_ok=True)

    da_run = _latest_domain_adapt_run()
    kf_run = _mfpt_kfold_run()

    ctx = {
        "source": load_source_summary(),
        "domain_adaptation": load_domain_adaptation(da_run) if da_run else {},
        "mfpt_kfold_dir": str(kf_run) if kf_run else None,
        "domain_adapt_run": str(da_run) if da_run else None,
    }

    copy_artifacts(da_run, kf_run)
    write_master_metrics(ctx)
    write_reviewer_response(ctx)
    write_manuscript(ctx)
    write_validation()

    print(f"Package: {PKG}")
    print(f"Missing: {len(MISSING)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
