"""
Harvest experiment artifacts from one or more roots and build a unified index + best-by-bucket.

Schemas recognized (best-effort):
  - Spectrogram fault run: directory containing both config.json and metrics.json
  - Publication bundle: results.json, training_history.json, ablation_results.json, cv_results.json
  - rigorous_cv_results.json
  - single_model_experiments_summary.json
  - fusion_summary.json
  - improved_experiments: exp*_results.json
  - Spectrogram ablation grid: ablation_results_*.json (list or rows) and ablation_results_*.csv

Usage (from repo root):
  python scripts/journal_aggregate_experiment_results.py
  python scripts/journal_aggregate_experiment_results.py --roots src/results,F:/Codes/.../results

Outputs (default under docs/journal_aggregation/):
  journal_experiment_index.json, journal_experiment_index.csv
  journal_best_by_bucket.json
  MISSING_DATA.md (when zero rows)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _primary_metric_from_dict(d: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    priority = [
        "val_balanced_accuracy",
        "balanced_accuracy",
        "auc_roc",
        "best_val_acc",
        "accuracy",
        "macro_f1",
        "f1",
        "mcc",
    ]
    for k in priority:
        if k in d and d[k] is not None:
            v = _safe_float(d[k])
            if v is not None:
                return k, v
    return "none", None


def _dataset_hint(cfg_or_metrics: Dict[str, Any]) -> str:
    for key in ("dataset_path", "train_root", "val_root", "dataset", "mimii_path"):
        v = cfg_or_metrics.get(key)
        if v and isinstance(v, str):
            return v[:120]
    return "unknown"


def _split_hint(cfg: Dict[str, Any], default: str = "unknown") -> str:
    if cfg.get("group_split") is True:
        return "stratified_group_kfold_first_fold"
    if cfg.get("group_split") is False:
        return "stratified_random_val"
    if "n_splits" in cfg or "cv" in str(cfg).lower():
        return "cross_validation"
    return default


def parse_spectrogram_run(config_path: Path, metrics_path: Path) -> Dict[str, Any]:
    cfg = _load_json(config_path)
    met = _load_json(metrics_path)
    name, val = _primary_metric_from_dict(met)
    run_id = config_path.parent.name
    return {
        "line_of_work": "spectrogram_image_supervised",
        "script": "experiments/train_spectrogram_fault.py",
        "run_id": run_id,
        "dataset_hint": _dataset_hint(cfg),
        "split_protocol": _split_hint(cfg),
        "primary_metric_name": name,
        "primary_metric_value": val,
        "extra_metrics": {k: met[k] for k in sorted(met.keys()) if isinstance(met[k], (int, float))},
        "artifact_path": str(config_path.parent.resolve()),
        "notes": f"model={cfg.get('model')}; augment={cfg.get('augment')}; loss={cfg.get('loss')}",
    }


def parse_single_model_summary(path: Path) -> List[Dict[str, Any]]:
    data = _load_json(path)
    rows: List[Dict[str, Any]] = []
    if not isinstance(data, dict):
        return rows
    for model_name, payload in data.items():
        if not isinstance(payload, dict) or "error" in payload:
            continue
        em = payload.get("evaluation_metrics") or {}
        flat = {**em, "best_val_acc": payload.get("best_val_acc")}
        name, val = _primary_metric_from_dict(flat)
        rows.append(
            {
                "line_of_work": "vision_supervised_zoo",
                "script": "experiments/run_single_model_experiments.py",
                "run_id": str(model_name),
                "dataset_hint": (payload.get("config") or {}).get("dataset_path", "unknown"),
                "split_protocol": "see_run_all_models",
                "primary_metric_name": name,
                "primary_metric_value": val,
                "extra_metrics": {k: flat[k] for k in sorted(flat) if isinstance(flat[k], (int, float))},
                "artifact_path": str(path.resolve()),
                "notes": "single_model_experiments_summary.json row",
            }
        )
    return rows


def parse_results_json_list(path: Path) -> List[Dict[str, Any]]:
    data = _load_json(path)
    rows: List[Dict[str, Any]] = []
    if not isinstance(data, list):
        return rows
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("model_name") or item.get("model") or "unknown"
        skip = {"model_name", "model", "y_true", "y_pred_probs", "y_pred"}
        metrics = {k: item[k] for k in item if k not in skip}
        mname, mval = _primary_metric_from_dict(metrics)
        rows.append(
            {
                "line_of_work": "publication_bundle",
                "script": "results.json",
                "run_id": str(name),
                "dataset_hint": "unknown",
                "split_protocol": "unknown",
                "primary_metric_name": mname,
                "primary_metric_value": mval,
                "extra_metrics": {k: metrics[k] for k in metrics if isinstance(metrics[k], (int, float))},
                "artifact_path": str(path.resolve()),
                "notes": "results.json list entry",
            }
        )
    return rows


def parse_rigorous_cv(path: Path) -> List[Dict[str, Any]]:
    data = _load_json(path)
    rows: List[Dict[str, Any]] = []
    if not isinstance(data, dict):
        return rows
    by_model = data.get("results_by_model") or data.get("models") or {}
    if isinstance(by_model, dict):
        for model_name, payload in by_model.items():
            if not isinstance(payload, dict):
                continue
            flat = dict(payload)
            name, val = _primary_metric_from_dict(flat)
            rows.append(
                {
                    "line_of_work": "cross_validation",
                    "script": "experiments/run_rigorous_cv.py",
                    "run_id": str(model_name),
                    "dataset_hint": data.get("dataset_path", "unknown"),
                    "split_protocol": "rigorous_cv",
                    "primary_metric_name": name,
                    "primary_metric_value": val,
                    "extra_metrics": {k: flat[k] for k in sorted(flat) if isinstance(flat[k], (int, float))},
                    "artifact_path": str(path.resolve()),
                    "notes": "rigorous_cv_results.json",
                }
            )
    return rows


def parse_fusion_summary(path: Path) -> List[Dict[str, Any]]:
    data = _load_json(path)
    rows: List[Dict[str, Any]] = []
    if not isinstance(data, dict):
        return rows
    for fusion_name, payload in data.items():
        if not isinstance(payload, dict):
            continue
        name, val = _primary_metric_from_dict(payload)
        rows.append(
            {
                "line_of_work": "audio_vision_fusion",
                "script": "experiments/run_fusion_experiments.py",
                "run_id": str(fusion_name),
                "dataset_hint": payload.get("dataset_path", "unknown"),
                "split_protocol": "see_fusion_config",
                "primary_metric_name": name,
                "primary_metric_value": val,
                "extra_metrics": {k: payload[k] for k in payload if isinstance(payload[k], (int, float))},
                "artifact_path": str(path.resolve()),
                "notes": "fusion_summary.json",
            }
        )
    return rows


def parse_improved_exp_results(path: Path) -> List[Dict[str, Any]]:
    data = _load_json(path)
    rows: List[Dict[str, Any]] = []
    if not isinstance(data, dict):
        return rows
    exp_id = path.stem
    best = data.get("best_metrics") or data.get("metrics") or data
    if not isinstance(best, dict):
        return rows
    name, val = _primary_metric_from_dict(best)
    rows.append(
        {
            "line_of_work": "improved_mimii_dinov2",
            "script": "experiments/run_improved_experiments.py",
            "run_id": exp_id,
            "dataset_hint": data.get("dataset_path", data.get("train_root", "unknown")),
            "split_protocol": str(data.get("split", "unknown")),
            "primary_metric_name": name,
            "primary_metric_value": val,
            "extra_metrics": {k: best[k] for k in sorted(best) if isinstance(best[k], (int, float))},
            "artifact_path": str(path.resolve()),
            "notes": "improved experiment JSON",
        }
    )
    return rows


def parse_ablation_json_generic(path: Path) -> List[Dict[str, Any]]:
    data = _load_json(path)
    rows: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for config_name, metrics in data.items():
            if not isinstance(metrics, dict):
                continue
            name, val = _primary_metric_from_dict(metrics)
            rows.append(
                {
                    "line_of_work": "ablation_studies",
                    "script": "experiments/run_ablation_comparison.py",
                    "run_id": f"{path.stem}:{config_name}",
                    "dataset_hint": "unknown",
                    "split_protocol": "unknown",
                    "primary_metric_name": name,
                    "primary_metric_value": val,
                    "extra_metrics": {k: metrics[k] for k in metrics if isinstance(metrics[k], (int, float))},
                    "artifact_path": str(path.resolve()),
                    "notes": str(config_name),
                }
            )
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                name, val = _primary_metric_from_dict(item)
                rows.append(
                    {
                        "line_of_work": "ablation_studies",
                        "script": "ablation_results.json",
                        "run_id": f"{path.stem}[{i}]",
                        "dataset_hint": "unknown",
                        "split_protocol": "unknown",
                        "primary_metric_name": name,
                        "primary_metric_value": val,
                        "extra_metrics": {
                            k: item[k] for k in item if isinstance(item[k], (int, float))
                        },
                        "artifact_path": str(path.resolve()),
                        "notes": "list row",
                    }
                )
    return rows


def parse_spectrogram_ablation_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            flat = {k: _safe_float(v) if v not in (None, "") else None for k, v in row.items()}
            flat_clean = {k: v for k, v in flat.items() if v is not None}
            name, val = _primary_metric_from_dict(flat_clean)
            if val is None:
                name, val = "best_val_bal_acc", flat_clean.get("best_val_bal_acc") or flat_clean.get(
                    "val_balanced_accuracy"
                )
                if val is not None:
                    name = "best_val_bal_acc"
            rows.append(
                {
                    "line_of_work": "spectrogram_image_supervised",
                    "script": "experiments/run_spectrogram_ablation.py",
                    "run_id": f"{path.stem}:row{i}",
                    "dataset_hint": row.get("dataset_path", row.get("out_dir", "unknown")),
                    "split_protocol": "ablation_grid",
                    "primary_metric_name": str(name),
                    "primary_metric_value": _safe_float(val),
                    "extra_metrics": flat_clean,
                    "artifact_path": str(path.resolve()),
                    "notes": json.dumps({k: row[k] for k in ("model", "augment", "loss") if k in row}),
                }
            )
    return rows


def _bucket_key(row: Dict[str, Any]) -> str:
    ds = re.sub(r"\s+", " ", str(row.get("dataset_hint", "unknown")))[:80]
    return f"{row.get('line_of_work')}|{row.get('split_protocol')}|{ds}"


def compute_best_by_bucket(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("primary_metric_value") is None:
            continue
        by_bucket[_bucket_key(r)].append(r)

    best_map: Dict[str, Any] = {}
    tie_rule = "max primary_metric_value; tie-breaker lexicographic run_id"

    for bucket, items in by_bucket.items():
        items_sorted = sorted(
            items,
            key=lambda x: (-float(x["primary_metric_value"]), str(x.get("run_id", ""))),
        )
        winner = items_sorted[0]
        best_map[bucket] = {
            "tie_rule": tie_rule,
            "winner": winner,
            "n_candidates": len(items),
        }
    return {"tie_rule_global": tie_rule, "buckets": best_map}


def discover_all_rows(roots: List[Path]) -> List[Dict[str, Any]]:
    seen_spectrogram_dirs: set[str] = set()
    seen_row_keys: set[Tuple[str, str, str]] = set()
    rows: List[Dict[str, Any]] = []

    def add_many(new_rows: Iterable[Dict[str, Any]]) -> None:
        for r in new_rows:
            key = (
                str(r.get("artifact_path", "")),
                str(r.get("run_id", "")),
                str(r.get("line_of_work", "")),
            )
            if key in seen_row_keys and key[0]:
                continue
            seen_row_keys.add(key)
            rows.append(r)

    for root in roots:
        if not root.is_dir():
            continue
        root = root.resolve()
        # Spectrogram runs: metrics.json + config.json same directory
        for metrics_path in root.rglob("metrics.json"):
            parent = metrics_path.parent
            cfg_path = parent / "config.json"
            if cfg_path.is_file():
                key = str(parent.resolve())
                if key in seen_spectrogram_dirs:
                    continue
                seen_spectrogram_dirs.add(key)
                try:
                    rows.append(parse_spectrogram_run(cfg_path, metrics_path))
                except (json.JSONDecodeError, OSError, KeyError):
                    pass

        for name, parser in (
            ("single_model_experiments_summary.json", parse_single_model_summary),
            ("rigorous_cv_results.json", parse_rigorous_cv),
            ("fusion_summary.json", parse_fusion_summary),
        ):
            for p in root.rglob(name):
                try:
                    add_many(parser(p))
                except (json.JSONDecodeError, OSError):
                    pass

        for p in root.rglob("results.json"):
            try:
                add_many(parse_results_json_list(p))
            except (json.JSONDecodeError, OSError):
                pass

        for p in root.rglob("exp*_results.json"):
            try:
                add_many(parse_improved_exp_results(p))
            except (json.JSONDecodeError, OSError):
                pass

        for p in root.rglob("ablation_results*.json"):
            try:
                add_many(parse_ablation_json_generic(p))
            except (json.JSONDecodeError, OSError):
                pass

        for p in root.rglob("ablation_results*.csv"):
            try:
                add_many(parse_spectrogram_ablation_csv(p))
            except (OSError, UnicodeDecodeError):
                pass

    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "line_of_work",
        "script",
        "run_id",
        "dataset_hint",
        "split_protocol",
        "primary_metric_name",
        "primary_metric_value",
        "artifact_path",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {c: r.get(c, "") for c in cols}
            em = r.get("extra_metrics")
            if em:
                row["notes"] = f"{row.get('notes','')}|extra={json.dumps(em)[:500]}"
            w.writerow(row)


def write_missing_data(out_dir: Path, roots: List[str], n: int) -> None:
    p = out_dir / "MISSING_DATA.md"
    p.write_text(
        "\n".join(
            [
                "# Missing experiment artifacts",
                "",
                f"No indexed rows were harvested (n={n}).",
                "",
                "Scanned roots:",
                "",
                *[f"- `{r}`" for r in roots],
                "",
                "Typical reasons:",
                "",
                "1. `results/` is gitignored and empty in this clone — run experiments locally or pass `--roots` to your populated directory.",
                "2. Artifacts use filenames not yet recognized by the harvester — extend `discover_all_rows` in `scripts/journal_aggregate_experiment_results.py`.",
                "",
                "After runs exist, re-run:",
                "`python scripts/journal_aggregate_experiment_results.py --roots <path1>,<path2>`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment results for journal paper.")
    parser.add_argument(
        "--roots",
        type=str,
        default="",
        help="Comma-separated directories to scan (default: repo src/results and results)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: docs/journal_aggregation)",
    )
    parser.add_argument(
        "--with-plots",
        action="store_true",
        help="After aggregation, run journal_build_comparative_figures.py",
    )
    args = parser.parse_args()
    root = _repo_root()
    out_dir = args.out_dir or (root / "docs" / "journal_aggregation")
    out_dir.mkdir(parents=True, exist_ok=True)

    default_roots = [root / "src" / "results", root / "results"]
    if args.roots.strip():
        roots = [Path(p.strip()).expanduser() for p in args.roots.split(",") if p.strip()]
    else:
        roots = default_roots

    roots = [r for r in roots if r.is_dir()]
    rows = discover_all_rows(roots)
    best = compute_best_by_bucket(rows)

    (out_dir / "journal_experiment_index.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    write_csv(rows, out_dir / "journal_experiment_index.csv")
    (out_dir / "journal_best_by_bucket.json").write_text(
        json.dumps(best, indent=2, default=str), encoding="utf-8"
    )

    if not rows:
        write_missing_data(out_dir, [str(r) for r in roots], 0)
    else:
        md = out_dir / "MISSING_DATA.md"
        if md.is_file():
            md.unlink()

    print(f"Indexed {len(rows)} rows -> {out_dir}")
    print(f"Buckets with a winner: {len(best.get('buckets', {}))}")

    if args.with_plots:
        fig_script = root / "scripts" / "journal_build_comparative_figures.py"
        idx_path = out_dir / "journal_experiment_index.json"
        subprocess.run(
            [sys.executable, str(fig_script), "--index", str(idx_path)],
            cwd=str(root),
            check=False,
        )


if __name__ == "__main__":
    main()
