"""
Orchestrator for CWRU reviewer-response validation suite.

Runs K-fold CV, cross-load experiments, emits audit report and manuscript-ready text.

Run from ``src/``:
  python -m experiments.run_cwru_reviewer_response \\
    --dataset_path ../data/cwru_spectrograms_full_png --k 5 --epochs 25 --do_kfold --do_cross_load
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluate_gradcam_physics import run_gradcam_physics
from experiments.run_cwru_cross_load import run_cross_load
from experiments.run_cwru_kfold_cv import run_kfold_cv
from modules.cwru_experiment_utils import load_manifest, prepare_cwru_image_dataset, set_global_seeds
from modules.cwru_split_checks import assert_no_file_overlap, write_verification_report
from utils.manuscript_io import PLOT_MODEL_LABEL

logger = logging.getLogger(__name__)

AUDIT_TEMPLATE = """# CWRU protocol audit (Task 1)

## Current training pipeline

- Entry point: `experiments/train_spectrogram_fault.py` uses `get_train_val_indices` from
  `modules/spectrogram_fault_data.py` for a single train/validation split.
- Default split: `StratifiedGroupKFold` first fold when more than one group exists; otherwise
  **stratified random `train_test_split` with shuffle** at the **window** level.
- `ImageDataset` in `run_all_models.py` assigns `group="default"` for flat `normal/` / `abnormal/`
  folders, so **all windows share one group** unless `--infer_mat_stem_groups` is used.
- **Leakage risk:** overlapping windows from the same `.mat` recording can appear in both train and
  validation when splitting randomly by window. This likely inflates validation accuracy.
- Mitigation: `--infer_mat_stem_groups` plus `get_train_val_indices_mat_stem_stratified` holds out
  whole recordings (stems) within each class.

## Live dataset check

- Dataset path: `{dataset_path}`
- Samples: {n_samples}
- Unique mat stems (groups): {n_groups}
- Manifest rows: {n_manifest}
- Loads in manifest: {loads}

## Default-split simulation (window-level)

{default_split_check}

## Reviewer-response experiments in this run

- K-fold CV: {kfold_status}
- Cross-load generalization: {cross_load_status}
- Grouped (recording-wise) CV is the **leakage-free** primary evidence; window-level stratified
  CV is reported for comparison and labeled accordingly.
"""


def _audit_default_split(dataset_path: str, seed: int = 42) -> Dict[str, Any]:
    """Simulate default random window split and check stem overlap."""
    from sklearn.model_selection import train_test_split

    train_ds, _, _ = prepare_cwru_image_dataset(dataset_path, augment="off")
    n = len(train_ds)
    labels = train_ds.labels
    idx = list(range(n))
    tr, va = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    train_paths = [train_ds.samples[i] for i in tr]
    val_paths = [train_ds.samples[i] for i in va]
    check = assert_no_file_overlap(train_paths, val_paths)
    return check


def _format_pct(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "N/A"
    if std is not None:
        return f"{mean * 100:.1f}% ± {std * 100:.1f}%"
    return f"{mean * 100:.1f}%"


def _write_manuscript_text(
    path: str,
    kfold_results: Optional[Dict[str, Any]],
    cross_load_results: Optional[Dict[str, Any]],
    k: int,
) -> None:
    lines = ["# Manuscript text snippets (auto-generated)", ""]

    if kfold_results and "modes" in kfold_results:
        grouped = kfold_results["modes"].get("grouped", {})
        strat = kfold_results["modes"].get("stratified", {})
        if grouped.get("aggregated"):
            agg = grouped["aggregated"]
            acc = agg.get("accuracy", {})
            f1 = agg.get("f1", {})
            lines.append(
                f"Five-fold cross-validation on the full CWRU 12 kHz drive-end binary task "
                f"(normal vs abnormal) yielded a mean accuracy of "
                f"{_format_pct(acc.get('mean'), acc.get('std'))} with grouped, recording-wise splits "
                f"(leakage-free), and {_format_pct(strat.get('aggregated', {}).get('accuracy', {}).get('mean'), strat.get('aggregated', {}).get('accuracy', {}).get('std'))} "
                f"under window-level stratified splits (k={k}). "
                f"Macro F1 was {_format_pct(f1.get('mean'), f1.get('std'))} (grouped)."
            )
            lines.append("")
        elif strat.get("aggregated"):
            agg = strat["aggregated"]
            acc = agg.get("accuracy", {})
            lines.append(
                f"{k}-fold stratified cross-validation yielded mean accuracy "
                f"{_format_pct(acc.get('mean'), acc.get('std'))}."
            )
            lines.append("")

    if cross_load_results and cross_load_results.get("rows"):
        accs = [r[2] for r in cross_load_results["rows"]]
        mean_acc = sum(accs) / len(accs) if accs else 0
        lines.append(
            f"Cross-load transfer among motor loads 1–3 HP (six directed pairs) averaged "
            f"{mean_acc * 100:.1f}% test accuracy, demonstrating generalization beyond a single "
            f"operating condition."
        )
        lines.append("")
        lines.append("| Train load | Test load | Accuracy |")
        lines.append("|---|---|---|")
        for row in cross_load_results["rows"]:
            lines.append(f"| {row[0]} HP | {row[1]} HP | {row[2] * 100:.1f}% |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _copy_if_exists(src: str, dst: str) -> bool:
    if os.path.isfile(src):
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def publish_manuscript_figures(
    run_dir: str,
    gradcam_dir: Optional[str],
    cross_load_dir: Optional[str],
    *,
    loc_summary: Optional[Dict[str, Any]] = None,
    cross_load_rows: Optional[List[List[Any]]] = None,
) -> str:
    """Copy flagship artifacts to results/manuscript_figures/ and write manuscript_summary.md."""
    ms_dir = os.path.join("results", "manuscript_figures")
    os.makedirs(ms_dir, exist_ok=True)

    if cross_load_dir:
        _copy_if_exists(
            os.path.join(cross_load_dir, "figures", "cross_load_heatmap.png"),
            os.path.join(ms_dir, "cross_load_heatmap.png"),
        )
        _copy_if_exists(
            os.path.join(cross_load_dir, "cross_load.csv"),
            os.path.join(ms_dir, "cross_load.csv"),
        )

    if gradcam_dir:
        for name in (
            "localization_fidelity_barplot.png",
            "frequency_markers.png",
            "spectrogram_fault_overlay.png",
            "gradcam_fault_overlay.png",
            "gradcam_physics_barplot.png",
            "fault_frequencies.csv",
            "localization_metrics.csv",
            "gradcam_physics_metrics.csv",
        ):
            _copy_if_exists(os.path.join(gradcam_dir, name), os.path.join(ms_dir, name))

    lines = [
        "# Manuscript summary (reviewer response)",
        "",
        f"Model (display): **{PLOT_MODEL_LABEL}**",
        "",
        "## 1. Cross-load generalization",
        "",
    ]
    if cross_load_rows:
        accs = [float(r[2]) for r in cross_load_rows]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        lines.append(
            f"Six directed cross-load pairs (train loads 1–3 HP → test loads 1–3 HP) "
            f"achieved **{mean_acc * 100:.1f}%** mean test accuracy, showing transfer beyond a single operating condition."
        )
        lines.append("")
        lines.append("| Train HP | Test HP | Accuracy |")
        lines.append("|---|---|---|")
        for row in cross_load_rows:
            lines.append(f"| {row[0]} | {row[1]} | {row[2] * 100:.1f}% |")
    else:
        lines.append("_Cross-load results not available for this run._")

    lines.extend(["", "## 2. Localization fidelity", ""])
    if loc_summary:
        lines.append(
            f"Grad-CAM localization fidelity (top-20% activation vs fault-frequency columns) "
            f"averaged **{loc_summary.get('mean', 0) * 100:.1f}% ± {loc_summary.get('std', 0) * 100:.1f}%** "
            f"over {loc_summary.get('n_samples', 0)} windows."
        )
        pc = loc_summary.get("per_class") or {}
        if pc:
            lines.append("")
            for cls, block in pc.items():
                lines.append(
                    f"- **{cls}**: {block.get('mean', 0) * 100:.1f}% ± {block.get('std', 0) * 100:.1f}% (n={block.get('n', 0)})"
                )
    else:
        lines.append("_Localization metrics not available._")

    lines.extend(
        [
            "",
            "## 3. Fault-frequency alignment",
            "",
            "Fault characteristic frequencies (BPFI, BPFO, BSF, FTF) were computed from the CWRU "
            "6205-2RS JEM drive-end geometry and overlaid on raw spectra and mel spectrograms. "
            "Grad-CAM energy within ±5% mel columns around BPFI/BPFO/BSF is reported in "
            "`gradcam_physics_metrics.csv`.",
            "",
            "## 4. Physics interpretation of Grad-CAM",
            "",
            "- **Inner race (IR):** attention should concentrate near **BPFI** harmonics.",
            "- **Outer race (OR):** attention should concentrate near **BPFO** harmonics.",
            "- **Ball (B):** attention should concentrate near **BSF/BSF2** harmonics.",
            "",
            "See `frequency_markers.png`, `spectrogram_fault_overlay.png`, and "
            "`gradcam_fault_overlay.png` for qualitative alignment. Low fault frequencies map to a "
            "narrow band on the mel axis; raw-spectrum markers provide the clearest physical reference.",
            "",
        ]
    )

    summary_path = os.path.join(ms_dir, "manuscript_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return ms_dir


def _copy_tree(src: str, dst: str) -> None:
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


def run_reviewer_response(
    dataset_path: str,
    output_dir: str,
    *,
    do_kfold: bool = True,
    do_cross_load: bool = True,
    do_gradcam_physics: bool = True,
    k: int = 5,
    epochs: int = 25,
    seed: int = 42,
    init_checkpoint: Optional[str] = None,
    cv_mode: str = "both",
    mat_dir: str = "../data/cwru_mat",
    gradcam_checkpoint: Optional[str] = None,
    per_class_samples: int = 8,
) -> Dict[str, Any]:
    set_global_seeds(seed)
    os.makedirs(output_dir, exist_ok=True)

    run_config = {
        "dataset_path": dataset_path,
        "k": k,
        "epochs": epochs,
        "seed": seed,
        "init_checkpoint": init_checkpoint,
        "do_kfold": do_kfold,
        "do_cross_load": do_cross_load,
        "do_gradcam_physics": do_gradcam_physics,
        "mat_dir": mat_dir,
        "cv_mode": cv_mode,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    try:
        manifest = load_manifest(dataset_path)
        n_manifest = len(manifest)
        loads = sorted(manifest["load_hp"].dropna().unique().tolist()) if "load_hp" in manifest.columns else []
    except FileNotFoundError:
        n_manifest = 0
        loads = []

    train_ds, _, _ = prepare_cwru_image_dataset(dataset_path, augment="off")
    n_groups = len(set(train_ds.groups))
    default_check = _audit_default_split(dataset_path, seed)

    kfold_results: Optional[Dict[str, Any]] = None
    cross_load_results: Optional[Dict[str, Any]] = None
    gradcam_results: Optional[Dict[str, Any]] = None
    cl_dir: Optional[str] = None
    gradcam_dir: Optional[str] = None

    if do_kfold:
        kfold_dir = os.path.join(output_dir, "kfold")
        kfold_results = run_kfold_cv(
            dataset_path,
            kfold_dir,
            cv_mode=cv_mode,
            k=k,
            epochs=epochs,
            seed=seed,
            init_checkpoint=init_checkpoint,
        )
        tables_dst = os.path.join(output_dir, "tables", "kfold")
        figures_dst = os.path.join(output_dir, "figures", "kfold")
        for mode in (kfold_results.get("modes") or {}):
            _copy_tree(os.path.join(kfold_dir, mode), os.path.join(tables_dst, mode))
            fig_src = os.path.join(kfold_dir, mode, "figures")
            _copy_tree(fig_src, os.path.join(figures_dst, mode))

    if do_cross_load:
        cl_dir = os.path.join(output_dir, "cross_load")
        cross_load_results = run_cross_load(
            dataset_path,
            cl_dir,
            epochs=epochs,
            seed=seed,
            init_checkpoint=init_checkpoint,
        )
        _copy_tree(cl_dir, os.path.join(output_dir, "tables", "cross_load"))
        _copy_tree(os.path.join(cl_dir, "figures"), os.path.join(output_dir, "figures", "cross_load"))

    if do_gradcam_physics:
        gradcam_dir = os.path.join(output_dir, "gradcam_physics")
        ck = gradcam_checkpoint
        if ck is None and do_kfold:
            ck = os.path.join(output_dir, "kfold", "grouped", "fold_0", "best.pt")
            if not os.path.isfile(ck):
                ck = find_grouped_checkpoint(output_dir)
        if ck is None:
            from experiments.evaluate_gradcam_physics import find_default_checkpoint

            ck = find_default_checkpoint()
        if ck and os.path.isfile(ck):
            gradcam_results = run_gradcam_physics(
                dataset_path,
                gradcam_dir,
                checkpoint=ck,
                mat_dir=mat_dir,
                per_class_samples=per_class_samples,
                seed=seed,
            )
            _copy_tree(gradcam_dir, os.path.join(output_dir, "figures", "gradcam_physics"))
        else:
            logger.warning("Grad-CAM physics skipped: no checkpoint available")

    publish_manuscript_figures(
        output_dir,
        gradcam_dir,
        cl_dir,
        loc_summary=(gradcam_results or {}).get("localization_summary"),
        cross_load_rows=(cross_load_results or {}).get("rows"),
    )

    audit = AUDIT_TEMPLATE.format(
        dataset_path=dataset_path,
        n_samples=len(train_ds),
        n_groups=n_groups,
        n_manifest=n_manifest,
        loads=loads,
        default_split_check=(
            f"Random 80/20 window split: **{'PASS' if default_check['passed'] else 'FAIL'}** "
            f"(overlap stems: {default_check.get('overlap_stems', [])[:10]})"
        ),
        kfold_status="executed" if do_kfold else "skipped",
        cross_load_status="executed" if do_cross_load else "skipped",
    )
    with open(os.path.join(output_dir, "audit_report.md"), "w", encoding="utf-8") as f:
        f.write(audit)

    write_verification_report(
        os.path.join(output_dir, "default_split_verification"),
        [default_check],
        title="Default window-level split verification",
    )

    _write_manuscript_text(
        os.path.join(output_dir, "manuscript_text.md"),
        kfold_results,
        cross_load_results,
        k,
    )

    summary = {
        "output_dir": output_dir,
        "kfold": kfold_results,
        "cross_load": cross_load_results,
        "gradcam_physics": gradcam_results,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "output_dir": output_dir,
                "kfold_modes": list((kfold_results or {}).get("modes", {}).keys()),
                "cross_load_pairs": len((cross_load_results or {}).get("rows", [])),
                "gradcam_physics_samples": (gradcam_results or {}).get("localization_summary", {}).get("n_samples", 0),
            },
            f,
            indent=2,
        )
    return summary


def find_grouped_checkpoint(output_dir: str) -> Optional[str]:
    ck = os.path.join(output_dir, "kfold", "grouped", "fold_0", "best.pt")
    return ck if os.path.isfile(ck) else None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CWRU reviewer-response orchestrator")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--do_kfold", action="store_true")
    p.add_argument("--do_cross_load", action="store_true")
    p.add_argument("--do_gradcam_physics", action="store_true")
    p.add_argument("--mat_dir", type=str, default="../data/cwru_mat")
    p.add_argument("--gradcam_checkpoint", type=str, default=None)
    p.add_argument("--per_class_samples", type=int, default=8)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_checkpoint", type=str, default=None)
    p.add_argument("--cv_mode", type=str, default="both", choices=["stratified", "grouped", "both"])
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    if not args.do_kfold and not args.do_cross_load and not args.do_gradcam_physics:
        args.do_kfold = True
        args.do_cross_load = True
        args.do_gradcam_physics = True
    out = args.output_dir or os.path.join(
        "results/cwru_reviewer_response", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_reviewer_response(
        args.dataset_path,
        out,
        do_kfold=args.do_kfold,
        do_cross_load=args.do_cross_load,
        do_gradcam_physics=args.do_gradcam_physics,
        k=args.k,
        epochs=args.epochs,
        seed=args.seed,
        init_checkpoint=args.init_checkpoint,
        cv_mode=args.cv_mode,
        mat_dir=args.mat_dir,
        gradcam_checkpoint=args.gradcam_checkpoint,
        per_class_samples=args.per_class_samples,
    )
    logger.info("Reviewer-response suite complete: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
