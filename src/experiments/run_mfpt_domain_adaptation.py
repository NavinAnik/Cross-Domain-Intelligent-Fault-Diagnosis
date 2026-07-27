"""
CWRU → MFPT domain adaptation experiments (zero-shot, few-shot, full fine-tune vs scratch).

Mirrors CWRU reviewer-response evaluation style and reporting.

Run from ``src/``:
  python -m experiments.run_mfpt_domain_adaptation \\
    --dataset_path ../data/mfpt_spectrograms --epochs 25 --do_zero_shot --do_few_shot --do_full_finetune
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train_spectrogram_fault import evaluate_model
from modules.cwru_split_checks import assert_no_file_overlap, write_verification_report
from modules.evaluation import compute_classification_metrics
from modules.mfpt_experiment_utils import (
    build_and_train_model,
    build_spectrogram_train_args,
    indices_for_label_fraction,
    indices_train_val_from_pool,
    load_manifest,
    persist_training_history,
    prepare_mfpt_image_dataset,
    save_test_evaluation_plots,
    set_global_seeds,
)
from modules.source_model_locator import find_best_cwru_checkpoint, write_source_model_summary
from modules.spectrogram_cnn_models import build_model
from utils.manuscript_io import PLOT_MODEL_LABEL, save_figure, save_table

logger = logging.getLogger(__name__)

FEW_SHOT_FRACTIONS = [0.01, 0.05, 0.10, 0.20]


def _metrics_to_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (float, np.floating)):
            out[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _evaluate_checkpoint(
    checkpoint: str,
    train_ds,
    val_ds,
    test_idx: List[int],
    *,
    model_name: str,
    batch_size: int,
    amp: bool,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = max(train_ds.labels) + 1
    model = build_model(model_name, num_classes=num_classes, pretrained=False).to(device)
    ck = os.path.normpath(os.path.expandvars(os.path.expanduser(checkpoint)))
    try:
        warm = torch.load(ck, map_location=device, weights_only=False)
    except TypeError:
        warm = torch.load(ck, map_location=device)
    sd = warm.get("model_state_dict", warm)
    model.load_state_dict(sd, strict=True)

    nw = 0 if os.name == "nt" else 2
    test_loader = DataLoader(
        Subset(val_ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )
    y_t, y_p, y_pr = evaluate_model(model, test_loader, device, num_classes, amp)
    metrics = compute_classification_metrics(y_t, y_p, y_pr, n_classes=num_classes)
    if out_dir:
        plot_paths = save_test_evaluation_plots(out_dir, y_t, y_p, y_pr, num_classes, prefix="test_")
        metrics["plot_paths"] = plot_paths
    return metrics


def _evaluate_model_on_indices(
    model: torch.nn.Module,
    val_ds,
    test_idx: List[int],
    *,
    batch_size: int,
    device: torch.device,
    num_classes: int,
    amp: bool,
    out_dir: str,
) -> Dict[str, Any]:
    nw = 0 if os.name == "nt" else 2
    test_loader = DataLoader(
        Subset(val_ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )
    y_t, y_p, y_pr = evaluate_model(model, test_loader, device, num_classes, amp)
    metrics = compute_classification_metrics(y_t, y_p, y_pr, n_classes=num_classes)
    plot_paths = save_test_evaluation_plots(out_dir, y_t, y_p, y_pr, num_classes, prefix="test_")
    metrics["plot_paths"] = plot_paths
    return metrics


def _plot_few_shot_curve(rows: List[List[Any]], path: str) -> None:
    fracs = [float(r[0]) * 100 for r in rows]
    accs = [float(r[2]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fracs, accs, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("MFPT training label fraction (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"CWRU→MFPT few-shot fine-tuning ({PLOT_MODEL_LABEL})")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    save_figure(fig, path)


def _plot_protocol_bars(rows: List[List[Any]], path: str) -> None:
    names = [str(r[0]) for r in rows]
    accs = [float(r[2]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, accs, color="steelblue", alpha=0.85)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"CWRU→MFPT domain adaptation ({PLOT_MODEL_LABEL})")
    ax.set_ylim(0, 105)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    save_figure(fig, path)


def run_zero_shot(
    dataset_path: str,
    output_dir: str,
    *,
    source_checkpoint: str,
    model: str = "resnet50",
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    set_global_seeds(seed)
    train_ds, val_ds, _ = prepare_mfpt_image_dataset(dataset_path, augment="off")
    test_idx = list(range(len(train_ds)))
    run_dir = os.path.join(output_dir, "zero_shot")
    metrics = _evaluate_checkpoint(
        source_checkpoint,
        train_ds,
        val_ds,
        test_idx,
        model_name=model,
        batch_size=batch_size,
        amp=True,
        out_dir=run_dir,
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": "zero_shot",
                "source_checkpoint": source_checkpoint,
                "n_test": len(test_idx),
                "seed": seed,
            },
            f,
            indent=2,
        )
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_metrics_to_json(metrics), f, indent=2)
    logger.info("Zero-shot | acc=%.4f f1=%.4f auc=%.4f", metrics["accuracy"], metrics["f1"], metrics.get("auc_roc", 0))
    return metrics


def run_few_shot(
    dataset_path: str,
    output_dir: str,
    *,
    source_checkpoint: str,
    fractions: List[float],
    epochs: int,
    seed: int,
    model: str = "resnet50",
    batch_size: int = 32,
    val_split: float = 0.15,
) -> List[Dict[str, Any]]:
    set_global_seeds(seed)
    _ = load_manifest(dataset_path)
    train_ds, val_ds, _ = prepare_mfpt_image_dataset(dataset_path)
    labels = train_ds.labels
    groups = train_ds.groups
    pool = list(range(len(train_ds)))

    train_args = build_spectrogram_train_args(
        dataset_path=dataset_path,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        infer_mat_stem_groups=True,
    )

    results: List[Dict[str, Any]] = []
    verification: List[Dict[str, Any]] = []

    for fi, frac in enumerate(fractions):
        frac_seed = seed + fi * 300
        train_pool, test_pool = indices_for_label_fraction(labels, groups, frac, seed=frac_seed, pool_indices=pool)
        if not train_pool or not test_pool:
            logger.warning("Skip few-shot %.0f%%: empty train or test pool", frac * 100)
            continue

        train_paths = [train_ds.samples[i] for i in train_pool]
        test_paths = [train_ds.samples[i] for i in test_pool]
        verification.append(assert_no_file_overlap(train_paths, test_paths))

        train_idx, val_idx = indices_train_val_from_pool(train_pool, labels, groups, val_split, frac_seed + 1)

        run_dir = os.path.join(output_dir, f"few_shot_{int(frac * 100)}pct")
        result = build_and_train_model(
            train_ds,
            val_ds,
            train_idx,
            val_idx,
            out_dir=run_dir,
            seed=frac_seed,
            init_checkpoint=source_checkpoint,
            train_args=train_args,
            save_plots=True,
        )
        persist_training_history(run_dir, result["history"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = max(train_ds.labels) + 1
        eval_model = build_model(model, num_classes=num_classes, pretrained=False).to(device)
        if result["best_state"]:
            eval_model.load_state_dict(result["best_state"])
        test_metrics = _evaluate_model_on_indices(
            eval_model,
            val_ds,
            test_pool,
            batch_size=batch_size,
            device=device,
            num_classes=num_classes,
            amp=train_args.amp,
            out_dir=run_dir,
        )

        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "protocol": "few_shot",
                    "fraction": frac,
                    "source_checkpoint": source_checkpoint,
                    "n_train_pool": len(train_pool),
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_pool),
                    "seed": frac_seed,
                },
                f,
                indent=2,
            )
        with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(_metrics_to_json(test_metrics), f, indent=2)

        results.append({"fraction": frac, "metrics": test_metrics, "run_dir": run_dir})
        logger.info(
            "Few-shot %.0f%% | acc=%.4f f1=%.4f",
            frac * 100,
            test_metrics["accuracy"],
            test_metrics["f1"],
        )

    write_verification_report(
        os.path.join(output_dir, "few_shot_split_verification"),
        verification,
        title="MFPT few-shot split verification",
    )
    return results


def run_full_finetune_comparison(
    dataset_path: str,
    output_dir: str,
    *,
    source_checkpoint: Optional[str],
    epochs: int,
    seed: int,
    model: str = "resnet50",
    batch_size: int = 32,
    val_split: float = 0.2,
) -> List[Dict[str, Any]]:
    set_global_seeds(seed)
    train_ds, val_ds, _ = prepare_mfpt_image_dataset(dataset_path)
    labels = train_ds.labels
    groups = train_ds.groups
    pool = list(range(len(train_ds)))

    # Hold out ~val_split fraction of stems as test; remainder is train pool.
    train_pool, test_idx = indices_for_label_fraction(
        labels, groups, 1.0 - val_split, seed=seed, pool_indices=pool
    )
    if not test_idx:
        test_idx = train_pool[-max(1, len(train_pool) // 5) :]
        train_pool = train_pool[: -len(test_idx)]

    verification = [
        assert_no_file_overlap(
            [train_ds.samples[i] for i in train_pool],
            [train_ds.samples[i] for i in test_idx],
        )
    ]
    write_verification_report(
        os.path.join(output_dir, "full_finetune_split_verification"),
        verification,
        title="MFPT full fine-tune split verification",
    )

    protocols = [
        ("pretrained_cwru", source_checkpoint, True),
        ("from_scratch", None, False),
    ]
    out_results: List[Dict[str, Any]] = []

    for pi, (name, init_ck, use_init) in enumerate(protocols):
        run_seed = seed + pi * 400
        run_dir = os.path.join(output_dir, name)
        args = build_spectrogram_train_args(
            dataset_path=dataset_path,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            pretrained=False,
            infer_mat_stem_groups=True,
        )
        inner_tr, inner_va = indices_train_val_from_pool(train_pool, labels, groups, 0.15, run_seed)

        result = build_and_train_model(
            train_ds,
            val_ds,
            inner_tr,
            inner_va,
            out_dir=run_dir,
            seed=run_seed,
            init_checkpoint=init_ck if use_init else None,
            train_args=args,
            save_plots=True,
        )
        persist_training_history(run_dir, result["history"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = max(train_ds.labels) + 1
        eval_model = build_model(model, num_classes=num_classes, pretrained=False).to(device)
        if result["best_state"]:
            eval_model.load_state_dict(result["best_state"])
        test_metrics = _evaluate_model_on_indices(
            eval_model,
            val_ds,
            test_idx,
            batch_size=batch_size,
            device=device,
            num_classes=num_classes,
            amp=args.amp,
            out_dir=run_dir,
        )

        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "protocol": name,
                    "init_checkpoint": init_ck if use_init else None,
                    "n_train": len(inner_tr),
                    "n_val": len(inner_va),
                    "n_test": len(test_idx),
                    "seed": run_seed,
                },
                f,
                indent=2,
            )
        with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(_metrics_to_json(test_metrics), f, indent=2)

        out_results.append({"protocol": name, "metrics": test_metrics})
        logger.info(
            "Full fine-tune %s | acc=%.4f f1=%.4f",
            name,
            test_metrics["accuracy"],
            test_metrics["f1"],
        )

    return out_results


def _write_manuscript_text(path: str, summary_rows: List[List[Any]], source_info: Dict[str, Any]) -> None:
    lines = [
        "# Manuscript text snippets — CWRU→MFPT domain adaptation",
        "",
        f"Source model: ResNet50 trained on CWRU 12 kHz drive-end spectrograms "
        f"(checkpoint `{source_info.get('checkpoint', 'N/A')}`, "
        f"grouped k-fold accuracy {_pct(source_info.get('accuracy'))}).",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- **{row[0]}**: accuracy {float(row[2])*100:.1f}%, F1 {float(row[5])*100:.1f}%"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _pct(x: Any) -> str:
    if x is None:
        return "N/A"
    return f"{float(x)*100:.1f}%"


def run_domain_adaptation(
    dataset_path: str,
    output_dir: str,
    *,
    do_zero_shot: bool = True,
    do_few_shot: bool = True,
    do_full_finetune: bool = True,
    do_kfold: bool = False,
    kfold_k: int = 5,
    kfold_cv_mode: str = "both",
    source_checkpoint: Optional[str] = None,
    epochs: int = 25,
    seed: int = 42,
    model: str = "resnet50",
    batch_size: int = 32,
) -> Dict[str, Any]:
    set_global_seeds(seed)
    os.makedirs(output_dir, exist_ok=True)

    source_info = find_best_cwru_checkpoint()
    ck = source_checkpoint or source_info["checkpoint"]
    source_info["checkpoint"] = ck

    summary_path = os.path.join(
        os.path.dirname(os.path.dirname(output_dir)),
        "domain_adaptation",
        "source_model_summary.md",
    )
    if not os.path.isfile(summary_path):
        write_source_model_summary(summary_path, source_info)

    run_config = {
        "dataset_path": dataset_path,
        "source_checkpoint": ck,
        "epochs": epochs,
        "seed": seed,
        "do_zero_shot": do_zero_shot,
        "do_few_shot": do_few_shot,
        "do_full_finetune": do_full_finetune,
        "do_kfold": do_kfold,
        "kfold_k": kfold_k,
        "kfold_cv_mode": kfold_cv_mode,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    table_rows: List[List[Any]] = []
    headers = ["Protocol", "Setting", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    if do_zero_shot:
        zm = run_zero_shot(
            dataset_path,
            output_dir,
            source_checkpoint=ck,
            model=model,
            batch_size=batch_size,
            seed=seed,
        )
        table_rows.append(
            [
                "zero_shot",
                "CWRU→MFPT",
                round(float(zm["accuracy"]), 4),
                round(float(zm["precision"]), 4),
                round(float(zm["recall"]), 4),
                round(float(zm["f1"]), 4),
                round(float(zm.get("auc_roc", 0) or 0), 4),
            ]
        )

    few_shot_results: List[Dict[str, Any]] = []
    if do_few_shot:
        few_shot_results = run_few_shot(
            dataset_path,
            output_dir,
            source_checkpoint=ck,
            fractions=FEW_SHOT_FRACTIONS,
            epochs=epochs,
            seed=seed + 100,
            model=model,
            batch_size=batch_size,
        )
        for r in few_shot_results:
            m = r["metrics"]
            table_rows.append(
                [
                    "few_shot",
                    f"{int(r['fraction']*100)}%",
                    round(float(m["accuracy"]), 4),
                    round(float(m["precision"]), 4),
                    round(float(m["recall"]), 4),
                    round(float(m["f1"]), 4),
                    round(float(m.get("auc_roc", 0) or 0), 4),
                ]
            )

    full_results: List[Dict[str, Any]] = []
    if do_full_finetune:
        full_results = run_full_finetune_comparison(
            dataset_path,
            output_dir,
            source_checkpoint=ck,
            epochs=epochs,
            seed=seed + 200,
            model=model,
            batch_size=batch_size,
        )
        for r in full_results:
            m = r["metrics"]
            table_rows.append(
                [
                    "full_finetune",
                    r["protocol"],
                    round(float(m["accuracy"]), 4),
                    round(float(m["precision"]), 4),
                    round(float(m["recall"]), 4),
                    round(float(m["f1"]), 4),
                    round(float(m.get("auc_roc", 0) or 0), 4),
                ]
            )

    kfold_results: Optional[Dict[str, Any]] = None
    if do_kfold:
        from experiments.run_mfpt_kfold_cv import run_kfold_cv

        kfold_dir = os.path.join(output_dir, "kfold")
        kfold_results = run_kfold_cv(
            dataset_path,
            kfold_dir,
            cv_mode=kfold_cv_mode,
            k=kfold_k,
            epochs=epochs,
            seed=seed + 300,
            model=model,
            batch_size=batch_size,
        )
        figures_dst = os.path.join(output_dir, "figures", "kfold")
        os.makedirs(figures_dst, exist_ok=True)
        for mode in (kfold_results.get("modes") or {}):
            fig_src = os.path.join(kfold_dir, mode, "figures")
            if os.path.isdir(fig_src):
                mode_dst = os.path.join(figures_dst, mode)
                if os.path.exists(mode_dst):
                    shutil.rmtree(mode_dst)
                shutil.copytree(fig_src, mode_dst)

    metrics_dir = os.path.join(output_dir, "metrics")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    save_table(
        table_rows,
        headers,
        os.path.join(output_dir, "domain_adaptation"),
        caption="CWRU to MFPT domain adaptation (ResNet50, log-Mel spectrograms)",
    )
    csv_path = os.path.join(output_dir, "domain_adaptation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(table_rows)

    if few_shot_results:
        fs_plot_rows = [
            [r["fraction"], "few_shot", r["metrics"]["accuracy"], r["metrics"]["precision"],
             r["metrics"]["recall"], r["metrics"]["f1"]]
            for r in few_shot_results
        ]
        _plot_few_shot_curve(fs_plot_rows, os.path.join(figures_dir, "few_shot_accuracy_curve"))
    if table_rows:
        _plot_protocol_bars(table_rows, os.path.join(figures_dir, "domain_adaptation_bars"))

    audit = f"""# MFPT domain adaptation audit

## Protocol
- Source: CWRU ResNet50 checkpoint `{ck}`
- Target: MFPT spectrograms at `{dataset_path}`
- Representation: log-Mel (same as CWRU pipeline)

## Experiments run
- Zero-shot: {"yes" if do_zero_shot else "no"}
- Few-shot (1/5/10/20%): {"yes" if do_few_shot else "no"}
- Full fine-tune vs scratch: {"yes" if do_full_finetune else "no"}
- K-fold CV (in-domain MFPT): {"yes" if do_kfold else "no"}

## Per-run figures (CWRU parity)
Training runs save: `training_curves.png`, `metric_curves.png` (precision/recall/F1), `confusion_matrix.png`, `roc_curve.png` (validation).
Test evaluation saves: `test_confusion_matrix.png`, `test_roc_curve.png`, `test_pr_curve.png`, `test_precrecf1_bar.png`.
K-fold saves: `figures/kfold/{{grouped,stratified}}/convergence_curves_*.png`, `fold_accuracy_*.png`, `cv_boxplot_*.png`.

## Results
See `domain_adaptation.csv` and `figures/`.
"""
    with open(os.path.join(output_dir, "audit_report.md"), "w", encoding="utf-8") as f:
        f.write(audit)

    _write_manuscript_text(os.path.join(output_dir, "manuscript_text.md"), table_rows, source_info)

    summary = {
        "output_dir": output_dir,
        "source_checkpoint": ck,
        "n_protocols": len(table_rows),
        "few_shot": few_shot_results,
        "full_finetune": full_results,
        "kfold": kfold_results,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "output_dir": output_dir,
                "n_rows": len(table_rows),
                "source_checkpoint": ck,
                "kfold_modes": list((kfold_results or {}).get("modes", {}).keys()),
            },
            f,
            indent=2,
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CWRU→MFPT domain adaptation experiments")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--source_checkpoint", type=str, default=None, help="CWRU checkpoint (auto-located if omitted)")
    p.add_argument("--do_zero_shot", action="store_true")
    p.add_argument("--do_few_shot", action="store_true")
    p.add_argument("--do_full_finetune", action="store_true")
    p.add_argument("--do_kfold", action="store_true", help="Run in-domain MFPT K-fold CV (convergence/boxplot curves)")
    p.add_argument("--kfold_k", type=int, default=5)
    p.add_argument("--kfold_cv_mode", type=str, default="both", choices=["stratified", "grouped", "both"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--batch_size", type=int, default=32)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    if not args.do_zero_shot and not args.do_few_shot and not args.do_full_finetune and not args.do_kfold:
        args.do_zero_shot = True
        args.do_few_shot = True
        args.do_full_finetune = True
    out = args.output_dir or os.path.join(
        "results/domain_adaptation", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_domain_adaptation(
        args.dataset_path,
        out,
        do_zero_shot=args.do_zero_shot,
        do_few_shot=args.do_few_shot,
        do_full_finetune=args.do_full_finetune,
        do_kfold=args.do_kfold,
        kfold_k=args.kfold_k,
        kfold_cv_mode=args.kfold_cv_mode,
        source_checkpoint=args.source_checkpoint,
        epochs=args.epochs,
        seed=args.seed,
        model=args.model,
        batch_size=args.batch_size,
    )
    logger.info("Domain adaptation complete: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
