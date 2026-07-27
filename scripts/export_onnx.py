#!/usr/bin/env python3
"""Export best model to ONNX and validate inference parity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from modules.efficiency_utils import (
    build_eval_model,
    ensure_subdirs,
    file_size_mb,
    make_split_loaders,
    resolve_model_context,
)


def export_onnx(model: torch.nn.Module, out_path: Path, image_size: int = 224) -> None:
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    export_kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(out_path), **export_kwargs)


def validate_onnx(
    pt_model: torch.nn.Module,
    onnx_path: Path,
    sample_batch: torch.Tensor,
) -> dict:
    onnx.checker.check_model(str(onnx_path))

    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(sample_batch).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": sample_batch.cpu().numpy()})[0]

    max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
    pt_pred = np.argmax(pt_out, axis=1)
    ort_pred = np.argmax(ort_out, axis=1)
    agreement = float(np.mean(pt_pred == ort_pred))

    return {
        "max_abs_diff": max_abs_diff,
        "argmax_agreement": agreement,
        "pt_shape": list(pt_out.shape),
        "ort_shape": list(ort_out.shape),
    }


def write_report(
    out_path: Path,
    *,
    ctx,
    pt_size_mb: float,
    onnx_size_mb: float,
    parity: dict,
    export_ok: bool,
) -> None:
    reduction = (1.0 - onnx_size_mb / pt_size_mb) * 100 if pt_size_mb > 0 else 0.0
    lines = [
        "# ONNX Export Report",
        "",
        f"- **Source checkpoint:** `{ctx.checkpoint}`",
        f"- **ONNX path:** `{out_path}`",
        f"- **Export success:** {'Yes' if export_ok else 'No'}",
        "",
        "## Model Size",
        "",
        f"| Format | Size (MB) |",
        f"|--------|-----------|",
        f"| PyTorch (.pt) | {pt_size_mb:.2f} |",
        f"| ONNX | {onnx_size_mb:.2f} |",
        f"| Size change | {reduction:+.1f}% |",
        "",
        "## Inference Parity (PyTorch vs ONNX Runtime)",
        "",
        f"- **Max absolute difference:** {parity['max_abs_diff']:.6e}",
        f"- **Argmax agreement:** {parity['argmax_agreement'] * 100:.2f}%",
        f"- **Output shape (PyTorch):** {parity['pt_shape']}",
        f"- **Output shape (ONNX):** {parity['ort_shape']}",
        "",
        "## Compatibility",
        "",
        "ONNX export uses opset 17 with dynamic batch dimension. Validated with ONNX Runtime "
        "CPUExecutionProvider on a real validation batch.",
        "",
    ]
    report_path = out_path.parent / "onnx_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Export model to ONNX")
    p.add_argument("--checkpoint", type=str, default=None)
    args = p.parse_args()

    dirs = ensure_subdirs()
    ctx = resolve_model_context(checkpoint=args.checkpoint)
    out_path = dirs["onnx"] / "best_model.onnx"

    model = build_eval_model(ctx, torch.device("cpu"))
    export_onnx(model, out_path, ctx.image_size)

    _, val_loader, _ = make_split_loaders(ctx, batch_size=8, device=torch.device("cpu"))
    sample_x, _ = next(iter(val_loader))

    parity = validate_onnx(model, out_path, sample_x)
    pt_size = file_size_mb(ctx.checkpoint)
    onnx_size = file_size_mb(str(out_path))
    write_report(out_path, ctx=ctx, pt_size_mb=pt_size, onnx_size_mb=onnx_size, parity=parity, export_ok=True)

    print(f"Exported ONNX to {out_path}")
    print(f"Parity: max_diff={parity['max_abs_diff']:.2e}, agreement={parity['argmax_agreement']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
