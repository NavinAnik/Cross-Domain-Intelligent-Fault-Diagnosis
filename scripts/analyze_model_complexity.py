#!/usr/bin/env python3
"""Compute parameter counts and FLOPs for the best ResNet50 spectrogram model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from thop import profile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from modules.efficiency_utils import (
    count_parameters,
    ensure_subdirs,
    resolve_model_context,
    write_source_model_summary,
    build_eval_model,
)


def compute_flops(model: torch.nn.Module, image_size: int = 224) -> dict:
    model.eval()
    x = torch.randn(1, 3, image_size, image_size)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    gflops = macs / 1e9
    return {
        "macs": int(macs),
        "gflops": float(gflops),
        "flops": int(macs * 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze model complexity (params + FLOPs)")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--dataset_path", type=str, default=None)
    args = p.parse_args()

    dirs = ensure_subdirs()
    ctx = resolve_model_context(checkpoint=args.checkpoint, dataset_path=args.dataset_path)
    write_source_model_summary(ctx, dirs["root"])

    model = build_eval_model(ctx, torch.device("cpu"))
    params = count_parameters(model)
    flops = compute_flops(model, ctx.image_size)

    row = {
        "model": ctx.model_name,
        "backbone": ctx.backbone,
        "input_size": f"{ctx.image_size}x{ctx.image_size}",
        **params,
        **flops,
        "checkpoint": ctx.checkpoint,
    }

    csv_path = dirs["root"] / "model_complexity.csv"
    json_path = dirs["root"] / "model_complexity.json"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    print(f"Total parameters: {params['total_parameters']:,}")
    print(f"GFLOPs: {flops['gflops']:.4f}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
