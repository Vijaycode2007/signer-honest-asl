"""Train DeiT III and write the bundle the server loads.

    python -m sign_reader.train dataset/manifests/all.csv --epochs 8

Needs a GPU. Writes `apps/sign-reader/weights/deit3_base.pt` with the weights,
class list and calibration. The threshold is picked on the validation split.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from slr import evaluate, model as M, safe_load, sources, train as T
from slr.sources import REPO

from . import APP

PRESET = "deit3_base"


def pick_threshold(report: dict, split: str, coverage: float) -> tuple[float, dict]:
    """The confidence cut-off that answers `coverage` of photos on `split`."""
    rows = report.get(split, {}).get("risk_coverage", [])
    if not rows:
        return 0.0, {}
    best = min(rows, key=lambda r: abs(r["coverage"] - coverage))
    return float(best["threshold"]), best


def bundle(run: Path, report: dict, out: Path, coverage: float,
           threshold_split: str) -> Path:
    """Combine the checkpoint and calibration into one file."""
    blob = safe_load(run / "best.pt")
    thresh, row = pick_threshold(report, threshold_split, coverage)

    result = json.loads((run / "result.json").read_text())
    metrics = {
        "val_f1": result.get("best", {}).get("val_f1"),
        "val_acc": result.get("best", {}).get("val_acc"),
        "test": result.get("test"),
        "params_m": result.get("params_m"),
        "operating_point": {**row, "picked_on": threshold_split},
        "ece_raw": report.get(threshold_split, {}).get("ece_raw"),
        "ece_calibrated": report.get(threshold_split, {}).get("ece_calibrated"),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "preset": blob["preset"],
        "classes": blob["classes"],
        "img_size": blob["img_size"],
        # fp16 halves the download; the server widens it back to fp32 at load.
        "state_dict": {k: (v.half() if v.is_floating_point() else v)
                       for k, v in blob["state_dict"].items()},
        "temperature": report.get("temperature", 1.0),
        "threshold": thresh,
        "coverage": row.get("coverage"),
        "metrics": metrics,
    }, out)

    mb = out.stat().st_size / 1e6
    print(f"\n[bundle] {out} ({mb:.0f} MB)")
    print(f"[bundle] temperature {report.get('temperature', 1.0):.3f}, "
          f"abstain below {thresh:.3f}")
    if row:
        print(f"[bundle] at that cut-off, on {threshold_split}: answers "
              f"{row['coverage']:.0%} of photos at {row['error']:.1%} error")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest", type=Path, nargs="?",
                    help="dataset manifest CSV with a split column")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=APP / "weights" / "deit3_base.pt")
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--coverage", type=float, default=0.9,
                    help="fraction of photos the service should be willing to answer")
    ap.add_argument("--threshold-split", default="val", choices=["val", "test"],
                    help="split the operating point is read off. Keep this on val")
    ap.add_argument("--from-run", type=Path,
                    help="skip training and bundle an existing run directory")
    ap.add_argument("--batch-size", type=int,
                    help="override the preset batch size. The presets are sized "
                         "for a 16 GB discrete card; 8 GB of shared Apple memory "
                         "needs roughly 8")
    ap.add_argument("--lr", type=float,
                    help="override the preset learning rate. The presets assume "
                         "batch 48; a much smaller batch needs a smaller rate")
    a = ap.parse_args(argv)

    if a.from_run:
        run = a.from_run
        report_path = run / "eval.json"
        if not report_path.exists():
            if not a.manifest:
                ap.error("--from-run without an eval.json needs the manifest to make one")
            evaluate.run(run / "best.pt", a.manifest, report_path, a.workers)
        report = json.loads(report_path.read_text())
        bundle(run, report, a.out, a.coverage, a.threshold_split)
        return 0

    if not a.manifest:
        ap.error("a manifest is required unless --from-run is given")

    p = M.PRESETS[PRESET]
    if a.batch_size or a.lr:
        M.PRESETS[PRESET] = p = dataclasses.replace(
            p, batch_size=a.batch_size or p.batch_size, lr=a.lr or p.lr)
    print(f"[train] {PRESET} ({p.timm_name}), {p.approx_params_m}M params, "
          f"{p.img_size}px, batch {p.batch_size}, lr {p.lr}, "
          f"layer_decay {p.layer_decay}")
    if M.device().type == "cpu":
        print("[train] WARNING: no GPU, this will take days on CPU")

    rows = sources.read_manifest(a.manifest)
    tag = f"deploy_{PRESET}"
    T.train_one(rows, PRESET, epochs=a.epochs, out_dir=a.runs, workers=a.workers,
                tag=tag, eval_test=True)

    run = a.runs / tag
    report = evaluate.run(run / "best.pt", a.manifest, run / "eval.json", a.workers)
    bundle(run, report, a.out, a.coverage, a.threshold_split)
    print("\n[ok] serve it with:  uv run uvicorn sign_reader.main:app --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
