"""Evaluation, calibration, and the abstention threshold.

A phone app that confidently announces the wrong letter is worse than one that
says "hold still, I'm not sure". So the deliverable is not a single accuracy
number, it is a calibrated confidence plus a threshold with a known error rate
above it.

Three things happen here:

  temperature scaling - one scalar fitted on the validation split, dividing the
      logits. Fine-tuned ViTs are badly overconfident; without this the softmax
      score is not a usable probability and any threshold picked from it is
      arbitrary.

  risk-coverage      - sweep the threshold and report, for each level of
      coverage (fraction of photos the model is willing to answer), the error
      rate among the answers it does give. This is the curve to quote in the
      report, and the row the app's threshold is read off.

  per-class report   - which letters actually fail. On honest splits the
      failures are the visually confusable pairs (M/N/S/T/E, D/1, U/V/R), which
      is the sanity check that the model learned handshape and not wallpaper.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data, hands, model as M, sources


@torch.no_grad()
def collect_logits(net: nn.Module, loader, dev: torch.device
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    net.eval()
    L, Y = [], []
    for x, y in loader:
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            L.append(net(x.to(dev)).float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, max_iter: int = 200) -> float:
    """Guo et al. temperature scaling: minimise NLL over a single scalar.

    Fitted on validation only. Fitting it on test would be a second, subtler
    form of the leakage this project is about.
    """
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(correct[m].mean() - conf[m].mean()).item()
    return ece


def risk_coverage(probs: torch.Tensor, y: torch.Tensor,
                  coverages=(1.0, 0.95, 0.9, 0.8, 0.7, 0.5)) -> list[dict]:
    """For each target coverage, the confidence threshold that achieves it and
    the error rate among the retained predictions."""
    conf, pred = probs.max(1)
    correct = (pred == y)
    order = conf.argsort(descending=True)
    conf, correct = conf[order], correct[order]
    n = len(conf)
    out = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        acc = float(correct[:k].float().mean())
        out.append({
            "coverage": round(k / n, 4),
            "threshold": round(float(conf[k - 1]), 4),
            "accuracy": round(acc, 4),
            "error": round(1 - acc, 4),
        })
    return out


def per_class(pred: torch.Tensor, y: torch.Tensor) -> dict:
    classes = sources.CLASSES
    n = len(classes)
    conf = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    rows = {}
    for i, c in enumerate(classes):
        support = int(conf[i].sum())
        if not support:
            continue
        tp = int(conf[i, i])
        prec = tp / max(int(conf[:, i].sum()), 1)
        rec = tp / support
        off = conf[i].clone()
        off[i] = 0
        rows[c] = {
            "support": support, "recall": round(rec, 4), "precision": round(prec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "most_confused_with": classes[int(off.argmax())] if int(off.sum()) else None,
        }
    return rows


def run(ckpt: Path, manifest: Path, out: Path, workers: int = 4) -> dict:
    dev = M.device()
    blob = torch.load(ckpt, map_location=dev)
    net = M.build(blob["preset"], len(blob["classes"]), pretrained=False).to(dev)
    net.load_state_dict(blob["state_dict"])

    rows = sources.read_manifest(manifest)
    # A crop-trained model scored on uncropped images (or the reverse) produces a
    # real number for a pipeline nobody will ever run. Loud, because the result
    # looks perfectly plausible - just several points off, in whichever direction
    # flatters the mistake.
    cropped = all(str(hands.CROP_ROOT) in r["path"] for r in rows)
    if cropped != bool(blob.get("hand_crop")):
        print(f"[eval] WARNING: checkpoint hand_crop={bool(blob.get('hand_crop'))} "
              f"but this manifest is {'cropped' if cropped else 'uncropped'}. "
              f"Train and eval preprocessing disagree; this number is not comparable.")
    dl = data.loaders(rows, blob["img_size"], M.PRESETS[blob["preset"]].batch_size, workers)

    temp = 1.0
    if "val" in dl:
        vl, vy = collect_logits(net, dl["val"], dev)
        temp = fit_temperature(vl, vy)
        print(f"[eval] temperature = {temp:.3f} (>1 means the raw model was overconfident)")

    report: dict = {"checkpoint": str(ckpt), "preset": blob["preset"], "temperature": temp}
    for split in ("val", "test"):
        if split not in dl:
            continue
        logits, y = collect_logits(net, dl[split], dev)
        raw = logits.softmax(1)
        cal = (logits / temp).softmax(1)
        pred = cal.argmax(1)
        report[split] = {
            "n": len(y),
            "accuracy": round(float((pred == y).float().mean()), 4),
            "ece_raw": round(expected_calibration_error(raw, y), 4),
            "ece_calibrated": round(expected_calibration_error(cal, y), 4),
            "risk_coverage": risk_coverage(cal, y),
            "per_class": per_class(pred, y),
            "sources": sorted({r["source"] for r in rows if r["split"] == split}),
        }
        print(f"[eval] {split}: acc={report[split]['accuracy']:.4f} "
              f"ECE {report[split]['ece_raw']:.4f} -> {report[split]['ece_calibrated']:.4f}")
        for r in report[split]["risk_coverage"]:
            print(f"[eval]   coverage {r['coverage']:.0%} @ conf>={r['threshold']:.3f} "
                  f"-> error {r['error']:.2%}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[ok] {out}")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="calibrated evaluation with abstention")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/eval.json"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    run(a.checkpoint, a.manifest, a.out, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
