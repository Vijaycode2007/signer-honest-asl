from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data, hands, model as M, safe_load, sources


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
    """Guo et al. temperature scaling: minimise NLL over a single scalar."""
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


def calibrate_fused(fused: torch.Tensor, y: torch.Tensor, tune: torch.Tensor,
                    target_coverage: float = 0.9, eps: float = 1e-6) -> dict:
    """Fit a temperature and abstention threshold for the fused probabilities.

    Fits on the rows where `tune` is True and reports on the rest.
    """
    tune = tune.bool()
    if not tune.any() or tune.all():
        raise ValueError("need both tuning rows and reporting rows")
    logp = fused.clamp_min(eps).log()
    temp = fit_temperature(logp[tune], y[tune])
    cal = (logp / temp).softmax(1)

    rows = risk_coverage(cal[tune], y[tune], coverages=(target_coverage,))
    threshold = rows[0]["threshold"]

    def at(mask: torch.Tensor) -> dict:
        conf, pred = cal[mask].max(1)
        keep = conf >= threshold
        correct = (pred == y[mask])
        return {
            "n": int(mask.sum()),
            "accuracy": round(float(correct.float().mean()), 4),
            "coverage": round(float(keep.float().mean()), 4),
            "accuracy_among_accepted": (
                round(float(correct[keep].float().mean()), 4) if keep.any() else None),
            "nll": round(float(-cal[mask][torch.arange(int(mask.sum())), y[mask]]
                               .clamp_min(eps).log().mean()), 4),
            "ece": round(expected_calibration_error(cal[mask], y[mask]), 4),
        }

    report = {
        "temperature": round(temp, 4),
        "threshold": threshold,
        "target_coverage": target_coverage,
        "fitted_on": "group-disjoint tuning half of the holdout corpus",
        "eps": eps,
        "tune": at(tune),
        "report": at(~tune),
        "uncalibrated_report_nll": round(
            float(-fused[~tune][torch.arange(int((~tune).sum())), y[~tune]]
                  .clamp_min(eps).log().mean()), 4),
    }
    print(f"[fuse-cal] T={temp:.4f} threshold={threshold:.4f} "
          f"(target coverage {target_coverage:.0%})")
    for half in ("tune", "report"):
        r = report[half]
        print(f"[fuse-cal]   {half:<6} n={r['n']:<5} coverage={r['coverage']:.4f} "
              f"acc among accepted={r['accuracy_among_accepted']} "
              f"nll={r['nll']:.4f} ece={r['ece']:.4f}")
    return report


def mcnemar_exact(a_right: np.ndarray, b_right: np.ndarray) -> dict:
    """Exact two-sided McNemar for two classifiers on the same rows."""
    from math import comb

    b = int((a_right & ~b_right).sum())
    c = int((~a_right & b_right).sum())
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"b": b, "c": c, "p_value": round(min(1.0, 2 * tail), 4)}


def paired_bootstrap(a_right: np.ndarray, b_right: np.ndarray, groups: np.ndarray,
                     n_boot: int = 10_000, seed: int = 0) -> dict:
    """95% interval for (b - a) accuracy, resampling whole groups."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    by_group = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([by_group[g] for g in
                               rng.integers(0, len(by_group), len(by_group))])
        diffs[i] = b_right[pick].mean() - a_right[pick].mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"delta": round(float(b_right.mean() - a_right.mean()), 4),
            "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
            "n_groups": int(len(uniq)), "n_boot": n_boot,
            "resampled": "whole near-duplicate components, not rows"}


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


def letter_reliability(probs: np.ndarray, y: np.ndarray, classes: list[str],
                       found: np.ndarray | None = None, threshold: float = 0.0,
                       letters: list[str] | None = None,
                       eligible: np.ndarray | None = None) -> dict:
    """Per-letter accuracy, detector misses and confident mistakes."""
    p, y = np.asarray(probs), np.asarray(y)
    if (p.ndim != 2 or y.ndim != 1 or p.shape != (len(y), len(classes)) or not len(y)
            or not np.issubdtype(y.dtype, np.integer)
            or len(set(classes)) != len(classes) or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(1), 1, atol=1e-5)
            or (y < 0).any() or (y >= len(classes)).any()
            or not 0 <= threshold <= 1):
        raise ValueError("need finite probability rows, valid class indices and threshold in [0,1]")
    found = np.ones(len(y), dtype=bool) if found is None else np.asarray(found, dtype=bool)
    if found.shape != y.shape:
        raise ValueError("found must have one value per example")
    letters = list("ABCDEFGHIKLMNOPQRSTUVWXY") if letters is None else letters
    pred, conf = p.argmax(1), p.max(1)
    correct, accepted = pred == y, conf >= threshold
    if eligible is not None:
        eligible = np.asarray(eligible, dtype=bool)
        if eligible.shape != y.shape:
            raise ValueError("eligible must have one value per example")
        accepted &= eligible
    names = np.asarray(classes)
    per = {}
    for c in letters:
        rows = names[y] == c
        n = int(rows.sum())
        guessed = names[pred] == c
        per[c] = {
            "n": n, "correct": int((correct & rows).sum()),
            "recall": float(correct[rows].mean()) if n else None,
            "precision": float(correct[guessed].mean()) if guessed.any() else None,
            "detector_misses": int((~found & rows).sum()),
            "confident_errors": int((rows & ~correct & (conf >= 0.9)).sum()),
        }
    pairs = Counter(zip(names[y[~correct]], names[pred[~correct]]))
    return {
        "n": len(y), "accuracy": float(correct.mean()),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "confident_errors": int((~correct & (conf >= 0.9)).sum()),
        "nll": float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
        "ece": expected_calibration_error(torch.from_numpy(p), torch.from_numpy(y)),
        "complete": all(r["n"] for r in per.values()),
        "per_letter": per,
        "weakest_letters": sorted(per, key=lambda c: (
            per[c]["recall"] is not None,
            per[c]["recall"] if per[c]["recall"] is not None else -1, c)),
        "confusions": [{"truth": a, "predicted": b, "n": n}
                       for (a, b), n in pairs.most_common()],
    }


def run(ckpt: Path, manifest: Path, out: Path, workers: int = 4) -> dict:
    dev = M.device()
    blob = safe_load(ckpt, map_location=dev)
    net = M.build(blob["preset"], len(blob["classes"]), pretrained=False).to(dev)
    net.load_state_dict(blob["state_dict"])

    rows = sources.read_manifest(manifest)
    cropped = all(str(hands.CROP_ROOT) in r["path"] for r in rows)
    if cropped != bool(blob.get("hand_crop")):
        print(f"[eval] WARNING: checkpoint hand_crop={bool(blob.get('hand_crop'))} "
              f"but this manifest is {'cropped' if cropped else 'uncropped'}")
    dl = data.loaders(rows, blob["img_size"], M.PRESETS[blob["preset"]].batch_size, workers)

    temp = 1.0
    if "val" in dl:
        vl, vy = collect_logits(net, dl["val"], dev)
        temp = fit_temperature(vl, vy)
        print(f"[eval] temperature = {temp:.3f}")

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
