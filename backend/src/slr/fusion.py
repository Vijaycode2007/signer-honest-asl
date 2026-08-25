"""Pixel/per-finger probability fusion and ratio selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn


# Which landmark representation a bundle was trained on, and how wide it is.
# Declared in the bundle rather than inferred, so a model trained on one
# representation can never be served coordinates from another - the failure that
# produces plausible-looking nonsense rather than an error.
GEOMETRY_FEATURES = {
    "per_finger_geometry_v1": 35,
    "normalised_landmarks_v1": 63,
    "coords_plus_finger_v1": 98,
}


def _geometry_matrix(feature: str, lm: np.ndarray, side=None) -> np.ndarray:
    from . import landmarks

    if feature == "per_finger_geometry_v1":
        return landmarks.finger_features(lm, side)
    if feature == "normalised_landmarks_v1":
        return landmarks.features(lm, side)
    if feature == "coords_plus_finger_v1":
        return np.concatenate([landmarks.features(lm, side),
                               landmarks.finger_features(lm, side)], axis=1)
    raise ValueError(f"unknown geometry feature {feature!r}")


@dataclass(frozen=True)
class GeometryBundle:
    net: nn.Module
    mean: np.ndarray
    std: np.ndarray
    classes: list[str]
    temperature: float
    feature: str = "per_finger_geometry_v1"


def fuse_probabilities(pixel: np.ndarray, geometry: np.ndarray,
                       pixel_weight: float) -> np.ndarray:
    """Linear probability pool; `pixel_weight=1` exactly reproduces pixels."""
    p = np.asarray(pixel, dtype=np.float64)
    g = np.asarray(geometry, dtype=np.float64)
    if p.shape != g.shape or p.ndim != 2:
        raise ValueError(f"probability shapes differ: pixel={p.shape}, geometry={g.shape}")
    if not 0.0 <= pixel_weight <= 1.0:
        raise ValueError(f"pixel_weight must be in [0, 1], got {pixel_weight}")
    if not np.isfinite(p).all() or not np.isfinite(g).all() or (p < 0).any() or (g < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    out = pixel_weight * p + (1.0 - pixel_weight) * g
    total = out.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("each probability row must have positive mass")
    return out / total


def align_probabilities(probs: np.ndarray, from_classes: Sequence[str],
                        to_classes: Sequence[str]) -> np.ndarray:
    """Reorder probability columns by label, refusing incomplete vocabularies."""
    p = np.asarray(probs)
    if p.ndim != 2 or p.shape[1] != len(from_classes):
        raise ValueError(f"{p.shape} does not match {len(from_classes)} source classes")
    if len(set(from_classes)) != len(from_classes):
        raise ValueError("source classes contain duplicates")
    missing = set(to_classes) - set(from_classes)
    extra = set(from_classes) - set(to_classes)
    if missing or extra:
        raise ValueError(f"class vocabularies differ: missing={sorted(missing)}, extra={sorted(extra)}")
    index = {name: i for i, name in enumerate(from_classes)}
    return p[:, [index[name] for name in to_classes]]


def choose_ratio(sweep: list[dict], max_ab_regression: float = 0.01) -> dict:
    """Maximise C macro-F1 subject to bounded A/B loss against pixel-only."""
    baseline = next((r for r in sweep if abs(float(r["pixel_weight"]) - 1.0) < 1e-12), None)
    if baseline is None:
        raise ValueError("ratio sweep must include pixel_weight=1.0 as the baseline")
    eligible = [
        row for row in sweep
        if all(float(row[p]["macro_f1"]) >=
               float(baseline[p]["macro_f1"]) - max_ab_regression - 1e-12
               for p in ("A", "B"))
    ]
    return max(eligible, key=lambda r: (
        float(r["C"]["macro_f1"]),
        (float(r["A"]["macro_f1"]) + float(r["B"]["macro_f1"])) / 2,
        float(r["pixel_weight"]),
    ))


def geometry_fallback(probs: np.ndarray, found: np.ndarray,
                      classes: Sequence[str]) -> np.ndarray:
    """Turn detector silence into the semantic `nothing` class."""
    out = np.asarray(probs, dtype=np.float64).copy()
    detected = np.asarray(found, dtype=bool)
    if out.ndim != 2 or len(out) != len(detected) or out.shape[1] != len(classes):
        raise ValueError("probabilities, found mask, and class vocabulary do not align")
    try:
        nothing = list(classes).index("nothing")
    except ValueError as e:
        raise ValueError("geometry classes must include 'nothing'") from e
    out[~detected] = 0.0
    out[~detected, nothing] = 1.0
    return out


def classification_metrics(probs: np.ndarray, truth: np.ndarray) -> dict:
    """Accuracy and macro-F1 over labels represented in the truth split."""
    p = np.asarray(probs)
    y = np.asarray(truth, dtype=np.int64)
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("probabilities and truth do not align")
    pred = p.argmax(axis=1)
    f1 = []
    for c in np.unique(y):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()) if len(y) else 0.0,
        "macro_f1": float(np.mean(f1)) if f1 else 0.0,
    }


def partition_tuning_rows(rows: list[dict], tune_fraction: float = 0.3,
                          seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Carve group-disjoint ratio-tuning rows from the cross-corpus holdout."""
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("tune_fraction must be between zero and one")
    from . import data

    copied = [dict(r, split="") for r in rows]
    data.split_by_group(copied, val=tune_fraction, test=0.0,
                        seed=seed)
    return ([r for r in copied if r["split"] == "val"],
            [r for r in copied if r["split"] == "train"])


def load_geometry(path: str | Path) -> GeometryBundle:
    """Load the per-finger MLP with its scaler and calibration constant."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    return geometry_from_blob(blob)


def geometry_from_blob(blob: dict) -> GeometryBundle:
    """Build a geometry model from either a standalone or embedded bundle."""
    from . import landmarks

    feature = blob.get("feature")
    if feature not in GEOMETRY_FEATURES:
        raise ValueError(f"bundle declares geometry feature {feature!r}, which is "
                         f"not one of {sorted(GEOMETRY_FEATURES)}")
    dim = GEOMETRY_FEATURES[feature]
    mean = np.asarray(blob["mean"])
    std = np.asarray(blob["std"])
    if mean.shape != (dim,) or std.shape != (dim,):
        raise ValueError(f"{feature} scaler must have {dim} values, "
                         f"got {mean.shape}/{std.shape}")
    net = landmarks.build(len(blob["classes"]), in_dim=dim,
                          hidden=int(blob.get("hidden", 256)))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return GeometryBundle(net, mean, std, list(blob["classes"]),
                          float(blob.get("temperature", 1.0)), feature)


@torch.no_grad()
def geometry_probabilities(bundle: GeometryBundle, lm: np.ndarray,
                           found: np.ndarray, side=None) -> np.ndarray:
    """Score cached landmarks, with detector misses mapped to `nothing`."""
    x = _geometry_matrix(bundle.feature, lm, side)
    xt = torch.as_tensor((x - bundle.mean) / bundle.std, dtype=torch.float32)
    probs = torch.softmax(bundle.net(xt) / bundle.temperature, dim=1).cpu().numpy()
    return geometry_fallback(probs, found, bundle.classes)


def make_protocol_rows(cross_rows: list[dict], protocol: str,
                       train_source: str, holdout_source: str,
                       seed: int = 0) -> list[dict]:
    """Build A/B/C assignments over one fixed train-source image pool."""
    from . import data

    train_rows = [dict(r, split="") for r in cross_rows if r["source"] == train_source]
    if protocol == "A":
        rng = random.Random(seed)
        for row in train_rows:
            q = rng.random()
            row["split"] = "train" if q < 0.70 else ("val" if q < 0.85 else "test")
        return train_rows
    if protocol == "B":
        return data.split_by_group(train_rows, val=0.15, test=0.15,
                                   seed=seed)
    if protocol == "C":
        holdout = [dict(r, split="") for r in cross_rows
                   if r["source"] == holdout_source]
        return data.split_cross_source(train_rows + holdout, [train_source],
                                       [holdout_source], val=0.15, seed=seed)
    raise ValueError(f"unknown protocol {protocol!r}; expected A, B, or C")


def cached_landmarks_for_rows(cache: dict, rows: list[dict]) -> dict:
    """Join cached geometry to manifest rows by absolute image path."""
    if "path" not in cache:
        raise ValueError("landmark cache has no paths; rebuild it before fusion")
    paths = [str(p) for p in cache["path"]]
    if len(set(paths)) != len(paths):
        raise ValueError("landmark cache contains duplicate paths")
    index = {path: i for i, path in enumerate(paths)}
    wanted = [str(r["path"]) for r in rows]
    missing = [path for path in wanted if path not in index]
    if missing:
        raise ValueError(f"landmark cache is missing {len(missing)} paths; first={missing[0]}")
    take = np.asarray([index[path] for path in wanted], dtype=np.int64)
    return {key: np.asarray(cache[key])[take] for key in ("lm", "found", "side")}


def train_geometry(rows: list[dict], cache: dict, out: str | Path,
                   epochs: int = 160, seed: int = 0,
                   hidden: int = 256) -> dict:
    """Fit and calibrate the 35-feature per-finger MLP for one protocol."""
    from . import evaluate, landmarks, sources

    aligned = cached_landmarks_for_rows(cache, rows)
    x = landmarks.finger_features(aligned["lm"], aligned["side"])
    found = aligned["found"].astype(bool)
    y = np.asarray([sources.CLASS_TO_IDX[r["label"]] for r in rows], dtype=np.int64)
    split = np.asarray([r["split"] for r in rows])
    train_idx = np.where((split == "train") & found)[0]
    val_found = np.where((split == "val") & found)[0]
    if len(train_idx) < 2:
        raise ValueError("geometry training needs at least two detected train hands")

    net, mean, std = landmarks.fit(
        x[train_idx], y[train_idx], len(sources.CLASSES), epochs=epochs,
        seed=seed, hidden=hidden,
    )
    temperature = 1.0
    if len(val_found):
        with torch.no_grad():
            vx = torch.as_tensor((x[val_found] - mean) / std, dtype=torch.float32)
            logits = net(vx)
        temperature = evaluate.fit_temperature(
            logits, torch.as_tensor(y[val_found]), max_iter=50)

    bundle = GeometryBundle(net, mean, std, list(sources.CLASSES), temperature)
    report: dict = {
        "feature": "per_finger_geometry_v1",
        "temperature": temperature,
        "n_train": int(len(train_idx)),
    }
    for name in ("val", "test"):
        idx = np.where(split == name)[0]
        if len(idx):
            probs = geometry_probabilities(
                bundle, aligned["lm"][idx], found[idx], aligned["side"][idx])
            report[name] = classification_metrics(probs, y[idx])

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(), "classes": list(sources.CLASSES),
        "mean": mean, "std": std, "hidden": hidden,
        "feature": "per_finger_geometry_v1", "temperature": temperature,
        "report": report,
    }, path)
    return report


def sweep_probabilities(protocols: dict[str, dict],
                        ratios: Sequence[float] | None = None) -> list[dict]:
    """Score a shared pixel weight over aligned A/B/C probability caches."""
    weights = list(ratios) if ratios is not None else [i / 100 for i in range(101)]
    required = {"A", "B", "C"}
    if set(protocols) != required:
        raise ValueError(f"need protocol inputs {sorted(required)}, got {sorted(protocols)}")
    out = []
    for weight in weights:
        row: dict = {"pixel_weight": float(weight),
                     "geometry_weight": float(1.0 - weight)}
        for name in ("A", "B", "C"):
            item = protocols[name]
            probs = fuse_probabilities(item["pixel"], item["geometry"], weight)
            row[name] = classification_metrics(probs, item["truth"])
        out.append(row)
    return out
