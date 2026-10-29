"""Classify a short clip of hand landmarks, for J, Z and word signs.

Serves the GRU in `runs/clip/clip.pt`: 26 letters plus HELLO, NO, SORRY,
THANKYOU and YES. The input is the landmarks `/api/landmarks` already returned
for each preview frame. The model is not calibrated, so it has no confidence
threshold.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from slr.sources import REPO

from . import APP

DEFAULT_WEIGHTS = APP / "weights" / "clip.pt"
FALLBACK_WEIGHTS = REPO / "runs" / "clip" / "clip.pt"

# One second at the 12 fps the model was trained on.
MIN_FRAMES = 12
MAX_FRAMES = 300
# A motion letter needs a trace larger than ordinary landmark jitter.
MIN_TRACE_SPAN = 0.5  # palm lengths

_STATE: dict = {}


def resolve_checkpoint() -> Path | None:
    configured = os.environ.get("SLR_MOTION_WEIGHTS")
    if configured:
        return Path(configured)
    for p in (DEFAULT_WEIGHTS, FALLBACK_WEIGHTS):
        if p.exists():
            return p
    return None


def load() -> dict:
    """Load the clip model. A missing checkpoint is reported, not raised."""
    from slr import landmarks as LM, safe_load, seq

    path = resolve_checkpoint()
    if path is None or not path.exists():
        where = str(path) if path else f"{DEFAULT_WEIGHTS} or {FALLBACK_WEIGHTS}"
        _STATE.update(ok=False, error=(
            f"no clip checkpoint at {where}; set SLR_MOTION_WEIGHTS"))
        print(f"[motion] not available: {_STATE['error']}")
        return status()

    blob = safe_load(path)
    if blob.get("kind") != "clip_gru":
        _STATE.update(ok=False, error=f"{path} is a {blob.get('kind')!r} bundle, "
                                      "expected clip_gru")
        return status()

    classes = list(blob["classes"])
    # `seq.build` adds a CTC blank class, so pass one fewer.
    net = seq.build(len(classes) - 1, LM.FEATURE_DIM, blob.get("hidden", 128))
    net.load_state_dict(blob["state_dict"])
    net.eval()

    report = blob.get("report", {})
    _STATE.update(
        ok=True, error=None, net=net, classes=classes,
        mean=np.asarray(blob["mean"]), std=np.asarray(blob["std"]),
        ckpt=str(path), report=report,
    )
    print(f"[motion] clip_gru from {path}: {len(classes)} classes "
          f"({', '.join(c for c in classes if len(c) > 1)} and the 26 letters)")
    return status()


def status() -> dict:
    if not _STATE.get("ok"):
        return {"ok": False, "error": _STATE.get("error", "not loaded")}
    r = _STATE["report"]
    return {
        "ok": True,
        "n_classes": len(_STATE["classes"]),
        "classes": list(_STATE["classes"]),
        "min_frames": MIN_FRAMES,
        "protocol": r.get("protocol"),
        "quotable_as_accuracy": r.get("quotable_as_accuracy", False),
        "caveat": r.get("caveat"),
        "motion_worth": r.get("motion_worth"),
        "frame_classifier_plurality": r.get("frame_classifier_plurality"),
    }


class BadClip(Exception):
    """The posted sequence was not something this can read."""


def read(frames: list, sides: list | None = None, top_k: int = 3) -> dict:
    """Landmark frames in, one sign out.

    `frames` is (N, 21, 3) pixel coordinates, detected frames only. `sides` is
    the per-frame handedness, used to mirror left hands.
    """
    import torch

    from slr import landmarks as LM

    if not _STATE.get("ok"):
        raise BadClip(_STATE.get("error", "motion model not loaded"))

    # Check the count first so an empty recording gets a useful message.
    if len(frames) < MIN_FRAMES:
        raise BadClip(f"only {len(frames)} frames with a hand in them; this needs "
                      f"at least {MIN_FRAMES}. Keep your hand in frame while recording.")
    try:
        arr = np.asarray(frames, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise BadClip(f"landmarks are not a numeric (N, 21, 3) array: {e}") from e
    if arr.ndim != 3 or arr.shape[1:] != (21, 3):
        raise BadClip(f"expected (N, 21, 3) landmarks, got {list(arr.shape)}")
    if not np.isfinite(arr).all():
        raise BadClip("landmarks must be finite numbers")
    if len(arr) > MAX_FRAMES:
        arr = arr[:MAX_FRAMES]
        sides = sides[:MAX_FRAMES] if sides else None

    t0 = time.perf_counter()
    side = np.asarray(sides[:len(arr)]) if sides else None
    feats = LM.features(arr, side)

    with torch.no_grad():
        x = torch.tensor((feats - _STATE["mean"]) / _STATE["std"],
                         dtype=torch.float32)[None]
        # Average over frames, as in training.
        probs = _STATE["net"](x)[0].mean(0).softmax(0)

    k = min(top_k, probs.numel())
    vals, idx = probs.topk(k)
    label = _STATE["classes"][int(idx[0])]
    needs_movement = False
    if label in {"J", "Z"}:
        tip = 20 if label == "J" else 8
        palm = np.median(np.linalg.norm(arr[:, 9, :2] - arr[:, 0, :2], axis=1))
        lo, hi = np.percentile(arr[:, tip, :2], [5, 95], axis=0)
        needs_movement = palm <= 1e-6 or np.linalg.norm(hi - lo) / palm < MIN_TRACE_SPAN
    return {
        "label": label,
        "confidence": round(float(vals[0]), 4),
        "calibrated": False,
        "abstain": bool(needs_movement),
        "reason": (f"{label} needs movement. Record its complete moving trace."
                   if needs_movement else None),
        "top_k": [{"label": _STATE["classes"][int(i)], "p": round(float(v), 4)}
                  for v, i in zip(vals, idx)],
        "frames": int(len(arr)),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "model": "clip_gru",
    }
