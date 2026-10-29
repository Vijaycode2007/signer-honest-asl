from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from . import hands, sources

WRIST = 0
FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
           "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
MIDDLE_MCP = 9

CACHE = sources.DEFAULT_ROOT.parent / "landmarks"

V1 = "normalised_landmarks_v1"
V2 = "landmark_distances_v2"
TRAIN_SOURCES = ("asl_alphabet", "signalphaset", "asl_dynamic")


def canonical_side(lm: np.ndarray, side=None) -> np.ndarray:
    """Which rows are left hands. No `side` means right."""
    n = 1 if lm.ndim == 2 else len(lm)
    if side is None:
        side = "Right"
    if isinstance(side, str):
        return np.full(n, side == "Left")
    return np.asarray(side) == "Left"


def normalise(lm: np.ndarray, side=None, rotate: bool = False) -> np.ndarray:
    """(N, 21, 3) pixel landmarks -> wrist at the origin, palm length 1, right-handed.

    `rotate` also spins the palm to point up.
    """
    left = canonical_side(lm, side)
    if lm.ndim == 2:
        lm = lm[None]
    p = np.asarray(lm) - np.asarray(lm)[:, WRIST:WRIST + 1]
    p[left, :, 0] *= -1
    palm = np.linalg.norm(p[:, MIDDLE_MCP], axis=-1)[:, None, None]
    p = p / (palm + 1e-8)
    if not rotate:
        return p
    a = p[:, MIDDLE_MCP, :2]
    c, sn = a[:, 1:2], a[:, 0:1]
    n = np.sqrt(c ** 2 + sn ** 2) + 1e-8
    c, sn = (c / n)[:, None, 0], (sn / n)[:, None, 0]
    x, y = p[..., 0].copy(), p[..., 1].copy()
    p[..., 0] = c * x - sn * y
    p[..., 1] = sn * x + c * y
    return p


def _coords(p: np.ndarray) -> np.ndarray:
    return p.reshape(len(p), -1)


_DIST_POINTS = [4, 8, 12, 16, 20, WRIST, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
_UPPER = np.triu_indices(len(_DIST_POINTS), 1)


def _coords_and_distances(p: np.ndarray) -> np.ndarray:
    q = p[:, _DIST_POINTS]
    d = np.linalg.norm(q[:, :, None] - q[:, None], axis=-1)
    return np.concatenate([_coords(p), d[:, _UPPER[0], _UPPER[1]]], 1)


class Feature(NamedTuple):
    fn: Callable[[np.ndarray], np.ndarray]
    dim: int


FEATURES = {
    V1: Feature(_coords, 3 * 21),
    V2: Feature(_coords_and_distances, 3 * 21 + len(_UPPER[0])),
}
FEATURE_DIM = FEATURES[V1].dim


def features(lm: np.ndarray, side=None, feature: str = V1,
             rotate: bool = False) -> np.ndarray:
    """The vector a classifier sees, for the named feature tag. (N, dim)."""
    return FEATURES[feature].fn(normalise(lm, side, rotate))


def finger_features(lm: np.ndarray, side=None) -> np.ndarray:
    """Per-finger geometry, for a person to read rather than for training.

    Per finger: three joint angles, an extension ratio (tip-to-knuckle distance
    over summed bone length, so 1.0 is straight and ~0.3 is a fist), and the
    tip's distance to the wrist and to the palm centre. Then the four gaps
    between adjacent fingertips, and the thumb-to-index pinch. 35 numbers.
    """
    p = normalise(lm, side)
    palm = p[:, [WRIST, 5, 9, 13, 17]].mean(1)
    out = []
    for idx in FINGERS.values():
        chain = (WRIST,) + idx
        for j in range(1, len(chain) - 1):
            v1 = _unit(p[:, chain[j - 1]] - p[:, chain[j]])
            v2 = _unit(p[:, chain[j + 1]] - p[:, chain[j]])
            out.append(np.arccos(np.clip((v1 * v2).sum(-1), -1.0, 1.0)))
        tip, mcp = p[:, idx[-1]], p[:, idx[0]]
        bone = sum(np.linalg.norm(p[:, chain[k + 1]] - p[:, chain[k]], axis=-1)
                   for k in range(len(chain) - 1))
        out.append(np.linalg.norm(tip - mcp, axis=-1) / (bone + 1e-8))
        out.append(np.linalg.norm(tip - p[:, WRIST], axis=-1))
        out.append(np.linalg.norm(tip - palm, axis=-1))
    tips = [p[:, f[-1]] for f in FINGERS.values()]
    for a, b in zip(tips, tips[1:]):
        out.append(np.linalg.norm(a - b, axis=-1))
    out.append(np.linalg.norm(tips[0] - tips[1], axis=-1))
    return np.stack(out, -1)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def describe(lm: np.ndarray) -> dict:
    """One hand's `finger_features` as a readable dict."""
    f = finger_features(lm.reshape(1, 21, 3))[0]
    names = list(FINGERS)
    return {n: {"extension": round(float(f[i * 6 + 3]), 3),
                "curl_radians": [round(float(f[i * 6 + j]), 3) for j in range(3)]}
            for i, n in enumerate(names)}


def extract(path: str | Path) -> tuple[np.ndarray, str] | None:
    """Landmarks and handedness for one image on disk."""
    with Image.open(path) as im:
        return hands.detect_hand(np.asarray(im.convert("RGB")))


def cache_manifest(rows: list[dict], out: Path, workers: int = 6) -> dict:
    """Run the detector over a manifest once and store the landmarks as an npz."""
    from concurrent.futures import ProcessPoolExecutor

    hands.model_path()
    paths = [r["path"] for r in rows]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        lms = list(ex.map(extract, paths, chunksize=32))

    found = np.array([r is not None for r in lms])
    stack = np.stack([r[0] if r is not None else np.zeros((21, 3), np.float32)
                      for r in lms])
    side = np.array([r[1] if r is not None else "Right" for r in lms])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, lm=stack, found=found, side=side,
        path=np.array([sources.repo_relative(r["path"]) for r in rows]),
        label=np.array([r["label"] for r in rows]),
        source=np.array([r["source"] for r in rows]),
        group=np.array([r["group"] for r in rows]),
        split=np.array([r.get("split", "") for r in rows]))

    report = {}
    for src in sorted(set(r["source"] for r in rows)):
        m = np.array([r["source"] == src for r in rows])
        report[src] = {"images": int(m.sum()), "landmarks": int(found[m].sum()),
                       "rate": round(float(found[m].mean()), 4)}
        print(f"[landmarks] {src:22s} {report[src]['landmarks']}/{report[src]['images']} "
              f"({report[src]['rate']:.1%})")
    print(f"[landmarks] -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return report


def load_cache(*paths: Path) -> dict:
    """Load one or more landmark caches and concatenate them.

    Keys missing from any cache are dropped. Every cache needs `side`.
    """
    ds = [np.load(p, allow_pickle=False) for p in paths]
    if old := [str(p) for p, d in zip(paths, ds) if "side" not in d.files]:
        raise ValueError(f"{old} have no handedness; re-run `slr.landmarks cache` "
                         "on them so left hands still fold")
    keys = set(ds[0].files).intersection(*(set(d.files) for d in ds))
    dropped = sorted(set().union(*(set(d.files) for d in ds)) - keys)
    if dropped:
        print(f"[landmarks] dropped keys not in every cache: {dropped}")
    out = {k: np.concatenate([d[k] for d in ds]) for k in keys}
    if "path" in out:
        out["path"] = np.array([sources.resolve(p) for p in out["path"]])
    return out


def build(n_classes: int, in_dim: int = FEATURE_DIM, hidden: int = 256,
          dropout: float = 0.3) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )


def _rotations(n: int, yaw: float, pitch: float, g: torch.Generator) -> torch.Tensor:
    a = (torch.rand(n, generator=g) * 2 - 1) * yaw
    b = (torch.rand(n, generator=g) * 2 - 1) * pitch
    ca, sa, cb, sb = a.cos(), a.sin(), b.cos(), b.sin()
    o, z = torch.ones(n), torch.zeros(n)
    ry = torch.stack([ca, z, sa, z, o, z, -sa, z, ca], 1).view(n, 3, 3)
    rx = torch.stack([o, z, z, z, cb, -sb, z, sb, cb], 1).view(n, 3, 3)
    return rx @ ry


def augment(p: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Jitter a batch of normalised hands: a camera a little off-axis, a little noise."""
    n = len(p)
    p = p @ _rotations(n, np.radians(25), np.radians(25), g).transpose(1, 2)
    p = p * (1 + (torch.rand(n, 1, 3, generator=g) * 2 - 1) * 0.05)
    p = p + torch.randn(p.shape, generator=g) * 0.02
    return p - p[:, WRIST:WRIST + 1]


def fit(rows: np.ndarray, y: np.ndarray, n_classes: int,
        featurize: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        weights: np.ndarray | None = None, augment=None, epochs: int = 60,
        seed: int = 0, hidden: int = 256, batch: int = 512, lr: float = 3e-3
        ) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """Train the MLP. Returns (net, mean, std).

    Batches are sampled with replacement by `weights`. `augment` runs on each
    batch before `featurize`. The scaler is fitted on unaugmented rows.
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    X = featurize(rows)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    rows_t, yt = torch.as_tensor(rows, dtype=torch.float32), torch.as_tensor(y)
    w = torch.as_tensor(np.ones(len(rows)) if weights is None else weights,
                        dtype=torch.float64)

    net = build(n_classes, X.shape[1], hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * max(1, len(rows) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    net.train()
    for _ in range(steps):
        b = torch.multinomial(w, batch, replacement=True, generator=g)
        r = rows_t[b] if augment is None else augment(rows_t[b], g)
        x = torch.tensor((featurize(r.numpy()) - mu) / sd, dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        crit(net(x), yt[b]).backward()
        opt.step()
        sched.step()
    net.eval()
    return net, mu, sd


class Model(NamedTuple):
    """A loaded landmark classifier, with its scaler and fusion share beside it."""

    net: nn.Module
    mean: np.ndarray
    std: np.ndarray
    classes: list[str]
    landmark_weight: float
    feature: str


def model_from_blob(blob: dict, landmark_weight: float | None = None) -> Model:
    """Build a servable model from a bundle dict, embedded or standalone."""
    tag = blob.get("feature")
    if tag not in FEATURES:
        raise ValueError(f"bundle declares feature {tag!r}; this code only knows "
                         f"how to rebuild {sorted(FEATURES)}")
    dim = FEATURES[tag].dim
    mean, std = np.asarray(blob["mean"]), np.asarray(blob["std"])
    if mean.shape != (dim,) or std.shape != (dim,):
        raise ValueError(f"{tag} scaler must have {dim} values, "
                         f"got {mean.shape}/{std.shape}")
    net = build(len(blob["classes"]), dim, blob.get("hidden", 256))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    w = (landmark_weight if landmark_weight is not None
         else float(blob.get("landmark_weight", blob.get("fusion_weight", 1.0))))
    return Model(net, mean, std, list(blob["classes"]), w, tag)


def load_model(path: Path) -> Model:
    """Read a bundle written by `run` back into something servable."""
    from . import safe_load

    return model_from_blob(safe_load(path))


@torch.no_grad()
def class_probs(model: Model, lm: np.ndarray, side=None) -> np.ndarray:
    """(N, len(model.classes)) softmax for detected hands, built with the model's own feature."""
    model.net.eval()
    x = (features(lm, side, model.feature) - model.mean) / model.std
    return torch.softmax(model.net(torch.tensor(x, dtype=torch.float32)), 1).numpy()


def predict(model: Model, lm: np.ndarray, found: np.ndarray, side=None) -> np.ndarray:
    """Landmarks in, class names out. No landmarks means `nothing`."""
    idx = class_probs(model, lm, side).argmax(1)
    return np.where(found, np.asarray(model.classes, dtype=object)[idx], "nothing")


def probs(model: Model, arr: np.ndarray, out_classes: list[str]
          ) -> tuple[np.ndarray, bool, str | None]:
    """Probabilities over `out_classes`, whether a hand was found, and which hand.

    Unknown classes get zero. No hand means all probability on `nothing`.
    """
    out = np.zeros(len(out_classes))
    got = hands.detect_hand(arr)
    if got is None:
        out[out_classes.index("nothing")] = 1.0
        return out, False, None
    lm, side = got
    p = class_probs(model, lm, side)[0]
    where = {c: i for i, c in enumerate(model.classes)}
    for j, name in enumerate(out_classes):
        if name in where:
            out[j] = p[where[name]]
    return out, True, side


def fuse(pixel: np.ndarray, landmark: np.ndarray,
         landmark_weight: float) -> np.ndarray:
    """Weighted average of pixel and landmark probabilities."""
    return (1.0 - landmark_weight) * pixel + landmark_weight * landmark


@torch.no_grad()
def predict_one(model: Model, arr: np.ndarray, k: int = 3) -> dict:
    """One photo (H, W, 3) -> label, confidence, top-k. Confidence is uncalibrated."""
    p, found, side = probs(model, arr, model.classes)
    if not found:
        return {"label": "nothing", "confidence": None, "hand_detected": False,
                "top_k": [{"label": "nothing", "p": None}]}
    order = np.argsort(-p)[:k]
    return {"label": model.classes[int(order[0])],
            "confidence": round(float(p[order[0]]), 4),
            "hand_detected": True, "hand": side,
            "top_k": [{"label": model.classes[int(i)], "p": round(float(p[i]), 4)}
                      for i in order]}


def detector_decomposition(pred: np.ndarray, truth: np.ndarray,
                           found: np.ndarray) -> dict:
    """Separate detector misses from classifier mistakes."""
    found = np.asarray(found).astype(bool)
    real = truth != "nothing"

    def hit(m):
        return float((pred[m] == truth[m]).mean()) if m.any() else None

    n_missed_real = int((real & ~found).sum())
    return {
        "n": int(len(truth)),
        "acc": hit(np.ones(len(truth), bool)),
        "acc_excluding_nothing": hit(real),
        "acc_given_detection": hit(real & found),
        "detection_rate": float(found.mean()),
        "detection_rate_excluding_nothing": float(found[real].mean()) if real.any() else None,
        "n_nothing": int((~real).sum()),
        "n_nothing_answered_by_rule": int((~real & ~found).sum()),
        "n_real_signs_forced_to_nothing": n_missed_real,
        "miss_policy": "no hand detected -> P(nothing)=1.0; see `probs`",
    }


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Accuracy and macro-F1 over whichever classes actually appear in `truth`."""
    labels = sorted(set(truth.tolist()))
    f1 = {}
    for c in labels:
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        f1[c] = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"n": int(len(truth)), "acc": float((pred == truth).mean()),
            "macro_f1": float(np.mean(list(f1.values()))), "per_class_f1": f1}


def by_side(pred: np.ndarray, truth: np.ndarray, side: np.ndarray) -> dict:
    """The same score split by which hand it was."""
    out = {}
    for s in ("Right", "Left"):
        m = side == s
        if m.any():
            r = score(pred[m], truth[m])
            out[s] = {"n": r["n"], "acc": round(r["acc"], 4),
                      "macro_f1": round(r["macro_f1"], 4)}
    return out


def _side_line(scores: dict) -> str:
    return "  [" + " ".join(f"{s} {v['acc']:.4f} (n={v['n']})"
                            for s, v in scores.items()) + "]"


def run(caches: list[Path], out: Path | None = None, seed: int = 0,
        train_sources: list[str] | None = None, feature: str = V2) -> dict:
    """Train on the training corpora and test on the holdout corpus.

    Validation is split off by group. Each training corpus is sampled equally,
    whatever its size.
    """
    from . import data

    d = load_cache(*caches)
    src, found, label, side = d["source"], d["found"], d["label"], d["side"]
    train_src = list(train_sources or TRAIN_SOURCES)
    test_src = [n for n, s in sources.REGISTRY.items() if s.role == "holdout"]
    if bad := sorted(n for n in train_src
                     if sources.REGISTRY[n].role not in ("train", "either")):
        raise ValueError(f"{bad} are not training sources")
    missing = sorted(set(train_src) - set(src.tolist()))
    is_test = np.isin(src, test_src)
    if missing or not is_test.any():
        raise ValueError(
            f"need rows from every train source {train_src} and a holdout source "
            f"{test_src}; these caches have {sorted(set(src.tolist()))}")
    is_train = np.isin(src, train_src)

    tr_pool = np.where(is_train & found)[0]
    classes = sorted(set(label[tr_pool].tolist()))

    rows = [{"label": label[i], "group": d["group"][i], "split": ""} for i in tr_pool]
    data.split_by_group(rows, val=0.15, test=0.0, seed=seed)
    sp = np.array([r["split"] for r in rows])
    tr, va = tr_pool[sp == "train"], tr_pool[sp == "val"]

    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx.get(lab, -1) for lab in label])
    P = normalise(d["lm"], side)

    def train(rows: np.ndarray) -> Model:
        counts = {s: int((src[rows] == s).sum()) for s in train_src}
        w = np.array([1.0 / counts[s] for s in src[rows]])
        net, mu, sd = fit(P[rows], y[rows], len(classes), FEATURES[feature].fn,
                          weights=w, augment=augment, seed=seed)
        return Model(net, mu, sd, classes, 1.0, feature)

    model = train(tr)
    pv = predict(model, d["lm"][va], found[va], side[va])
    val = score(pv, label[va])
    val["by_side"] = by_side(pv, label[va], side[va])
    print(f"[landmarks] val   n={val['n']:5d}  acc={val['acc']:.4f}  "
          f"macro_f1={val['macro_f1']:.4f}" + _side_line(val["by_side"]))

    both = np.concatenate([tr, va])
    model = train(both)
    te = np.where(is_test)[0]
    pt = predict(model, d["lm"][te], found[te], side[te])
    test = score(pt, label[te])
    test["by_side"] = by_side(pt, label[te], side[te])
    test["detector"] = detector_decomposition(pt, label[te], found[te])
    val["detector"] = detector_decomposition(pv, label[va], found[va])
    print(f"[landmarks] TEST  n={test['n']:5d}  acc={test['acc']:.4f}  "
          f"macro_f1={test['macro_f1']:.4f}" + _side_line(test["by_side"]))
    dec = test["detector"]
    print(f"[landmarks] TEST  hand found {dec['detection_rate']:.4f}, "
          f"acc when found {dec['acc_given_detection']:.4f}, "
          f"{dec['n_real_signs_forced_to_nothing']} signs missed by the detector")
    worst = sorted(test["per_class_f1"].items(), key=lambda kv: kv[1])[:5]
    print("[landmarks] weakest: " + "  ".join(f"{c}={v:.2f}" for c, v in worst))

    report = {"classes": classes, "feature": feature,
              "n_train": len(tr), "n_val": len(va),
              "train_sources": sorted(set(src[tr].tolist())),
              "final_fit_rows_by_source": {s: int((src[both] == s).sum())
                                           for s in train_src},
              "val": val, "test": test,
              "landmark_rate": {"train": float(found[is_train].mean()),
                                "test": float(found[is_test].mean())},
              "left_hand_share": {
                  "train": float((side[is_train] == "Left").mean()),
                  "test": float((side[is_test] == "Left").mean())}}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.net.state_dict(), "classes": classes,
                    "mean": model.mean, "std": model.std, "hidden": 256,
                    "feature": feature, "report": report},
                   out)
        (out.parent / "landmark_eval.json").write_text(json.dumps(report, indent=2))
        print(f"[landmarks] -> {out}")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="classify signs from hand landmarks")
    ap.add_argument("cmd", choices=["cache", "train"])
    ap.add_argument("path", type=Path, nargs="+",
                    help="cache: a manifest. train: one or more npz caches")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", nargs="*", default=None,
                    help=f"training corpora (default {' '.join(TRAIN_SOURCES)})")
    a = ap.parse_args(argv)
    a.path = a.path if a.cmd == "train" else a.path[0]

    if a.cmd == "cache":
        if not hands.available():
            raise SystemExit("landmark extraction needs mediapipe and opencv; run `uv sync`")
        rows = sources.read_manifest(a.path)
        cache_manifest(rows, a.out or (CACHE / "all.npz"), a.workers)
        return 0

    run(a.path, a.out or Path("runs") / "landmark.pt", a.seed, a.sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
