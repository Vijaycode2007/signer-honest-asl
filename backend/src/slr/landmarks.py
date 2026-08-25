"""Classifying the hand from its 21 landmarks instead of from its pixels.

`slr.hands` uses MediaPipe to find the hand and then throws the landmarks away,
keeping only a box. This module keeps the landmarks and never looks at the
photo again.

That sounds like less information, and it is. It is also, on the only protocol
this project treats as honest, worth thirty accuracy points:

    protocol C, cross-corpus, all 1,740 test images, 29 classes
      ViT-S on pixels             60.6% acc    0.568 macro-F1
      21 landmarks -> 2-layer MLP 90.6% acc    0.902 macro-F1

The reason is the same reason the rest of this repository exists. A pixel model
trained on `asl_alphabet` can reach the answer through the wallpaper, the
signer's skin, the white balance of one webcam - cues worth a great deal in
training and nothing at all on a stranger's phone. Landmarks carry none of them.
There is no room in 21 coordinate triples to encode a room, so the model cannot
learn to and does not have to unlearn it. What survives the corpus boundary is
the only thing the features ever held: the shape of the hand.

This is not a novel idea and the project's own bibliography says so - FSboard
(arXiv:2407.15806) reports 11.1% CER on signer-disjoint phone fingerspelling
from MediaPipe landmarks. It is the honest-transfer result in this field, and
until now this repository was not using it.

The whole thing trains on a laptop CPU in under a minute, because detection runs
once offline (`cache_manifest`) and what is left is a 63-input MLP.

## What the features are, and an unresolved disagreement

Wrist-centred, palm-scaled 3D coordinates: 21 points, 63 numbers.

Explicit per-finger geometry - joint angles at every knuckle, fingertip-to-palm
distances, the spread between adjacent tips - is in `finger_features`, because
it is the obvious thing to reach for when the goal is "track each finger". Which
of the two to feed the classifier is **not settled**, and this is the honest
state of it:

    variant                     val macro-F1   test macro-F1
    coords              (63)        0.9635         0.8977
    coords, rotated     (63)        0.9524         0.8057
    per-finger only     (35)        0.8852         0.8362
    coords + per-finger (98)        0.9543         0.9263

Validation prefers the coordinates alone. The cross-corpus test prefers the
concatenation, by nearly three F1 points. Selection in this repository happens
on validation and never on test, so `features` returns the coordinates and that
is what ships - but "per-finger geometry adds nothing" is a claim the test
column does not support, and it should not be repeated as if it did.

The disagreement most likely comes from the validation split being weak rather
than from the features. `leakage.recover_groups` collapses 85,664 of
asl_alphabet's 87,000 images into a single near-duplicate component, so a
group-disjoint validation split is whatever failed to merge into that blob: 795
images that are unrepresentative by construction. Fixing that split is worth
more than guessing between these two rows.

Rotation normalisation does lose, and by enough to be sure of: nine F1 points on
test. ASL orientation is meaningful - G and H are D and U pointed sideways - and
a canonical rotation is exactly the information that separates them.

On the one corpus with ground-truth signer identity, all four variants are
indistinguishable: trained on 151 `sl_digits` signers and tested on 66 held-out
ones, every row above scores 98.7-99.2%. That is a real signer-independence
check and the features pass it; it just cannot referee this particular choice,
because ten well-lit digits are too easy to separate the candidates.

## What it cannot do

`nothing` is an empty room, so it has almost no landmarks to classify from, and
the detector's silence is the answer instead. On the test corpus all 60
`nothing` images produced no landmarks, so the fallback in `predict` answered
every one of them and was right 60 times out of 60.

"Almost" rather than "no": on the training corpus the detector false-fires on 10
of 3,000 empty frames, a 0.33% rate, so `nothing` does end up in the
classifier's label space with ten examples. They are left there rather than
filtered out, because a spurious hand should map to `nothing` too, and ten rows
teaching that cost nothing.

The cost is the other 22 images out of 1,740 where a hand was there and the
detector missed it: 14 `space`, 6 `I`, 2 `K`, all scored as errors. That is
already inside the 90.6% above.

Where it is weakest is where a palm detector would be expected to be weakest.
V (0.57 F1), N (0.58), M (0.75) and K (0.76) turn on where the thumb is, and the
thumb is the part folded under the fingers where the camera cannot see it and
MediaPipe is inferring rather than measuring. A pixel model can at least look at
the shading there, which is the argument for fusing the two. That is not built
here.

## A coupling to know about

`predict` calls `features` with the shipped settings. Anything comparing feature
variants must run its own forward pass rather than go through it, or it will
train on one representation and score on another - which is how the rotation row
above first came out at chance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from . import hands, sources

# MediaPipe's landmark order, which is the only reason these indices are magic.
WRIST = 0
FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
           "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
MIDDLE_MCP = 9          # the scale reference: wrist -> here is the palm's length

CACHE = sources.DEFAULT_ROOT.parent / "landmarks"
FEATURE_DIM = 3 * 21


# --- features ----------------------------------------------------------------

def canonical_side(lm: np.ndarray, side=None) -> np.ndarray:
    """Which rows are left hands. Reads `Hand.side` when the caller passes none.

    Unknown means right, so a plain array behaves exactly as it did before this
    existed and the older caches stay readable.
    """
    n = 1 if lm.ndim == 2 else len(lm)
    if side is None:
        side = "Right"
    if isinstance(side, str):
        return np.full(n, side == "Left")
    return np.asarray(side) == "Left"


def normalise(lm: np.ndarray, side=None, rotate: bool = False) -> np.ndarray:
    """(N, 21, 3) pixel landmarks -> wrist at the origin, palm length 1, right-handed.

    Left hands are mirrored in x onto the right-handed frame. This is the single
    most important line in the module for anybody who is not the one signer the
    training corpus was shot from.

    Chirality is not a nuisance variable that averages out: it is a rigid
    reflection of every coordinate, so a left-handed B lands nowhere near a
    right-handed B in feature space. Measured on twelve photos of a left-handed
    signer, folding them onto the right-handed frame took the classifier from
    2/12 to 9/12. asl_alphabet is 91% right hands and asl_alphabet_test 98%, so
    nothing in the benchmark ever exposed this - the corpora agree with each
    other and disagree with half the population.

    Folding is deliberately not the same as flip augmentation, which `slr.data`
    refuses for pixels on the grounds that mirroring an image can change what a
    sign means. Nothing here is being taught that a mirrored sign is the same
    sign; the reflection is undone before the features are built, so the model
    only ever sees one chirality and never has to have an opinion about the
    other.

    Translation and scale are camera facts: how far the signer stood, where the
    hand happened to sit in frame. Dividing by the wrist-to-middle-knuckle
    distance rather than by the bounding box means the scale reference is a bone
    rather than a silhouette, so a spread hand and a fist normalise the same.

    `rotate` additionally spins the hand so the palm points up, removing
    in-plane orientation. It is off by default because it measurably loses: ASL
    orientation carries meaning, and G and H are the same handshape as D and U
    pointed sideways. The switch exists so that claim stays reproducible.
    """
    left = canonical_side(lm, side)
    if lm.ndim == 2:
        lm = lm[None]
    p = np.asarray(lm) - np.asarray(lm)[:, WRIST:WRIST + 1]
    p[left, :, 0] *= -1                      # fold left hands onto right
    palm = np.linalg.norm(p[:, MIDDLE_MCP], axis=-1)[:, None, None]
    p = p / (palm + 1e-8)
    if not rotate:
        return p
    a = p[:, MIDDLE_MCP, :2]
    # cos = uy, sin = ux sends the unit palm axis (ux, uy) exactly onto +y:
    #   x' = x*uy - y*ux -> 0,   y' = x*ux + y*uy -> 1
    c, sn = a[:, 1:2], a[:, 0:1]
    n = np.sqrt(c ** 2 + sn ** 2) + 1e-8
    c, sn = (c / n)[:, None, 0], (sn / n)[:, None, 0]
    x, y = p[..., 0].copy(), p[..., 1].copy()
    p[..., 0] = c * x - sn * y
    p[..., 1] = sn * x + c * y
    return p


def features(lm: np.ndarray, side=None, rotate: bool = False) -> np.ndarray:
    """The vector the classifier actually sees. (N, 63)."""
    n = normalise(lm, side, rotate)
    return n.reshape(len(n), -1)


def finger_features(lm: np.ndarray, side=None) -> np.ndarray:
    """Per-finger geometry, for reading rather than for training.

    Measured against `features` and beaten by it (see the module docstring), so
    nothing is fed this. It is here because "the index is extended and the thumb
    is folded across the palm" is a sentence a person can check against a photo,
    and 63 normalised coordinates is not.

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
    """One hand's `finger_features` as a readable dict. For debugging a call."""
    f = finger_features(lm.reshape(1, 21, 3))[0]
    names = list(FINGERS)
    return {n: {"extension": round(float(f[i * 6 + 3]), 3),
                "curl_radians": [round(float(f[i * 6 + j]), 3) for j in range(3)]}
            for i, n in enumerate(names)}


# --- the offline pass --------------------------------------------------------

def extract(path: str | Path) -> tuple[np.ndarray, str] | None:
    """Landmarks and handedness for one image on disk."""
    with Image.open(path) as im:
        return hands.detect_hand(np.asarray(im.convert("RGB")))


def _worker(path: str) -> tuple[np.ndarray, str] | None:
    return extract(path)


def cache_manifest(rows: list[dict], out: Path, workers: int = 6) -> dict:
    """Run the detector over a manifest once and store the landmarks.

    Landmarks are ~250 bytes an image against ~10 KB for the crop, so the whole
    88k corpus fits in one 20 MB npz and training reads it in a second. This is
    the step that makes the rest of the module runnable on a laptop: detection
    is the only expensive part and it happens exactly once.
    """
    from concurrent.futures import ProcessPoolExecutor

    hands.model_path()
    paths = [r["path"] for r in rows]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        lms = list(ex.map(_worker, paths, chunksize=32))

    found = np.array([r is not None for r in lms])
    stack = np.stack([r[0] if r is not None else np.zeros((21, 3), np.float32)
                      for r in lms])
    side = np.array([r[1] if r is not None else "Right" for r in lms])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, lm=stack, found=found, side=side,
        # The path is what lets a cache be joined against another model's
        # predictions on the same images, which is the whole basis of a fair
        # fusion sweep: without it the only way to line two models up is to
        # trust that two manifests were walked in the same order.
        path=np.array([r["path"] for r in rows]),
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


def load_cache(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


# --- the model ---------------------------------------------------------------

def build(n_classes: int, in_dim: int = FEATURE_DIM, hidden: int = 256,
          dropout: float = 0.3) -> nn.Module:
    """Two hidden layers. Deliberately small.

    63 inputs is a tabular problem, not a perception problem - the perception
    already happened inside MediaPipe. Widening or deepening this past a couple
    of hundred units buys nothing and starts memorising signers instead.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )


def fit(X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int = 160,
        lr: float = 3e-3, batch: int = 256, seed: int = 0, hidden: int = 256
        ) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """Train the MLP. Returns (net, mean, std) - the scaler travels with it.

    Standardisation constants are fitted on the training rows only and returned
    alongside the weights, because a scaler fitted on the test corpus is
    leakage in the same way a random split is: quieter, and just as fatal.
    """
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    net = build(n_classes, X.shape[1], hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue          # BatchNorm needs two rows to have a variance
            opt.zero_grad(set_to_none=True)
            crit(net(xt[b]), yt[b]).backward()
            opt.step()
        sched.step()
    net.eval()
    return net, mu, sd


@torch.no_grad()
def predict(net: nn.Module, mu: np.ndarray, sd: np.ndarray, classes: list[str],
            lm: np.ndarray, found: np.ndarray, side=None) -> np.ndarray:
    """Landmarks in, class names out. No landmarks means `nothing`.

    That fallback is the whole handling of the 29th class. An empty room has no
    hand to find, the detector says so, and saying `nothing` is both the honest
    answer and, on the test corpus, the correct one 60 times out of 60.
    """
    net.eval()
    x = torch.tensor((features(lm, side) - mu) / sd, dtype=torch.float32)
    idx = net(x).argmax(1).numpy()
    return np.where(found, np.asarray(classes, dtype=object)[idx], "nothing")


class Model(NamedTuple):
    """A loaded landmark classifier, with everything serving needs beside it.

    The scaler and the fusion weight travel with the weights for the same reason
    the temperature does in `slr.api`: a constant fitted during an experiment and
    re-derived at serve time is a constant that can silently disagree.
    """

    net: nn.Module
    mean: np.ndarray
    std: np.ndarray
    classes: list[str]
    weight: float                # share of the fused answer this model gets


def load_model(path: Path) -> Model:
    """Read a bundle written by `run` back into something servable.

    The scaler travels inside the bundle for the same reason the temperature
    does in `slr.api`: constants fitted during training and re-derived at serve
    time are constants that can silently disagree.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("feature") != "normalised_landmarks_v1":
        raise ValueError(f"{path} carries features '{blob.get('feature')}', "
                         "which this code does not know how to build")
    net = build(len(blob["classes"]), FEATURE_DIM, blob.get("hidden", 256))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return Model(net, blob["mean"], blob["std"], blob["classes"],
                 float(blob.get("fusion_weight", 1.0)))


@torch.no_grad()
def probs(model: Model, arr: np.ndarray, out_classes: list[str]
          ) -> tuple[np.ndarray, bool, str | None]:
    """Probabilities over `out_classes`, plus whether a hand was found and which.

    Re-ordered into the caller's class list rather than this model's, so a fused
    answer is adding like for like. The classifier's label space is a subset -
    it never sees the classes that have no landmarks - and the missing slots stay
    at zero.

    No hand found returns all the mass on `nothing`. That is not a fallback bolted
    on beside the model, it is the model's answer: an empty frame has no hand to
    classify, and on the holdout corpus the detector's silence was right 60 times
    out of 60. At the fused weight it also dominates the blend, which is the
    intended behaviour - a photo of a wall should not be given a letter because
    the pixel model had an opinion about it.
    """
    out = np.zeros(len(out_classes))
    got = hands.detect_hand(arr)
    if got is None:
        out[out_classes.index("nothing")] = 1.0
        return out, False, None
    lm, side = got
    p = torch.softmax(model.net(torch.tensor(
        (features(lm, side) - model.mean) / model.std, dtype=torch.float32)), 1)[0].numpy()
    where = {c: i for i, c in enumerate(model.classes)}
    for j, name in enumerate(out_classes):
        if name in where:
            out[j] = p[where[name]]
    return out, True, side


def fuse(pixel: np.ndarray, landmark: np.ndarray, weight: float) -> np.ndarray:
    """Blend two probability vectors over the same classes.

    `weight` is the landmark model's share, chosen by least log-loss on a
    leakage-free half of the holdout corpus that neither model trained on and
    that is never reported. Selecting it on the in-corpus validation split gave
    w=0, which is the worst possible answer: within-corpus validation rewards
    whichever model best exploits the corpus, and that is precisely the bias this
    repository exists to measure rather than inherit.
    """
    return (1.0 - weight) * pixel + weight * landmark


@torch.no_grad()
def predict_one(model: Model, arr: np.ndarray, k: int = 3) -> dict:
    """One photo (H, W, 3) -> label, confidence, top-k. No hand means `nothing`.

    Softmax here is uncalibrated: unlike the pixel path there is no fitted
    temperature yet, so `confidence` orders candidates honestly but should not
    be read as a probability of being right.
    """
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


# --- driving it --------------------------------------------------------------

def run(cache: Path, out: Path | None = None, seed: int = 0) -> dict:
    """Protocol C on landmarks: train on one corpus, test on the other.

    Validation is carved group-disjointly out of the training corpus and is the
    only thing any choice here is allowed to see, exactly as in `slr.train`.
    """
    from . import data

    d = load_cache(cache)
    src, found, label = d["source"], d["found"], d["label"]
    # Caches written before chirality was recorded predate the fold; treating
    # them as all-right reproduces exactly the behaviour they were built with,
    # rather than silently folding on a value that was never measured.
    side = d.get("side")
    if side is None:
        print("[landmarks] WARNING: this cache has no handedness, so left hands "
              "cannot be folded. Re-run `slr.landmarks cache` to fix it.")
    train_src = [n for n, s in sources.REGISTRY.items() if s.role == "train"]
    test_src = [n for n, s in sources.REGISTRY.items() if s.role == "holdout"]
    is_train = np.isin(src, train_src)
    is_test = np.isin(src, test_src)
    if not is_train.any() or not is_test.any():
        raise ValueError(
            f"need rows from a train source {train_src} and a holdout source "
            f"{test_src}; this cache has {sorted(set(src.tolist()))}")

    # Only landmark-bearing rows can train. `nothing` frames almost never have
    # landmarks, so `predict`'s fallback is what really answers that class; the
    # handful the detector false-fires on stay in the label space on purpose, so
    # a spurious hand maps to `nothing` rather than to a letter.
    tr_pool = np.where(is_train & found)[0]
    classes = sorted(set(label[tr_pool].tolist()))

    rows = [{"label": label[i], "group": d["group"][i], "split": ""} for i in tr_pool]
    data.split_by_group(rows, val=0.15, test=0.0, seed=seed)
    sp = np.array([r["split"] for r in rows])
    tr, va = tr_pool[sp == "train"], tr_pool[sp == "val"]

    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[l] for l in label[tr_pool]])
    ytr, yva = y[sp == "train"], y[sp == "val"]

    X = features(d["lm"], side)
    net, mu, sd = fit(X[tr], ytr, len(classes), seed=seed)
    val = score(predict(net, mu, sd, classes, d["lm"][va], found[va],
                        None if side is None else side[va]), label[va])
    print(f"[landmarks] val   n={val['n']:5d}  acc={val['acc']:.4f}  "
          f"macro_f1={val['macro_f1']:.4f}")

    # Refit on train+val, then touch the holdout corpus exactly once.
    both = np.concatenate([tr, va])
    yboth = np.concatenate([ytr, yva])
    net, mu, sd = fit(X[both], yboth, len(classes), seed=seed)
    te = np.where(is_test)[0]
    test = score(predict(net, mu, sd, classes, d["lm"][te], found[te],
                         None if side is None else side[te]), label[te])
    print(f"[landmarks] TEST  n={test['n']:5d}  acc={test['acc']:.4f}  "
          f"macro_f1={test['macro_f1']:.4f}   (cross-corpus, all images, "
          f"{len(set(label[te].tolist()))} classes)")
    worst = sorted(test["per_class_f1"].items(), key=lambda kv: kv[1])[:5]
    print("[landmarks] weakest: " + "  ".join(f"{c}={v:.2f}" for c, v in worst))

    report = {"classes": classes, "n_train": len(tr), "n_val": len(va),
              "val": val, "test": test,
              "landmark_rate": {"train": float(found[is_train].mean()),
                                "test": float(found[is_test].mean())},
              "left_hand_share": None if side is None else {
                  "train": float((side[is_train] == "Left").mean()),
                  "test": float((side[is_test] == "Left").mean())}}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "classes": classes,
                    "mean": mu, "std": sd, "hidden": 256,
                    "feature": "normalised_landmarks_v1", "report": report},
                   out)
        (out.parent / "landmark_eval.json").write_text(json.dumps(report, indent=2))
        print(f"[landmarks] -> {out}")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="classify signs from hand landmarks")
    ap.add_argument("cmd", choices=["cache", "train"])
    ap.add_argument("path", type=Path, help="cache: a manifest. train: an npz cache")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "cache":
        if not hands.available():
            raise SystemExit(
                "landmark extraction needs mediapipe and opencv:\n"
                "    pip install 'mediapipe>=0.10.14,<1' opencv-contrib-python")
        rows = sources.read_manifest(a.path)
        cache_manifest(rows, a.out or (CACHE / "all.npz"), a.workers)
        return 0

    run(a.path, a.out or Path("runs") / "landmark.pt", a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
