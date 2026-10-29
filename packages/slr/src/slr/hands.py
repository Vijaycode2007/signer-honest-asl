from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from . import sources

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = sources.DEFAULT_ROOT.parent / "models" / "hand_landmarker.task"

CROP_ROOT = sources.DEFAULT_ROOT.parent / "crops"

MARGIN = 0.3
MIN_CONFIDENCE = 0.2

_DETECTOR = None


def model_path() -> Path:
    """Return the landmarker bundle, downloading it once if it is not here."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[hands] fetching landmarker -> {MODEL_PATH}")
        with urllib.request.urlopen(MODEL_URL, timeout=300) as r:
            MODEL_PATH.write_bytes(r.read())
    return MODEL_PATH


def available() -> bool:
    """True when this machine can actually detect hands."""
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
        return True
    except ImportError:
        return False


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        from mediapipe.tasks.python import BaseOptions, vision

        _DETECTOR = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path())),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=MIN_CONFIDENCE,
            )
        )
    return _DETECTOR


def _landmarks(arr: np.ndarray):
    import mediapipe as mp

    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))
    res = _detector().detect(img)
    if not res.hand_landmarks:
        return None
    side = res.handedness[0][0].category_name if res.handedness else "Right"
    return res.hand_landmarks[0], side


def detect_hand(arr: np.ndarray, retry: bool = True
                ) -> tuple[np.ndarray, str] | None:
    """The 21 landmarks as (21, 3) pixel coordinates, plus which hand.

    Pixels rather than MediaPipe's 0-1 coordinates, which squash the hand on a
    non-square photo. z is scaled like x.

    If nothing is found, retry once on a padded, upscaled copy, since MediaPipe
    struggles when the hand fills the frame. `retry=False` skips that.
    """
    import cv2

    h, w = arr.shape[:2]
    got = _landmarks(arr)
    if got is not None:
        lm, side = got
        return np.array([[p.x * w, p.y * h, p.z * w] for p in lm],
                        dtype=np.float32), side

    if not retry:
        return None

    pad = int(max(h, w) * MARGIN)
    big = cv2.resize(cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_REPLICATE),
                     (384, 384), interpolation=cv2.INTER_CUBIC)
    got = _landmarks(big)
    if got is None:
        return None
    lm, side = got
    sx, sy = (w + 2 * pad) / 384, (h + 2 * pad) / 384
    return np.array([[p.x * 384 * sx - pad, p.y * 384 * sy - pad, p.z * 384 * sx]
                     for p in lm], dtype=np.float32), side


def landmarks(arr: np.ndarray, retry: bool = True) -> np.ndarray | None:
    """Just the points, for callers that do not care whose hand it is."""
    got = detect_hand(arr, retry)
    return None if got is None else got[0]


def detect(arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Square pixel box (l, t, r, b) around the one hand, or None."""
    lm = landmarks(arr)
    if lm is None:
        return None
    h, w = arr.shape[:2]
    return _box([(x, y) for x, y, _ in lm], w, h)


def _box(pts: list[tuple[float, float]], w: int, h: int) -> tuple[int, int, int, int]:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * MARGIN)
    n = min(round(side), w, h)
    x0 = min(max(round(cx - n / 2), 0), w - n)
    y0 = min(max(round(cy - n / 2), 0), h - n)
    return (x0, y0, x0 + n, y0 + n)


def crop(img: Image.Image) -> tuple[Image.Image, bool]:
    """Crop a PIL image to its hand. Returns (image, hand_found)."""
    box = detect(np.asarray(img.convert("RGB")))
    return (img.crop(box), True) if box else (img, False)


def _out_path(row: dict, out_root: Path) -> Path:
    p = Path(row["path"])
    try:
        rel = p.relative_to(sources.DEFAULT_ROOT)
    except ValueError:
        rel = Path(row["source"]) / p.name
    return out_root / rel.with_suffix(".jpg")


def _worker(task: tuple[dict, str]) -> tuple[str, bool | None]:
    row, dest = task
    out = Path(dest)
    if out.exists():
        return dest, None
    with Image.open(row["path"]) as im:
        img = im.convert("RGB")
        cropped, found = crop(img)
        out.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out, quality=95)
    return dest, found


def crop_manifest(rows: list[dict], out_root: Path = CROP_ROOT, workers: int = 4
                  ) -> tuple[list[dict], dict]:
    """Write a cropped copy of every image and point the rows at it.

    Images already cropped are skipped and left out of the detection rate.
    """
    from concurrent.futures import ProcessPoolExecutor

    model_path()
    tasks = [(r, str(_out_path(r, out_root))) for r in rows]
    stats: dict[str, list[int]] = {}

    skipped = 0

    def note(row: dict, found: bool | None) -> None:
        nonlocal skipped
        if found is None:
            skipped += 1
            return
        s = stats.setdefault(row["source"], [0, 0])
        s[0] += found
        s[1] += 1

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for row, (dest, found) in zip(rows, ex.map(_worker, tasks, chunksize=32)):
                row["path"] = dest
                note(row, found)
    else:
        for row, task in zip(rows, tasks):
            dest, found = _worker(task)
            row["path"] = dest
            note(row, found)

    report = {src: {"images": n, "hand_found": k, "rate": round(k / n, 4)}
              for src, (k, n) in stats.items()}
    if skipped:
        report["_skipped"] = {"already_cropped": skipped}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "detection.json").write_text(json.dumps(report, indent=2))
    for src, (k, n) in stats.items():
        print(f"[hands] {src:22s} {k}/{n} hands found ({k / n:.1%})")
    if skipped:
        print(f"[hands] {skipped} images were already cropped")
    return rows, report


def probe(rows: list[dict], n: int = 200, seed: int = 0) -> dict:
    """Measure the detection rate on a sample without writing anything."""
    import random
    from collections import defaultdict

    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)

    report = {}
    for src, sub in sorted(by_src.items()):
        pick = random.Random(seed).sample(sub, min(n, len(sub)))
        k = 0
        for r in pick:
            with Image.open(r["path"]) as im:
                k += detect(np.asarray(im.convert("RGB"))) is not None
        report[src] = {"sampled": len(pick), "hand_found": k,
                       "rate": round(k / len(pick), 4)}
        print(f"[hands] {src:22s} {k}/{len(pick)} hands found ({k / len(pick):.1%})")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="crop corpora to the hand")
    ap.add_argument("cmd", choices=["crop", "probe"])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="cropped manifest csv")
    ap.add_argument("--out-root", type=Path, default=CROP_ROOT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=200, help="probe: images per source")
    a = ap.parse_args(argv)

    if not available():
        raise SystemExit("hand detection needs mediapipe and opencv; run `uv sync`")

    rows = sources.read_manifest(a.manifest)
    if a.cmd == "probe":
        probe(rows, a.n)
        return 0

    rows, _ = crop_manifest(rows, a.out_root, a.workers)
    out = a.out or a.manifest.with_name(a.manifest.stem + "_cropped.csv")
    sources.write_manifest(rows, out)
    print(f"[hands] train on this with:  python -m slr.train {out} --preset <p>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
