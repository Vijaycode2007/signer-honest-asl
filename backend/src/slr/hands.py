"""Hand localisation: find the hand, crop to it, throw the room away.

The corpora this project trains on are continuous webcam captures, and the
failure mode that `slr.leakage` measures - a classifier that scores well by
memorising a wallpaper - is a failure of *framing*, not of architecture. The
augmentation in `slr.data` attacks the background indirectly, by making it
unreliable. Cropping attacks it directly, by deleting it.

MediaPipe's palm detector plus its 21-landmark model gives a box; OpenCV does
the geometry. Both are used only here, and both are optional: nothing else in
the package imports this module, so a machine without them can still train,
evaluate and serve on uncropped images.

Detection is not free and it is not perfect, which is why this module reports a
rate instead of assuming one. Over every image in both corpora, not a sample:

    corpus              images   hands found
    asl_alphabet        87,000   71,717   82.4%
    asl_alphabet_test    1,740    1,658   95.3%

Those come from a full `crop` run and are written to `dataset/crops/detection.json`
as it goes; `probe` gives the same figure off a sample without writing anything.

`detect` looks twice. On a miss it pads the image with a replicated border,
upscales it, and looks again, which on a 300-image sample of asl_alphabet took
74.7% to 81.3% and did nothing at all for asl_alphabet_test. That is the shape
you would expect - MediaPipe's palm detector wants a hand *within* a scene, and
a 200x200 frame that is 90% hand does not look like one. Padding hands it the
scene it is looking for; a photo that already has one gains nothing.

Of the 15,283 images asl_alphabet still misses, 3,000 are the `nothing` class,
which detects at 0% and correctly so: those frames are an empty room with no
hand in them. Across the classes that do hold a hand the miss rate is 14.6%,
concentrated in the fist-like shapes - M, N, A and S sit at the bottom of a
per-class sample, where the fingers are folded in and there is least for a palm
detector to find.

Contrast normalisation (CLAHE on the L channel) was tried here and *lowered*
detection to 64% on asl_alphabet. It is not in the pipeline, and this paragraph
exists so nobody adds it back on the assumption that it must help.

Images where no hand is found are not dropped. They pass through untouched,
because a fifth of the training set is a worse thing to throw away than to keep
as an uncropped minority - and for `nothing` the uncropped frame is the correct
input anyway. The serve path in `slr.api` applies this same fallback, so what
the model sees on a stranger's phone is what it was trained on.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from . import sources

# float16 hand landmarker, the current published bundle. Cached beside the data,
# never committed - it is 7.8 MB of weights.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = sources.DEFAULT_ROOT.parent / "models" / "hand_landmarker.task"

# Where `crop_manifest` mirrors the corpora. `slr.train` looks for this prefix to
# decide whether to stamp a checkpoint as crop-trained, so it is load-bearing.
CROP_ROOT = sources.DEFAULT_ROOT.parent / "crops"

# Fraction of the landmark box added on every side. Landmarks stop at the
# fingertips and the wrist; a bare box clips the nail edge of a pointed finger
# and, on letters like R and U, that edge is the whole distinction.
# ponytail: 0.3 is reasoned, not tuned - the number to tune it against is the
# cross-corpus accuracy of the same architecture trained on cropped vs uncropped
# manifests, and that run has not happened yet. At 0.3 the box keeps a median
# 50% of asl_alphabet's pixels and 56% of asl_alphabet_test's, and clamps to the
# whole frame on 18-20% of images where the hand already fills it.
MARGIN = 0.3
MIN_CONFIDENCE = 0.2       # low on purpose: a wrong box is repaired by MARGIN,
                           # a missed hand is not repaired by anything.

_DETECTOR = None           # created lazily, per process - see `_worker`.


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
        import cv2          # noqa: F401
        import mediapipe    # noqa: F401
        return True
    except ImportError:
        return False


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

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
    """Raw MediaPipe output for one image: (landmark list, "Left"|"Right") or None.

    Handedness comes back because a mirrored hand is a different hand as far as
    coordinate geometry is concerned, and `slr.landmarks` needs to know in order
    to fold both onto one chirality. It is MediaPipe's own call rather than
    something derived from the points, because the 2D winding of the landmarks
    flips for a right hand seen from the back just as it does for a left hand
    seen from the palm, and those two are not the same thing.
    """
    import mediapipe as mp

    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))
    res = _detector().detect(img)
    if not res.hand_landmarks:
        return None
    side = res.handedness[0][0].category_name if res.handedness else "Right"
    return res.hand_landmarks[0], side


def detect_hand(arr: np.ndarray, retry: bool = True
                ) -> tuple[np.ndarray, str] | None:
    """The 21 landmarks as (21, 3) **pixel** coordinates, plus which hand.

    A plain tuple rather than an array carrying the side as an attribute. The
    obvious-looking ndarray subclass silently loses that attribute through both
    a pickle and a slice, and this value's whole job is to survive a trip
    through a process pool into a cache - where losing it would mark every hand
    right-handed and quietly undo the fold in `slr.landmarks`.

    Two passes. The second pads and upscales, which is what lifts asl_alphabet
    from 74.7% to 81.3% on a sample; its landmarks are mapped back onto the
    original image's coordinates so a caller cannot tell which pass found them.

    Pixels, not MediaPipe's 0-1 normalised coordinates, because normalised x and
    y are each divided by a different number on a non-square photo. A 640x480
    frame would come back with the hand squashed horizontally, and every shape
    feature computed from it would be wrong in a way that depends on the phone.
    z keeps MediaPipe's convention of being scaled like x.
    """
    import cv2

    h, w = arr.shape[:2]
    got = _landmarks(arr)
    if got is not None:
        lm, side = got
        return np.array([[p.x * w, p.y * h, p.z * w] for p in lm],
                        dtype=np.float32), side

    if not retry:
        # The live overlay passes retry=False: a second pass doubles the cost of
        # exactly the frames that have no hand in them, which is the case where
        # the preview most needs to stay responsive.
        return None

    pad = int(max(h, w) * MARGIN)
    big = cv2.resize(cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_REPLICATE),
                     (384, 384), interpolation=cv2.INTER_CUBIC)
    got = _landmarks(big)
    if got is None:
        return None
    lm, side = got
    # Undo the letterbox: 384px spans (w + 2*pad) original pixels, offset by pad.
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
    """Landmark cloud -> a margined square box that stays inside the image."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    # Square before margining, so a flat handshape (B) and a tall one (D) get the
    # same amount of context rather than the same amount of distortion.
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * MARGIN)
    n = min(round(side), w, h)                  # cannot be bigger than the frame
    # Round the side once and derive the far edge from it, rather than rounding
    # both edges: rounding both gives a 111x110 "square", and every such image
    # is one more pixel of aspect distortion the model has to absorb.
    x0 = min(max(round(cx - n / 2), 0), w - n)  # slide inside, never shrink
    y0 = min(max(round(cy - n / 2), 0), h - n)
    return (x0, y0, x0 + n, y0 + n)


def crop(img: Image.Image) -> tuple[Image.Image, bool]:
    """Crop a PIL image to its hand. Returns (image, hand_found).

    On a miss the image is returned untouched and the caller is told so. This is
    the one function both the offline pass and the live API call, so the two can
    never drift apart.
    """
    box = detect(np.asarray(img.convert("RGB")))
    return (img.crop(box), True) if box else (img, False)


# --- offline pass ------------------------------------------------------------

def _out_path(row: dict, out_root: Path) -> Path:
    p = Path(row["path"])
    try:
        rel = p.relative_to(sources.DEFAULT_ROOT)
    except ValueError:
        rel = Path(row["source"]) / p.name
    return out_root / rel.with_suffix(".jpg")


def _worker(task: tuple[dict, str]) -> tuple[str, bool | None]:
    """Crop one image. Runs in a pool process, which is why `_detector` is lazy:
    a landmarker built in the parent does not survive a fork.

    Returns None for `found` when the crop was already on disk - the image is
    not re-examined, so counting it either way would be a guess.
    """
    row, dest = task
    out = Path(dest)
    if out.exists():
        return dest, None                    # resumable; see `crop_manifest`
    with Image.open(row["path"]) as im:
        img = im.convert("RGB")
        cropped, found = crop(img)
        out.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out, quality=95)
    return dest, found


def crop_manifest(rows: list[dict], out_root: Path = CROP_ROOT, workers: int = 4
                  ) -> tuple[list[dict], dict]:
    """Write a cropped mirror of every row and return rows pointing at it.

    Idempotent: an image already written is left alone, so a pass interrupted
    after 60k of 88k images resumes rather than restarting. Skipped images are
    left out of the detection rate rather than assumed to have hit, so a resumed
    run reports the rate over what it actually looked at and says how many it
    skipped. `probe` measures without writing anything.
    """
    from concurrent.futures import ProcessPoolExecutor

    model_path()                             # download once, in the parent
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
        print(f"[hands] {skipped} images were already cropped and are excluded "
              f"from that rate")
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
    # Moving this off CROP_ROOT means `slr.train` will not recognise the result
    # as cropped and will stamp the checkpoint as whole-photo. Use it to compare
    # two margins side by side, not to relocate the real run.
    ap.add_argument("--out-root", type=Path, default=CROP_ROOT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=200, help="probe: images per source")
    a = ap.parse_args(argv)

    if not available():
        raise SystemExit(
            "hand detection needs mediapipe and opencv:\n"
            "    pip install 'slr[vision]'\n"
            "or  pip install 'mediapipe>=0.10.14,<1' opencv-contrib-python"
        )

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
