"""Dataset registry, downloaders, and manifest construction.

Every corpus we train or evaluate on is declared here exactly once. A source
carries three things the rest of the pipeline needs:

  * how to fetch it (kaggle / huggingface / plain zip over https),
  * how to read a class label out of a file path,
  * how to recover a *group key* - the unit that must never straddle a split.

The group key is the whole game. Sign-language corpora are shot as continuous
sessions: consecutive frames of one person holding one handshape differ by a
few pixels. A random train/test split scatters those frames across both sides
and the reported accuracy measures nothing but the model's ability to memorise
a background. Where the corpus hands us a signer id we use it. Where it does
not, `slr.leakage` recovers pseudo-groups by clustering near-duplicates, and
the group key falls back to that.
"""

from __future__ import annotations

import csv
import io
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# --- label space -------------------------------------------------------------

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
CLASSES = LETTERS + ["space", "del", "nothing"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Corpora spell the three non-letter classes every way imaginable.
_ALIASES = {
    "space": "space", "sp": "space", "blank": "space", "_": "space",
    "del": "del", "delete": "del", "dl": "del",
    "nothing": "nothing", "none": "nothing", "empty": "nothing", "bg": "nothing",
}


def normalise_label(raw: str) -> str | None:
    """Map a directory or column name onto the 29-class vocabulary, or None."""
    s = raw.strip()
    if len(s) == 1 and s.upper() in CLASS_TO_IDX:
        return s.upper()
    return _ALIASES.get(s.lower().replace(" ", "").replace("-", ""))


# --- group-key rules ---------------------------------------------------------

def group_by_parent_dir(path: Path, root: Path) -> str:
    """Fallback: the whole corpus is one group, so it can only ever be a train
    source or a whole-corpus holdout. Never split it internally."""
    return "all"


_DIGITS_RE = re.compile(r"IMG_(\d+)", re.I)


def group_sl_digits(path: Path, root: Path) -> str:
    """Recover signer identity for ardamavi/Sign-Language-Digits-Dataset.

    218 students were photographed back to back on one camera, ten shots each
    (digits 0..9 in order). The camera's own counter therefore encodes the
    signer: class 0 starts at IMG_1118, class 1 at IMG_1119, and every tenth
    frame after that is the next student. Verified against the published tree -
    each class's filenames form an arithmetic sequence of stride 10 offset by
    the class index.

    Nobody publishes this dataset with signer labels. This one line is what
    makes signer-disjoint evaluation possible on it.
    """
    m = _DIGITS_RE.search(path.name)
    if not m:
        return f"unknown:{path.name}"
    return f"signer{(int(m.group(1)) - 1118) // 10:04d}"


_INDEXED_RE = re.compile(r"^([A-Za-z_]+)(\d+)")


def group_by_frame_block(block: int) -> Callable[[Path, Path], str]:
    """For webcam-capture corpora named `<class><frame>.jpg` (grassknoted style).

    Consecutive frame indices are consecutive video frames, so bucketing by
    `frame // block` yields contiguous chunks that at least keep neighbouring
    frames together. Weaker than a true signer id - use only when
    `leakage.recover_groups` has not been run.
    """

    def _fn(path: Path, root: Path) -> str:
        m = _INDEXED_RE.match(path.stem)
        if not m:
            return f"unknown:{path.name}"
        return f"blk{int(m.group(2)) // block:05d}"

    return _fn


# --- source registry ---------------------------------------------------------

@dataclass(frozen=True)
class Source:
    name: str
    kind: str                      # "kaggle" | "hf" | "zip"
    ref: str                       # dataset slug / repo id / url
    role: str                      # "train" | "holdout" | "either" | "audit"
    group_fn: Callable[[Path, Path], str] = group_by_parent_dir
    label_depth: int = 1           # how many dirs up from the file the class name sits
    subdir: str = ""               # path inside the download to scan
    note: str = ""
    needs_group_recovery: bool = False
    verbatim_labels: bool = False  # take dir names as labels, outside the 29-class space


REGISTRY: dict[str, Source] = {s.name: s for s in [
    Source(
        name="asl_alphabet",
        kind="kaggle", ref="grassknoted/asl-alphabet", role="train",
        subdir="asl_alphabet_train/asl_alphabet_train",
        group_fn=group_by_frame_block(200),
        needs_group_recovery=True,
        note="87k imgs, 29 classes. Near-single-signer continuous webcam capture; "
             "its bundled 28-image 'test' set is unusable. Train source only.",
    ),
    Source(
        name="asl_alphabet_test",
        kind="kaggle", ref="danrasband/asl-alphabet-test", role="holdout",
        group_fn=group_by_parent_dir,
        note="870 imgs shot by a different person against varied backgrounds, "
             "explicitly built to validate models trained on asl_alphabet. "
             "This is the primary cross-corpus test set.",
    ),
    Source(
        name="asl_alphabet_v2",
        kind="kaggle", ref="debashishsau/aslamerican-sign-language-aplhabet-dataset",
        role="either", subdir="ASL_Alphabet_Dataset/asl_alphabet_train",
        group_fn=group_by_frame_block(200), needs_group_recovery=True,
        note="Second independent 29-class corpus. Used for train->test transfer "
             "in the other direction.",
    ),
    Source(
        name="asl_27class",
        kind="kaggle", ref="ardamavi/27-class-sign-language-dataset", role="either",
        needs_group_recovery=True,
        note="Mavi & Dikle 2022, arXiv:2203.03859. Collected from 173 volunteers - "
             "the only static ASL corpus with real signer diversity.",
    ),
    Source(
        name="sl_digits",
        kind="zip",
        ref="https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/refs/heads/master.zip",
        role="audit", subdir="Sign-Language-Digits-Dataset-master/Dataset",
        group_fn=group_sl_digits, verbatim_labels=True,
        note="218 signers, digits 0-9. Outside the 29-class label space, so it is "
             "never trained on. It is here because its true signer ids are "
             "recoverable from the camera's frame counter, which makes it the one "
             "corpus where `leakage.recover_groups` can be scored against ground "
             "truth. Free to download, no credentials needed.",
    ),
    Source(
        name="mine",
        kind="local", ref="photos taken in the app and labelled in the page",
        role="either",
        group_fn=lambda p, root: p.relative_to(root).parts[0],
        note="Your own hand, captured through app/web and filed as "
             "<session>/<LABEL>/. The session directory IS the group key: one "
             "sitting shares a light, a shirt and a camera distance, so its "
             "frames are near-copies and must never straddle a split. This is "
             "one signer, so anything trained on it is signer-dependent by "
             "construction - report it as adaptation, never as a headline. "
             "Never downloaded and never committed.",
    ),
    Source(
        name="slmnist",
        kind="hf", ref="Voxel51/American-Sign-Language-MNIST", role="either",
        needs_group_recovery=True,
        note="Sign Language MNIST, 27k/7k 28x28 grey. Its official test split is "
             "augmentation of the same base photographs as train - the canonical "
             "leaky benchmark. We keep it to demonstrate the audit.",
    ),
    Source(
        name="asl_alphabets_v03",
        kind="hf", ref="Marxulia/asl_sign_languages_alphabets_v03", role="either",
        needs_group_recovery=True,
        note="10.8k imgs, no auth required. Useful when Kaggle credentials are "
             "not available.",
    ),
]}

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "dataset" / "raw"


# --- downloading -------------------------------------------------------------

def _have_kaggle_creds() -> bool:
    return bool(os.environ.get("KAGGLE_USERNAME")) or (Path.home() / ".kaggle" / "kaggle.json").exists()


def download(name: str, root: Path = DEFAULT_ROOT) -> Path:
    """Fetch one source into `root/<name>`. Idempotent: skips a non-empty dir."""
    src = REGISTRY[name]
    dest = root / name
    if dest.exists() and any(dest.rglob("*")):
        print(f"[skip] {name} already at {dest}")
        return dest
    if src.kind == "local":
        print(f"[local] {name} is collected on this machine, not downloaded - "
              f"take photos in the app and label them; they land in {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    if src.kind == "kaggle":
        if not _have_kaggle_creds():
            raise SystemExit(
                f"{name} lives on Kaggle, which refuses unauthenticated downloads.\n"
                "Put your API token at ~/.kaggle/kaggle.json (chmod 600), or set "
                "KAGGLE_USERNAME / KAGGLE_KEY. Get it from kaggle.com -> Settings -> "
                "Create New Token."
            )
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", src.ref, "-p", str(dest), "--unzip"],
            check=True,
        )
    elif src.kind == "hf":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=src.ref, repo_type="dataset", local_dir=str(dest))
    elif src.kind == "zip":
        print(f"[get] {src.ref}")
        with urllib.request.urlopen(src.ref, timeout=300) as r:
            blob = r.read()
        zipfile.ZipFile(io.BytesIO(blob)).extractall(dest)
    else:
        raise ValueError(f"unknown source kind {src.kind}")

    print(f"[ok] {name} -> {dest}")
    return dest


# --- manifest ----------------------------------------------------------------

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Where `server.captures` parks a photo between being taken and being named.
# It is a waiting room, not a class, so it never reaches a manifest.
STAGING = "_inbox"
MANIFEST_COLS = ["path", "label", "label_idx", "source", "group", "split"]


def scan(name: str, root: Path = DEFAULT_ROOT) -> list[dict]:
    """Walk one downloaded source and emit manifest rows.

    Rows carry an empty `split`; assigning splits is `slr.data`'s job and is
    kept separate on purpose so a split can be recomputed without re-walking
    hundreds of thousands of files.
    """
    src = REGISTRY[name]
    base = root / name / src.subdir if src.subdir else root / name
    if not base.exists():
        raise FileNotFoundError(f"{base} missing - run `python -m slr.sources download {name}` first")

    rows: list[dict] = []
    skipped: set[str] = set()
    staged = 0
    for p in sorted(base.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        parts = p.relative_to(base).parts
        if len(parts) <= src.label_depth - 1:
            continue
        raw = parts[-(src.label_depth + 1)]
        if raw == STAGING:
            staged += 1                 # a capture nobody has named yet
            continue
        if src.verbatim_labels:
            label, idx = raw, -1        # sentinel: not trainable, audit only
        else:
            label = normalise_label(raw)
            if label is None:
                skipped.add(raw)
                continue
            idx = CLASS_TO_IDX[label]
        rows.append({
            "path": str(p.resolve()),
            "label": label,
            "label_idx": idx,
            "source": name,
            "group": f"{name}/{src.group_fn(p, base)}",
            "split": "",
        })
    if staged:
        print(f"[scan] {name}: {staged} captures are still unlabelled in {STAGING}/ "
              "and are not in this manifest; label them in the app first")
    if skipped:
        print(f"[warn] {name}: dropped dirs outside the 29-class vocabulary: {sorted(skipped)}")
    if src.verbatim_labels:
        print(f"[scan] {name}: labels kept verbatim with label_idx=-1; "
              "this source is for auditing only and cannot be trained on")
    print(f"[scan] {name}: {len(rows)} images, {len({r['group'] for r in rows})} groups")
    return rows


def write_manifest(rows: Iterable[dict], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {len(rows)} rows -> {out}")
    return out


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label_idx"] = int(r["label_idx"])
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="fetch corpora and build a manifest")
    ap.add_argument("cmd", choices=["list", "download", "manifest"])
    ap.add_argument("names", nargs="*", help="source names, or 'all'")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    names = list(REGISTRY) if (not a.names or a.names == ["all"]) else a.names

    if a.cmd == "list":
        for s in REGISTRY.values():
            print(f"{s.name:20s} {s.kind:5s} {s.role:8s} {s.ref}")
            print(f"{'':20s} {s.note}")
        return 0

    if a.cmd == "download":
        for n in names:
            download(n, a.root)
        return 0

    rows: list[dict] = []
    for n in names:
        try:
            rows += scan(n, a.root)
        except FileNotFoundError as e:
            print(f"[skip] {e}", file=sys.stderr)
    out = a.out or (a.root.parent / "manifests" / "all.csv")
    write_manifest(rows, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
