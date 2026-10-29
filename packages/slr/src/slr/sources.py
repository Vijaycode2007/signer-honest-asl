"""Dataset registry, downloads and manifests.

Each source says how to fetch it, where its labels are, and how to compute a
group key. Rows that share a group (e.g. frames from one session) are kept in
the same split. Sources without one get groups from `slr.leakage`.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

# --- label space -------------------------------------------------------------

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
CLASSES = LETTERS + ["space", "del", "nothing"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Datasets name the three non-letter classes differently.
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
    """The whole corpus is one group, so it is never split internally."""
    return "all"


_DIGITS_RE = re.compile(r"IMG_(\d+)", re.I)


def group_sl_digits(path: Path, root: Path) -> str:
    """Signer id for ardamavi/Sign-Language-Digits-Dataset.

    Each student took ten shots in a row (digits 0-9) starting at IMG_1118, so
    the image number tells you the student.
    """
    m = _DIGITS_RE.search(path.name)
    if not m:
        return f"unknown:{path.name}"
    return f"signer{(int(m.group(1)) - 1118) // 10:04d}"


_INDEXED_RE = re.compile(r"^([A-Za-z_]+)(\d+)")


def group_by_frame_block(block: int) -> Callable[[Path, Path], str]:
    """Group `<class><frame>.jpg` files into blocks of `block` frames.

    The class stays in the key because the frame counter restarts per class.
    """

    def _fn(path: Path, root: Path) -> str:
        m = _INDEXED_RE.match(path.stem)
        if not m:
            return f"unknown:{path.name}"
        return f"{m.group(1)}blk{int(m.group(2)) // block:05d}"

    return _fn


# --- source registry ---------------------------------------------------------

@dataclass(frozen=True)
class Source:
    name: str
    kind: str                      # "kaggle" | "kaggle_comp" | "hf" | "zip" | "tgz" | "local"
    ref: str                       # dataset slug / repo id / url
    role: str                      # "train" | "holdout" | "either" | "audit" | "excluded"
    media: str = "image"           # "image" | "video" | "landmarks"
    group_fn: Callable[[Path, Path], str] = group_by_parent_dir
    label_depth: int = 1           # how many dirs up from the file the class name sits
    subdir: str = ""               # path inside the download to scan
    note: str = ""
    files: tuple[str, ...] = ()    # fetch only these files
    needs_group_recovery: bool = False
    verbatim_labels: bool = False  # take dir names as labels, outside the 29-class space
    sha256: str = ""               # archive digest for zip/tgz; see `download`


REGISTRY: dict[str, Source] = {s.name: s for s in [
    Source(
        name="asl_alphabet",
        kind="kaggle", ref="grassknoted/asl-alphabet", role="train",
        subdir="asl_alphabet_train/asl_alphabet_train",
        group_fn=group_by_frame_block(200),
        needs_group_recovery=True,
        note="87k images, 29 classes, mostly one signer on a webcam. Its 28-image "
             "test set is too small to use.",
    ),
    Source(
        name="asl_alphabet_test",
        kind="kaggle", ref="danrasband/asl-alphabet-test", role="holdout",
        group_fn=group_by_parent_dir,
        note="870 images by a different person on varied backgrounds, made to "
             "test models trained on asl_alphabet. Main test set.",
    ),
    Source(
        name="asl_alphabet_v2",
        kind="kaggle", ref="debashishsau/aslamerican-sign-language-aplhabet-dataset",
        role="excluded", subdir="ASL_Alphabet_Dataset/asl_alphabet_train",
        group_fn=group_by_frame_block(200), needs_group_recovery=True,
        note="223k images, but 170,970 are copies of asl_alphabet and 1,488 of "
             "asl_alphabet_test. Do not train on it.",
    ),
    Source(
        name="asl_27class",
        kind="kaggle", ref="ardamavi/27-class-sign-language-dataset", role="either",
        needs_group_recovery=True,
        note="Mavi & Dikle 2022, arXiv:2203.03859. 173 people, but only A-E are "
             "letters. Ships as X.npy/Y.npy, which `scan` cannot read.",
    ),
    Source(
        name="mine",
        kind="local", ref="photos taken in the app and labelled in the page",
        role="either",
        group_fn=lambda p, root: p.relative_to(root).parts[0],
        note="Photos taken in the app, filed as <session>/<LABEL>/. The session "
             "is the group. One signer only. Local, never committed.",
    ),
    Source(
        name="asl_alphabets_v03",
        kind="hf", ref="Marxulia/asl_sign_languages_alphabets_v03", role="either",
        subdir="images", group_fn=group_by_frame_block(100), needs_group_recovery=True,
        note="10.9k images, 26 letters, many people and backgrounds, some "
             "drawings. Saved as images/<letter>/img<row>.png. It made the "
             "holdout and own-hand scores worse, so the served model skips it.",
    ),

    Source(
        name="signalphaset",
        kind="zip",
        ref="https://data.mendeley.com/public-files/datasets/8fmvr9m98w/files/"
            "a12e2b20-74de-4301-858f-35ab312d77b7/file_downloaded",
        role="train", subdir="SignAlphaSet",
        group_fn=group_by_frame_block(100), needs_group_recovery=True,
        note="SignAlphaSet (Data in Brief 2025, doi:10.17632/8fmvr9m98w.3). 26k "
             "images, 1000 per letter, black background, letters only. Used for "
             "its left-hand images: asl_alphabet is 92.5% right hands. There are "
             "no signer or handedness labels, so groups are filename blocks.",
    ),
    Source(
        name="asl_dynamic",
        kind="zip", media="video",
        ref="https://data.mendeley.com/public-files/datasets/8fmvr9m98w/files/"
            "9dcdeaa8-4df1-4eb9-98df-3231f66a10c7/file_downloaded",
        role="either", subdir="ASL_dynamic",
        note="Video from the same release: 300 AVI clips, ten per letter plus "
             "HELLO, THANKYOU, SORRY, YES and NO. Same signers as signalphaset.",
    ),
]}

REPO = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = REPO / "dataset" / "raw"


# --- downloading -------------------------------------------------------------

def _unpack_members(dest: Path, members: Iterable[str]) -> None:
    """Unzip Kaggle single-file downloads and move them back into their folders."""
    for z in dest.rglob("*.zip"):
        zipfile.ZipFile(z).extractall(z.parent)
        z.unlink()
    for member in members:
        sub = Path(member)
        flat = dest / sub.name
        if sub.parent != Path(".") and flat.exists():
            (dest / sub.parent).mkdir(parents=True, exist_ok=True)
            flat.rename(dest / sub)


def _check_digest(src: Source, got: str, archive: Path) -> None:
    """Check a download against its pinned sha256, or print the hash to pin."""
    if not src.sha256:
        print(f"[hash] {src.name}: sha256={got} (not pinned)")
        return
    if got.lower() != src.sha256.lower():
        archive.unlink(missing_ok=True)
        raise SystemExit(
            f"{src.name}: sha256 mismatch\n"
            f"  expected {src.sha256}\n  got      {got}")
    print(f"[hash] {src.name}: sha256 matches the pin")


def _parquet_to_images(dest: Path) -> None:
    """Write a HuggingFace image dataset's parquet shards out as image files."""
    import io
    import json

    import pyarrow.parquet as pq
    from PIL import Image

    row = 0
    for shard in sorted((dest / "data").glob("[!.]*.parquet")):
        t = pq.read_table(shard)
        names = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
        for img, label in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
            out = dest / "images" / names[label] / f"img{row:06d}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(out)
            row += 1


def download(name: str, root: Path = DEFAULT_ROOT) -> Path:
    """Fetch one source into `root/<name>`. Idempotent: skips a non-empty dir."""
    src = REGISTRY[name]
    dest = root / name
    if dest.exists() and any(dest.rglob("*")):
        print(f"[skip] {name} already at {dest}")
        return dest
    if src.kind == "local":
        print(f"[local] {name} is not downloaded; label photos in the app and "
              f"they go to {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    has_kaggle_creds = (bool(os.environ.get("KAGGLE_USERNAME"))
                        or (Path.home() / ".kaggle" / "kaggle.json").exists())
    if src.kind in ("kaggle", "kaggle_comp") and not has_kaggle_creds:
        raise SystemExit(
            f"{name} needs Kaggle credentials: put your token in "
            "~/.kaggle/kaggle.json or set KAGGLE_USERNAME / KAGGLE_KEY."
        )

    if src.kind == "kaggle":
        if src.files:
            for member in src.files:
                subprocess.run(
                    [sys.executable, "-m", "kaggle", "datasets", "download",
                     src.ref, "-f", member, "-p", str(dest)], check=True)
                print(f"[ok] {member}")
            _unpack_members(dest, src.files)
        else:
            subprocess.run(
                [sys.executable, "-m", "kaggle", "datasets", "download",
                 "-d", src.ref, "-p", str(dest), "--unzip"],
                check=True,
            )
    elif src.kind == "kaggle_comp":
        # A 403 here almost always means the rules have not been accepted.
        for member in src.files:
            p = subprocess.run(
                [sys.executable, "-m", "kaggle", "competitions", "download",
                 src.ref, "-f", member, "-p", str(dest)],
                capture_output=True, text=True)
            if p.returncode or "403" in p.stdout + p.stderr:
                raise SystemExit(
                    f"Kaggle refused {member} for {src.ref}.\n"
                    f"  {p.stdout.strip()} {p.stderr.strip()}\n"
                    f"Accept the rules first: https://www.kaggle.com/competitions/"
                    f"{src.ref}/rules")
            print(f"[ok] {member}")
        _unpack_members(dest, src.files)
    elif src.kind == "hf":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=src.ref, repo_type="dataset", local_dir=str(dest))
        _parquet_to_images(dest)
    elif src.kind in ("zip", "tgz"):
        print(f"[get] {src.ref}")
        # Stream to disk, archives can be huge.
        tmp = dest / f"_download{'.zip' if src.kind == 'zip' else '.tgz'}"
        digest = hashlib.sha256()
        with urllib.request.urlopen(src.ref, timeout=300) as r, tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                digest.update(chunk)
                f.write(chunk)
        _check_digest(src, digest.hexdigest(), tmp)
        if src.kind == "zip":
            zipfile.ZipFile(tmp).extractall(dest)
        else:
            import tarfile
            with tarfile.open(tmp) as t:
                t.extractall(dest, filter="data")
        tmp.unlink()
    else:
        raise ValueError(f"unknown source kind {src.kind}")

    print(f"[ok] {name} -> {dest}")
    return dest


# --- manifest ----------------------------------------------------------------

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Unlabelled app captures wait here. Never part of a manifest.
STAGING = "_inbox"
MANIFEST_COLS = ["path", "label", "label_idx", "source", "group", "split"]


def scan(name: str, root: Path = DEFAULT_ROOT) -> list[dict]:
    """Walk one downloaded source and emit manifest rows with an empty `split`."""
    src = REGISTRY[name]
    if src.media != "image":
        raise ValueError(f"{name} is a {src.media} corpus, not images")
    base = root / name / src.subdir if src.subdir else root / name
    if not base.exists():
        raise FileNotFoundError(
            f"{base} missing - run `python -m slr.sources download {name}` first")

    rows: list[dict] = []
    skipped: set[str] = set()
    staged = 0
    for p in sorted(base.rglob("*")):
        # Skip macOS `._` metadata files.
        if p.suffix.lower() not in IMG_EXT or p.name.startswith("._"):
            continue
        parts = p.relative_to(base).parts
        if len(parts) <= src.label_depth:
            continue
        raw = parts[-(src.label_depth + 1)]
        if raw == STAGING:
            staged += 1
            continue
        if src.verbatim_labels:
            label, idx = raw, -1        # not trainable
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
        print(f"[scan] {name}: skipped {staged} unlabelled captures in {STAGING}/")
    if skipped:
        print(f"[warn] {name}: dropped dirs outside the 29-class vocabulary: {sorted(skipped)}")
    if src.verbatim_labels:
        print(f"[scan] {name}: labels kept as-is with label_idx=-1 (not trainable)")
    print(f"[scan] {name}: {len(rows)} images, {len({r['group'] for r in rows})} groups")
    return rows


def repo_relative(path: str | Path) -> str:
    """Path relative to REPO if it is inside it, so manifests survive a move."""
    try:
        return Path(path).relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def resolve(path: str | Path) -> str:
    """A stored path as an absolute one. Absolute paths pass through unchanged."""
    return str(REPO / path)


def write_manifest(rows: Iterable[dict], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    # Fail on a missing column rather than write an empty group.
    rows = [{**{k: r[k] for k in MANIFEST_COLS}, "path": repo_relative(r["path"])}
            for r in rows]
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
        r["path"] = resolve(r["path"])
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
    if a.cmd == "manifest" and (not a.names or a.names == ["all"]):
        names = [n for n in names
                 if REGISTRY[n].media == "image" and REGISTRY[n].role != "excluded"]

    if a.cmd == "list":
        for s in REGISTRY.values():
            print(f"{s.name:20s} {s.kind:11s} {s.media:9s} {s.role:8s} {s.ref}")
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
