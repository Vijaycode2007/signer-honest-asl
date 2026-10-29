from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HASH_BITS = 64
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def unique_image_indices(paths: Sequence[str | Path], labels: Sequence
                         ) -> tuple[np.ndarray, dict]:
    """Indices of the first copy of each pixel-identical image.

    Raises if two identical images have different labels. Similar or resized
    images are not merged.
    """
    if len(paths) != len(labels):
        raise ValueError("need one label per image path")
    seen, keep = {}, []
    roster = hashlib.sha256()
    for i, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            size = b"".join(n.to_bytes(8, "big") for n in rgb.size)
            digest = hashlib.sha256(size + rgb.tobytes()).digest()
        roster.update(digest + str(label).encode() + b"\0")
        if digest in seen:
            first, first_label = seen[digest]
            if first_label != label:
                raise ValueError(f"identical images have conflicting labels: "
                                 f"{paths[first]} ({first_label}) and {path} ({label})")
        else:
            seen[digest] = (i, label)
            keep.append(i)
    return np.array(keep, dtype=np.int64), {
        "method": "SHA256 of RGB dimensions and exact pixels",
        "input_rows": len(paths), "unique_images": len(keep),
        "duplicates_removed": len(paths) - len(keep), "roster_sha256": roster.hexdigest(),
    }


def dhash(path: str | Path, size: int = 8) -> int:
    """64-bit difference hash: grey, resize to (size+1, size), compare columns."""
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int(np.packbits(bits).view(">u8")[0])


def hash_many(paths: Sequence[str], workers: int = 0) -> np.ndarray:
    """Hash a list of images, returning a uint64 array aligned with `paths`."""
    workers = workers or min(8, (len(paths) // 500) + 1)
    if workers <= 1:
        return np.array([dhash(p) for p in paths], dtype=np.uint64)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return np.fromiter(ex.map(dhash, paths), dtype=np.uint64, count=len(paths))


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise Hamming distance between two uint64 arrays, elementwise."""
    x = np.bitwise_xor(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64))
    return _POPCOUNT8[x.view(np.uint8).reshape(-1, 8)].sum(axis=1).astype(np.int16)


def _bands(h: np.ndarray, n_bands: int) -> np.ndarray:
    width = HASH_BITS // n_bands
    mask = np.uint64((1 << width) - 1)
    return np.stack([(h >> np.uint64(i * width)) & mask for i in range(n_bands)], axis=1)


def candidate_pair_blocks(h: np.ndarray, max_dist: int, cap: int = 4000,
                          stats: dict | None = None
                          ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield `(ia, ib)` index arrays of pairs that could be within `max_dist` bits.

    Pigeonhole: with B = max_dist + 1 bands, a pair differing in at most
    max_dist bits leaves at least one band untouched, so it collides there.

    Buckets larger than `cap` (a flat band shared by thousands of images) are
    skipped, so a close pair that only collides inside them is missed. Pass a
    `stats` dict to learn whether that happened, and raise `cap` if the answer
    has to be exact.
    """
    n_bands = max(2, max_dist + 1)
    dropped_buckets = dropped_rows = 0
    for col in _bands(h, n_bands).T:
        order = np.argsort(col, kind="stable")
        vals = col[order]
        starts = np.flatnonzero(np.r_[True, vals[1:] != vals[:-1]])
        for s, e in zip(starts, np.r_[starts[1:], len(vals)]):
            if e - s < 2:
                continue
            if e - s > cap:
                dropped_buckets += 1
                dropped_rows += int(e - s)
                continue
            idx = order[s:e]
            ia, ib = np.triu_indices(len(idx), k=1)
            yield idx[ia], idx[ib]
    if stats is not None:
        stats["dropped_buckets"] = dropped_buckets
        stats["dropped_rows"] = dropped_rows
        stats["cap"] = cap
        stats["exact"] = dropped_buckets == 0


def candidate_pairs(h: np.ndarray, max_dist: int) -> Iterable[tuple[int, int]]:
    """`candidate_pair_blocks`, one pair at a time."""
    for ia, ib in candidate_pair_blocks(h, max_dist):
        yield from zip(ia.tolist(), ib.tolist())


class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def recover_groups(rows: list[dict], max_dist: int = 6, workers: int = 0,
                   collapse_at: float = 0.5) -> list[dict]:
    """Set each row's `group` to its near-duplicate cluster within its source.

    If one cluster holds more than `collapse_at` of a source, that source keeps
    its old groups. Also stores each row's hash for `audit` to reuse.
    """
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_source[r["source"]].append(i)

    for source, idxs in by_source.items():
        paths = [rows[i]["path"] for i in idxs]
        h = hash_many(paths, workers)
        for i, hv in zip(idxs, h):
            rows[i]["dhash"] = int(hv)

        dsu = _DSU(len(idxs))
        n_edges = 0
        stats: dict = {}
        for ia, ib in candidate_pair_blocks(h, max_dist, stats=stats):
            keep = hamming(h[ia], h[ib]) <= max_dist
            n_edges += int(keep.sum())
            for a, b in zip(ia[keep].tolist(), ib[keep].tolist()):
                dsu.union(a, b)
        if not stats["exact"]:
            print(f"[groups] {source}: skipped {stats['dropped_buckets']} buckets "
                  f"over {stats['cap']} images, so some close pairs may be missed")

        roots = [dsu.find(local) for local in range(len(idxs))]
        biggest = Counter(roots).most_common(1)[0][1]
        if biggest > collapse_at * len(idxs):
            print(f"[groups] {source}: one cluster holds {biggest}/{len(idxs)} "
                  f"images ({biggest / len(idxs):.1%}); keeping the existing groups")
            continue

        comps: dict[int, int] = {}
        for local, gi in enumerate(idxs):
            cid = comps.setdefault(roots[local], len(comps))
            rows[gi]["group"] = f"{source}/dup{cid:06d}"
        print(f"[groups] {source}: {len(idxs)} imgs -> {len(comps)} components "
              f"({n_edges} near-dup edges, d<={max_dist})")

        if biggest > 0.2 * len(idxs):
            print(f"[groups] {source}: warning, largest cluster holds "
                  f"{biggest}/{len(idxs)} images ({biggest / len(idxs):.1%}); "
                  f"check the split sizes")
    return rows


def audit(rows: list[dict], max_dist: int = 6, workers: int = 0) -> dict:
    """Count val/test images that have a near-duplicate in train.

    Marks those rows with `leak_dist` and `leak_of` for `purge`.
    """
    need = [r for r in rows if "dhash" not in r]
    for r, hv in zip(need, hash_many([r["path"] for r in need], workers)):
        r["dhash"] = int(hv)

    train = [r for r in rows if r["split"] == "train"]
    if not train:
        raise ValueError("manifest has no train rows - assign splits first")
    th = np.array([r["dhash"] for r in train], dtype=np.uint64)

    n_bands = max(2, max_dist + 1)
    buckets = [defaultdict(list) for _ in range(n_bands)]
    for i, row in enumerate(_bands(th, n_bands).tolist()):
        for b, v in enumerate(row):
            buckets[b][v].append(i)

    report: dict = {"max_dist": max_dist, "n_train": len(train), "splits": {}}
    for split in sorted({r["split"] for r in rows} - {"train", ""}):
        probe = [r for r in rows if r["split"] == split]
        ph = np.array([r["dhash"] for r in probe], dtype=np.uint64)
        probe_bands = _bands(ph, n_bands).tolist()

        exact = near = 0
        per_source: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "leaked": 0})
        for j, r in enumerate(probe):
            per_source[r["source"]]["n"] += 1
            cands: set[int] = set()
            for b, v in enumerate(probe_bands[j]):
                cands.update(buckets[b].get(v, ()))
            if not cands:
                continue
            ci = np.fromiter(cands, dtype=np.int64)
            d = hamming(th[ci], np.full(len(ci), ph[j], dtype=np.uint64))
            k = int(d.argmin())
            if d[k] <= max_dist:
                near += 1
                exact += int(d[k] == 0)
                per_source[r["source"]]["leaked"] += 1
                r["leak_dist"] = int(d[k])
                r["leak_of"] = train[int(ci[k])]["path"]

        n = len(probe)
        report["splits"][split] = {
            "n": n,
            "exact_duplicates": exact,
            "near_duplicates": near,
            "leak_rate": round(near / n, 4) if n else 0.0,
            "by_source": {s: {**v, "leak_rate": round(v["leaked"] / v["n"], 4)}
                          for s, v in per_source.items()},
        }
        print(f"[audit] {split}: {near}/{n} ({near / max(n, 1):.1%}) have a train "
              f"near-duplicate within {max_dist} bits ({exact} pixel-identical)")
    return report


def purge(rows: list[dict]) -> list[dict]:
    """Drop the val/test rows `audit` flagged."""
    keep = [r for r in rows if r["split"] == "train" or "leak_dist" not in r]
    print(f"[purge] {len(rows) - len(keep)} contaminated eval rows removed")
    return keep


def main(argv: list[str] | None = None) -> int:
    import argparse

    from . import sources

    ap = argparse.ArgumentParser(description="near-duplicate audit and group recovery")
    ap.add_argument("cmd", choices=["groups", "audit"])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="rewrite the manifest here")
    ap.add_argument("--report", type=Path, default=Path("runs/leakage_report.json"))
    ap.add_argument("--max-dist", type=int, default=6)
    ap.add_argument("--purge", action="store_true", help="drop contaminated eval rows")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(argv)

    rows = sources.read_manifest(a.manifest)
    if a.cmd == "groups":
        recover_groups(rows, a.max_dist, a.workers)
    else:
        rep = audit(rows, a.max_dist, a.workers)
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(rep, indent=2))
        print(f"[ok] report -> {a.report}")
        if a.purge:
            rows = purge(rows)

    sources.write_manifest(rows, a.out or a.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
