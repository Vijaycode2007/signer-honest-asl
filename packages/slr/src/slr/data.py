from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import sources

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _deal_units(rows: list[dict], val: float, test: float, seed: int,
                key: str = "group") -> list[dict]:
    units: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        units[str(r[key])].append(i)

    rng = random.Random(seed)
    names = sorted(units)
    rng.shuffle(names)

    n = len(rows)
    targets = {"train": (1.0 - val - test) * n, "val": val * n, "test": test * n}
    filled = {"train": 0, "val": 0, "test": 0}
    assigned: dict[str, str] = {}

    def labels_of(u: str) -> set[str]:
        return {rows[i]["label"] for i in units[u]}

    def class_in(split: str) -> set[str]:
        out: set[str] = set()
        for u, s in assigned.items():
            if s == split:
                out |= labels_of(u)
        return out

    def unassigned_with(lab: str, exclude: str | None = None) -> list[str]:
        return [u for u in units
                if u not in assigned and u != exclude and lab in labels_of(u)]

    def legal(u: str, split: str) -> bool:
        if split == "train":
            return True
        if targets[split] <= 0:
            return False
        train_short = filled["train"] < targets["train"]
        if train_short and filled[split] + len(units[u]) > filled["train"]:
            return False
        if train_short and filled[split] >= targets[split]:
            return False
        return all(lab in class_in("train") or unassigned_with(lab, exclude=u)
                   for lab in labels_of(u))

    def pick(u: str) -> str:
        scored: list[tuple[float, int, str]] = []
        for split in ("train", "val", "test"):
            if not legal(u, split):
                continue
            size_def = targets[split] - filled[split]
            cover = sum(1 for lab in labels_of(u) if lab not in class_in(split))
            tie = 0 if split == "train" else (1 if split == "val" else 2)
            scored.append((-(size_def + 50.0 * cover), tie, split))
        if not scored:
            return "train"
        scored.sort()
        return scored[0][2]

    for u in sorted(names, key=lambda g: (-len(units[g]), g)):
        dest = pick(u)
        assigned[u] = dest
        filled[dest] += len(units[u])

    present = {r["label"] for r in rows}
    for lab in sorted(present):
        if lab in class_in("train"):
            continue
        cands = [(len(units[u]), u) for u, s in assigned.items()
                 if s != "train" and lab in labels_of(u)]
        if not cands:
            continue
        _, u = min(cands)
        old = assigned[u]
        assigned[u] = "train"
        filled[old] -= len(units[u])
        filled["train"] += len(units[u])

    for u, dest in assigned.items():
        for i in units[u]:
            rows[i]["split"] = dest
    return rows


def split_by_group(rows: list[dict], val: float = 0.15, test: float = 0.15,
                   seed: int = 0) -> list[dict]:
    """Assign splits so each group lands in exactly one split.

    Train always gets every class present in `rows`. Sizes aim for 70/15/15; a
    group bigger than a val/test share goes to train.
    """
    _deal_units(rows, val, test, seed, key="group")
    _report(rows)
    return rows


def split_random_stratified(rows: list[dict], val: float = 0.15, test: float = 0.15,
                            seed: int = 0) -> list[dict]:
    """Per-class 70/15/15 that ignores groups, so near-copies can end up on both sides."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    for items in by_label.values():
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test))
        n_val = int(round(n * val))
        n_train = n - n_val - n_test
        if n_train < 1 and n:
            if n_val:
                n_val -= 1
            else:
                n_test -= 1
            n_train += 1
        for i, r in enumerate(items):
            r["split"] = ("train" if i < n_train
                          else "val" if i < n_train + n_val
                          else "test")
    _report(rows, allow_straddle=True)
    return rows


def carve_tuning_split(rows: list[dict], frac: float = 0.5, max_dist: int = 6,
                       seed: int = 0, workers: int = 6) -> tuple[list[dict], list[dict]]:
    """Split a holdout corpus into a tuning half and a reporting half, by group."""
    from . import leakage

    work = [dict(r) for r in rows]
    leakage.recover_groups(work, max_dist=max_dist, workers=workers)
    for r in work:
        r["split"] = ""
    split_by_group(work, val=frac, test=0.0, seed=seed)

    by_path = {r["path"]: r["split"] for r in work}
    tune = [r for r in rows if by_path[r["path"]] == "val"]
    report = [r for r in rows if by_path[r["path"]] == "train"]
    print(f"[tuning] {len(tune)} to tune on, {len(report)} held for reporting, "
          f"group-disjoint")
    return tune, report


def split_cross_source(rows: list[dict], train_sources: list[str],
                       test_sources: list[str], val: float = 0.15,
                       seed: int = 0) -> list[dict]:
    """Train on some corpora and test on others.

    Validation comes out of the training corpora, split by group.
    """
    overlap = set(train_sources) & set(test_sources)
    if overlap:
        raise ValueError(f"train and test sources overlap: {sorted(overlap)}")

    keep, dropped = [], Counter()
    for r in rows:
        if r["source"] in train_sources:
            r["split"] = "train"
            keep.append(r)
        elif r["source"] in test_sources:
            r["split"] = "test"
            keep.append(r)
        else:
            dropped[r["source"]] += 1
    if dropped:
        print(f"[split] ignoring sources not named: {dict(dropped)}")

    train_rows = [r for r in keep if r["source"] in train_sources]
    _deal_units(train_rows, val=val, test=0.0, seed=seed, key="group")
    _report(keep)
    return keep


def _report(rows: list[dict], allow_straddle: bool = False) -> None:
    by_split = Counter(r["split"] for r in rows)
    print(f"[split] {dict(by_split)}")
    vocab = {r["label"] for r in rows}
    for s in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == s]
        if not sub:
            continue
        missing = vocab - {r["label"] for r in sub}
        g = len({r["group"] for r in sub})
        print(f"[split]   {s}: {len(sub)} imgs, {g} groups, "
              f"{len(vocab) - len(missing)}/{len(vocab)} classes"
              + (f", MISSING {sorted(missing)}" if missing else ""))
    if allow_straddle:
        return
    seen: dict[str, str] = {}
    for r in rows:
        if r["split"] and seen.setdefault(r["group"], r["split"]) != r["split"]:
            raise AssertionError(f"group {r['group']} straddles splits - split is invalid")


def build_transforms(img_size: int, train: bool):
    from torchvision import transforms as T

    if not train:
        return T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
        T.RandomApply([T.RandomRotation(18)], p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])


class SignDataset(Dataset):
    def __init__(self, rows: list[dict], img_size: int = 224, train: bool = False):
        bad = [r for r in rows if r["label_idx"] < 0]
        if bad:
            raise ValueError(
                f"{len(bad)} rows have label_idx=-1 (not one of the 29 classes), "
                f"from {sorted({r['source'] for r in bad})}. Drop them before training."
            )
        self.rows = rows
        self.tf = build_transforms(img_size, train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        with Image.open(r["path"]) as im:
            img = im.convert("RGB")
        return self.tf(img), r["label_idx"]


def loaders(rows: list[dict], img_size: int, batch_size: int, workers: int = 4
            ) -> dict[str, DataLoader]:
    out: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        is_train = split == "train"
        out[split] = DataLoader(
            SignDataset(sub, img_size, train=is_train),
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train,
            persistent_workers=workers > 0,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="assign leakage-safe splits")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--policy", choices=["group", "cross_source"], default="group")
    ap.add_argument("--train-sources", nargs="*", default=None)
    ap.add_argument("--test-sources", nargs="*", default=None)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    rows = sources.read_manifest(a.manifest)
    if a.policy == "group":
        rows = split_by_group(rows, a.val, a.test, a.seed)
    else:
        tr = a.train_sources or [n for n, s in sources.REGISTRY.items() if s.role == "train"]
        te = a.test_sources or [n for n, s in sources.REGISTRY.items() if s.role == "holdout"]
        print(f"[split] train on {tr} -> test on {te}")
        rows = split_cross_source(rows, tr, te, a.val, a.seed)

    sources.write_manifest(rows, a.out or a.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
