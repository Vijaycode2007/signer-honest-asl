"""Splits, transforms, and the torch Dataset.

Two split policies, and only two, because these are the only two that produce a
number worth quoting:

  `group`        - groups (recovered signer/session clusters) are dealt whole to
                   exactly one of train/val/test. No frame of a session can face
                   its neighbour across the split boundary.

  `cross_source` - train on one corpus, test on a different corpus shot by
                   different people in a different room. This is the number that
                   predicts what the phone app does on a stranger's hand, and it
                   is the number nobody reports.

A plain random split is deliberately not implemented. If you want one, you want
`leakage.audit` instead.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .sources import REGISTRY

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --- splitting ---------------------------------------------------------------

def _deal_units(rows: list[dict], val: float, test: float, seed: int,
                key: str = "group") -> list[dict]:
    """Deal whole units (`group` or `signer`) to ~70/15/15 without inversion.

    A first-fit walk that fills val then train fails on heavy-tailed sessions:
    one recovered component is handed to val, val becomes most of the corpus,
    and train is left with a handful of leftover classes. That is the 262-train
    / 11k-val failure. This dealer instead:

      * targets train:(1-val-test), val, test (defaults 70/15/15)
      * places largest units first into the split furthest below its target
      * never places a unit into val/test when that would make the split
        larger than train while train is still short of its target
      * never moves a class's last unit out of train
      * prefers a split that is still missing a class the unit would add
    """
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
        size = len(units[u])
        if split != "train" and targets[split] <= 0:
            return False
        if split != "train":
            would = filled[split] + size
            if filled["train"] < targets["train"] and would > filled["train"]:
                return False
            if filled[split] >= targets[split] and filled["train"] < targets["train"]:
                return False
            for lab in labels_of(u):
                if lab not in class_in("train") and not unassigned_with(lab, exclude=u):
                    return False
        return True

    def pick(u: str) -> str:
        scored: list[tuple[float, str]] = []
        for split in ("train", "val", "test"):
            if not legal(u, split):
                continue
            size_def = targets[split] - filled[split]
            cover = sum(1 for lab in labels_of(u) if lab not in class_in(split))
            # train wins ties so leftover capacity stays where the model learns
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

    # Repair: every class that appears in `rows` must appear in train.
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
    """Assign splits so that no group straddles a boundary.

    Groups are dealt whole. Placement is stratified by class so train always
    contains every letter present in `rows`, and sizes track 70/15/15 unless a
    single session is larger than those fractions (in which case that session
    stays in train rather than emptying it).
    """
    _deal_units(rows, val, test, seed, key="group")
    _report(rows)
    return rows


def split_random_stratified(rows: list[dict], val: float = 0.15, test: float = 0.15,
                            seed: int = 0) -> list[dict]:
    """Per-class 70/15/15. Sessions *are* allowed to straddle; that is protocol A.

    Stratifying by label (not group) is what keeps all 29 classes in train when
    a plain `random() < 0.7` draw would otherwise miss a rare letter.
    """
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
            steal = "val" if n_val else "test"
            if steal == "val":
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
    """Split a holdout corpus into a tuning half and a reporting half, group-disjoint.

    Use this, not an in-corpus validation split, whenever a number has to be
    chosen rather than merely reported: a fusion weight, an abstention threshold,
    a feature set.

    Why validation cannot come from the training corpus. `asl_alphabet` is one
    continuous take in one room, so `leakage.recover_groups` chains it into a
    single component holding 98.5% of the images, and a "group split" of it is not
    a split. That is not fixable by regrouping, and it was measured three ways:
    lowering the hash threshold, hashing the hand crop instead of the frame, and
    bucketing the camera's own frame counter. Frame blocks cannot chain and still
    audited at 72% leakage.

    That 72% is itself misleading, and the measurement explaining everything is
    this: with a whole-frame hash at d<=6, **66.6% of the "near-duplicate" pairs
    in asl_alphabet show different letters**. The hash is reading the room. On the
    hand crop at d<=2 that falls to 2.2%. One room photographed 87,000 times holds
    no honest internal split, and the leak number was overstated about two-to-one
    by a background-dominated hash.

    A holdout corpus shot by a different person across varied backgrounds does
    have real session structure: `asl_alphabet_test` recovers 796 components from
    1,740 images, and halving it on those components audits at **0.00% leakage,
    0 exact duplicates**. That is a selection signal worth having.

    It costs test data, and that is the honest price. What it buys is visible: the
    pixel/geometry fusion weight chosen on the in-corpus validation split came
    back as 0.00, the worst of the candidates; chosen on this tuning half it came
    back as 0.925, which then held up on the untouched reporting half.
    """
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
    """Train on one set of corpora, test on a disjoint set.

    Validation is carved group-disjointly out of the *training* corpora, so the
    test corpora stay completely untouched until the final evaluation. The carve
    uses the same train-first dealer as `split_by_group` so a giant recovered
    session cannot be parked in val and starve train of whole letters.
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
    for r in keep:
        if r["source"] in test_sources:
            r["split"] = "test"

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
    seen: dict[str, str] = {}
    for r in rows:
        if r["split"] and seen.setdefault(r["group"], r["split"]) != r["split"]:
            if allow_straddle:
                break
            raise AssertionError(f"group {r['group']} straddles splits - split is invalid")


# --- dataset -----------------------------------------------------------------

def build_transforms(img_size: int, train: bool):
    """Augmentation is deliberately heavy on the train side.

    The failure mode of these corpora is background memorisation, so the train
    pipeline attacks every cue that is not the hand: crop, rotate, recolour,
    desaturate, and erase patches. The eval pipeline does nothing but resize.
    """
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
    """Manifest-backed image dataset. Never applies a horizontal flip.

    ASL is handed. Mirroring an image turns a right-handed sign into a
    left-handed one, and for a few letter pairs it changes what the sign means.
    torchvision's default recipes flip; ours must not.
    """

    def __init__(self, rows: list[dict], img_size: int = 224, train: bool = False):
        bad = [r for r in rows if r["label_idx"] < 0]
        if bad:
            raise ValueError(
                f"{len(bad)} rows carry label_idx=-1, meaning they come from an "
                f"audit-only source outside the 29-class vocabulary "
                f"(e.g. {sorted({r['source'] for r in bad})}). Exclude them before "
                "training."
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


# Hand cropping lives in `slr.hands`, as an offline pass that mirrors a corpus
# into `dataset/crops/` and hands back a manifest pointing at it. It is not a
# transform here on purpose: detection costs ~25 ms an image, which is an hour
# added to every epoch rather than once to the dataset, and a crop that is
# recomputed every epoch cannot be looked at. Train the same architecture on the
# original manifest and on the cropped one; the difference between the two
# cross-corpus numbers is what cropping is worth.


def main(argv: list[str] | None = None) -> int:
    import argparse

    from . import sources

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
        tr = a.train_sources or [n for n, s in REGISTRY.items() if s.role == "train"]
        te = a.test_sources or [n for n, s in REGISTRY.items() if s.role == "holdout"]
        print(f"[split] train on {tr} -> test on {te}")
        rows = split_cross_source(rows, tr, te, a.val, a.seed)

    sources.write_manifest(rows, a.out or a.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
