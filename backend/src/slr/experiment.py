"""The three-protocol experiment, as one runnable module.

Packaged as code rather than notebook cells so the run is reproducible and so
driving it remotely is a matter of calling `stage(name)` instead of retyping
analysis into a browser.

Stages, in order:

    data   download the corpora, build the manifest, recover session groups
           (`crop=True` additionally mirrors every image to its hand box, and
           every stage after it then runs on crops)
    A      random split          - what the literature reports
    B      group split           - sessions kept whole
    C      cross-corpus          - tested on a corpus with different signers
    arch   architecture search on protocol C, selected on validation
    cal    temperature scaling and the abstention threshold for the winner

Each stage writes `runs/results.json` incrementally, so a disconnected Colab
session loses at most the stage in flight.

Subsampling is group-aware by construction: `--per-class` caps images per class
by dropping whole groups, never by cherry-picking frames, because sampling
individual frames out of a session would quietly reintroduce the leak the
experiment exists to measure.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from . import data, evaluate, hands, leakage, model as M, sources, train

RUNS = Path("runs")
RESULTS = RUNS / "results.json"
MANIFEST = Path("dataset/manifests/all.csv")

TRAIN_CORPUS = "asl_alphabet"
TEST_CORPUS = "asl_alphabet_test"


# --- bookkeeping -------------------------------------------------------------

def load() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}


def save(state: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(state, indent=2))
    print(f"[state] -> {RESULTS}")


def _kaggle_path() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def write_kaggle_credentials(username: str, key: str) -> None:
    p = _kaggle_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"username": username, "key": key}))
    p.chmod(0o600)
    print(f"[kaggle] credentials for {username} written to {p}")


def _install_kaggle_if_available(get=None) -> bool:
    """Return True once ~/.kaggle/kaggle.json exists. Never prints secrets."""
    dest = _kaggle_path()
    if dest.exists():
        dest.chmod(0o600)
        print("[kaggle] credentials already present")
        return True
    downloads = Path.home() / "Downloads" / "kaggle.json"
    if downloads.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(downloads.read_bytes())
        dest.chmod(0o600)
        print(f"[kaggle] copied credentials from {downloads}")
        return True
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        write_kaggle_credentials(user, key)
        return True
    if get is not None:
        try:
            write_kaggle_credentials(get("KAGGLE_USERNAME"), get("KAGGLE_KEY"))
            return True
        except Exception:
            return False
    return False


def wait_for_kaggle(get=None, poll: int = 5) -> None:
    """Block until Kaggle credentials are on disk, then continue.

    Looks for, in order: ~/.kaggle/kaggle.json, ~/Downloads/kaggle.json,
    KAGGLE_USERNAME / KAGGLE_KEY in the environment, then an optional `get`
    callable (Colab Secrets). Does not raise SystemExit, so a notebook kernel
    is not killed while the file is still missing.
    """
    dest = _kaggle_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _install_kaggle_if_available(get):
        return
    print("[kaggle] missing credentials. Create a legacy API token at")
    print("         https://www.kaggle.com/settings  -> API")
    print(f"         then put kaggle.json at {dest} (chmod 600), or drop it")
    print("         in ~/Downloads, or export KAGGLE_USERNAME and KAGGLE_KEY.")
    print(f"[kaggle] waiting; re-checking every {poll}s", flush=True)
    while not _install_kaggle_if_available(get):
        time.sleep(poll)


# --- sampling ----------------------------------------------------------------

def subsample(rows: list[dict], per_class: int, seed: int = 0) -> list[dict]:
    """Cap images per class, spreading the budget across sessions when possible.

    Sessions are never shuffled internally: a contiguous prefix is kept so a
    subsample cannot itself create or hide a leak. When a class has several
    recovered sessions they share the cap (up to three) so a later 70/15/15
    group split can put that class in train, val, and test. A single oversized
    session still falls back to one prefix, which is the only way to honour a
    cap of a few hundred on a session of thousands of frames.
    """
    if not per_class:
        return rows
    by_class_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_class_group[r["label"]][r["group"]].append(r)

    rng = random.Random(seed)
    kept: list[dict] = []
    for label, groups in by_class_group.items():
        names = sorted(groups)
        rng.shuffle(names)
        n_parts = min(len(names), 3)
        base, extra = divmod(per_class, n_parts)
        budgets = [base + (1 if i < extra else 0) for i in range(n_parts)]
        taken_from: dict[str, int] = {}
        for g, b in zip(names, budgets):
            take = groups[g][:b]
            kept += take
            taken_from[g] = len(take)
        n = sum(taken_from.values())
        if n < per_class:
            for g in names:
                if n >= per_class:
                    break
                start = taken_from.get(g, 0)
                extra_take = groups[g][start:start + (per_class - n)]
                kept += extra_take
                taken_from[g] = start + len(extra_take)
                n += len(extra_take)
    print(f"[sample] {len(rows)} -> {len(kept)} images "
          f"(<={per_class}/class, whole groups only)")
    return kept


# --- stages ------------------------------------------------------------------

def stage_data(per_class: int = 0, max_dist: int = 6, crop: bool = False) -> dict:
    """Build the manifest every other stage reads.

    `crop=True` mirrors the corpora to their hand boxes and points the manifest
    at the mirror, which is the whole switch: A, B, C, arch and cal all read
    MANIFEST, so one flag here decides what the entire experiment is run on.
    Re-running without the flag puts it back. Running both and comparing the two
    protocol-C numbers is what says whether cropping was worth anything.
    """
    for name in (TRAIN_CORPUS, TEST_CORPUS):
        sources.download(name)

    rows: list[dict] = []
    for name in (TRAIN_CORPUS, TEST_CORPUS):
        rows += sources.scan(name)

    t0 = time.time()
    leakage.recover_groups(rows, max_dist=max_dist, workers=8)
    print(f"[data] group recovery took {time.time() - t0:.0f}s")

    sources.write_manifest([{k: r[k] for k in sources.MANIFEST_COLS} for r in rows], MANIFEST)
    summary = {
        "images": len(rows),
        "by_source": {s: sum(1 for r in rows if r["source"] == s)
                      for s in {r["source"] for r in rows}},
        "groups": len({r["group"] for r in rows}),
        "classes": len({r["label"] for r in rows}),
        "recovery_seconds": round(time.time() - t0, 1),
    }
    if crop:
        rows, det = hands.crop_manifest(rows, workers=8)
        sources.write_manifest([{k: r[k] for k in sources.MANIFEST_COLS} for r in rows],
                               MANIFEST)
        summary["hand_crop"] = det
    summary["cropped"] = crop
    print(f"[data] {summary}")
    return summary


def _rows(per_class: int, source: str | None = None) -> list[dict]:
    rows = sources.read_manifest(MANIFEST)
    if source:
        rows = [r for r in rows if r["source"] == source]
    return subsample(rows, per_class)


def _protocol_result(rows: list[dict], res: dict, rep: dict) -> dict:
    return {
        "n_train": sum(1 for r in rows if r["split"] == "train"),
        "n_test": sum(1 for r in rows if r["split"] == "test"),
        "test_groups": len({r["group"] for r in rows if r["split"] == "test"}),
        "test_leak_rate": rep["splits"].get("test", {}).get("leak_rate"),
        "test_exact_duplicates": rep["splits"].get("test", {}).get("exact_duplicates"),
        "test_acc": res.get("test", {}).get("acc"),
        "test_macro_f1": res.get("test", {}).get("macro_f1"),
        "val_f1": res["best"]["val_f1"],
        "params_m": res["params_m"],
    }


def stage_A(per_class: int, preset: str, epochs: int, workers: int = 2) -> dict:
    """Random split. Sessions are scattered across train and test on purpose."""
    rows = _rows(per_class, TRAIN_CORPUS)
    data.split_random_stratified(rows, val=0.15, test=0.15, seed=0)
    rep = leakage.audit(rows, max_dist=6, workers=8)
    res = train.train_one(rows, preset, epochs, RUNS, workers, tag="A_random")
    return _protocol_result(rows, res, rep)


def stage_B(per_class: int, preset: str, epochs: int, workers: int = 2) -> dict:
    """Group split. Same corpus, same model, sessions kept whole."""
    rows = _rows(per_class, TRAIN_CORPUS)
    data.split_by_group(rows, val=0.15, test=0.15, seed=0)
    rep = leakage.audit(rows, max_dist=6, workers=8)
    res = train.train_one(rows, preset, epochs, RUNS, workers, tag="B_group")
    return _protocol_result(rows, res, rep)


def _cross_rows(per_class: int) -> list[dict]:
    tr = subsample([r for r in sources.read_manifest(MANIFEST)
                    if r["source"] == TRAIN_CORPUS], per_class)
    te = [r for r in sources.read_manifest(MANIFEST) if r["source"] == TEST_CORPUS]
    return data.split_cross_source(tr + te, [TRAIN_CORPUS], [TEST_CORPUS], val=0.15)


def stage_C(per_class: int, preset: str, epochs: int, workers: int = 2) -> dict:
    """Cross-corpus. Different signer, room, and camera. Signer-disjoint by
    construction, with no need to recover anyone's identity."""
    rows = _cross_rows(per_class)
    rep = leakage.audit(rows, max_dist=6, workers=8)
    res = train.train_one(rows, preset, epochs, RUNS, workers, tag="C_cross")
    out = _protocol_result(rows, res, rep)
    out["cross_corpus_leak_rate"] = rep["splits"].get("test", {}).get("leak_rate")
    return out


def stage_arch(per_class: int, presets: list[str], epochs: int, workers: int = 2) -> dict:
    """Architecture search under protocol C, ranked on validation."""
    rows = _cross_rows(per_class)
    return train.sweep(rows, presets, epochs, RUNS, workers, tag="C_arch")


def stage_cal(winner: str, per_class: int, workers: int = 2) -> dict:
    rows = _cross_rows(per_class)
    man = Path("dataset/manifests/cross.csv")
    sources.write_manifest([{k: r[k] for k in sources.MANIFEST_COLS} for r in rows], man)
    ckpt = RUNS / f"C_arch_WINNER_{winner}" / "best.pt"
    if not ckpt.exists():
        ckpt = RUNS / "C_cross" / "best.pt"
    rep = evaluate.run(ckpt, man, out=ckpt.parent / "eval.json", workers=workers)
    return {"checkpoint": str(ckpt), "temperature": rep["temperature"],
            "test": rep.get("test")}


# --- driver ------------------------------------------------------------------

def cropped_manifest() -> bool:
    """Whether MANIFEST currently points at hand crops."""
    rows = sources.read_manifest(MANIFEST) if MANIFEST.exists() else []
    return bool(rows) and all(str(hands.CROP_ROOT) in r["path"] for r in rows)


def run(stage: str, per_class: int = 400, preset: str = "vit_small", epochs: int = 4,
        presets: list[str] | None = None, winner: str = "vit_small",
        workers: int = 2, crop: bool = False) -> dict:
    state = load()
    t0 = time.time()

    if stage == "data":
        out = stage_data(per_class, crop=crop)
    elif stage == "A":
        out = stage_A(per_class, preset, epochs, workers)
    elif stage == "B":
        out = stage_B(per_class, preset, epochs, workers)
    elif stage == "C":
        out = stage_C(per_class, preset, epochs, workers)
    elif stage == "arch":
        out = stage_arch(per_class, presets or M.SWEEPS["quick"], epochs, workers)
    elif stage == "cal":
        out = stage_cal(winner, per_class, workers)
    else:
        raise ValueError(f"unknown stage {stage}")

    out["seconds"] = round(time.time() - t0, 1)
    # Cropped and uncropped results are the same protocol on different input, so
    # they get their own slots rather than overwriting each other - the point of
    # running both is to put the two numbers side by side.
    key = f"{stage}_crop" if (crop or cropped_manifest()) else stage
    state["stages"][key] = out
    save(state)
    print(f"\n[{stage}] done in {out['seconds']}s")
    return out


def summary() -> str:
    """The three protocols side by side. This is the deliverable table."""
    s = load()["stages"]
    lines = ["", f"{'protocol':<34}{'test acc':>10}{'macro F1':>10}{'leakage':>10}", "-" * 64]
    names = {"A": "A  random split (as published)",
             "B": "B  group split (sessions whole)",
             "C": "C  cross-corpus (new signer)",
             "A_crop": "A  random split, hand-cropped",
             "B_crop": "B  group split, hand-cropped",
             "C_crop": "C  cross-corpus, hand-cropped"}
    for k, label in names.items():
        r = s.get(k)
        if not r:
            continue
        lines.append(f"{label:<34}{r['test_acc']:>10.4f}{r['test_macro_f1']:>10.4f}"
                     f"{r['test_leak_rate']:>10.1%}")
    if "A" in s and "C" in s:
        d = s["A"]["test_acc"] - s["C"]["test_acc"]
        lines += ["-" * 64,
                  f"A minus C = {d:.1%} of the published headline was leakage, not skill."]
    if "C" in s and "C_crop" in s:
        d = s["C_crop"]["test_acc"] - s["C"]["test_acc"]
        lines.append(f"cropping moved the honest number by {d:+.1%}.")
    out = "\n".join(lines)
    print(out)
    return out
