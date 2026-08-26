"""The checks that fail if the honesty guarantees break.

Everything here runs on synthetic images in a tmpdir - no dataset download, no
GPU, no network. Fast enough to run on every commit.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from slr import data, evaluate, hands, leakage, sources


# --- label + group recovery --------------------------------------------------

def test_normalise_label_covers_the_29_class_vocabulary():
    assert sources.normalise_label("A") == "A"
    assert sources.normalise_label("z") == "Z"
    assert sources.normalise_label("Space") == "space"
    assert sources.normalise_label("nothing") == "nothing"
    assert sources.normalise_label("Del") == "del"
    assert sources.normalise_label("random_folder") is None
    assert len(sources.CLASSES) == 29


def test_digits_signer_recovery_matches_the_capture_order():
    """IMG_1118..IMG_1127 is one student shooting digits 0..9; 1128 starts the next."""
    g = sources.group_sl_digits
    first = [g(Path(f"IMG_{n}.JPG"), Path(".")) for n in range(1118, 1128)]
    assert len(set(first)) == 1, "one student's ten shots must share a group"
    assert g(Path("IMG_1128.JPG"), Path(".")) != first[0], "next block is a new student"
    assert g(Path("IMG_1138.JPG"), Path(".")) != g(Path("IMG_1128.JPG"), Path("."))


# --- near-duplicate machinery ------------------------------------------------

def test_hamming_matches_python_bit_counting():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**63, size=64, dtype=np.uint64)
    b = rng.integers(0, 2**63, size=64, dtype=np.uint64)
    want = [bin(int(x) ^ int(y)).count("1") for x, y in zip(a, b)]
    assert list(leakage.hamming(a, b)) == want


@pytest.mark.parametrize("max_dist", [1, 3, 6])
def test_candidate_pairs_misses_no_true_near_duplicate(max_dist):
    """The pigeonhole guarantee is the whole reason we can skip the O(n^2) scan.

    If banding ever drops a real match, the audit silently under-reports leakage,
    which is the one failure mode this project cannot tolerate.
    """
    rng = random.Random(7)
    base = [rng.getrandbits(64) for _ in range(120)]
    # Plant near-duplicates by flipping exactly `max_dist` bits.
    planted = []
    for i in range(0, 40, 2):
        h = base[i]
        for bit in rng.sample(range(64), max_dist):
            h ^= 1 << bit
        planted.append((i, len(base)))
        base.append(h)
    h = np.array(base, dtype=np.uint64)

    truth = {
        (i, j)
        for i in range(len(h)) for j in range(i + 1, len(h))
        if leakage.hamming(h[i : i + 1], h[j : j + 1])[0] <= max_dist
    }
    found = {
        (min(i, j), max(i, j))
        for i, j in leakage.candidate_pairs(h, max_dist)
        if leakage.hamming(h[i : i + 1], h[j : j + 1])[0] <= max_dist
    }
    assert truth <= found, f"banding lost {sorted(truth - found)}"
    for a, b in planted:
        assert (min(a, b), max(a, b)) in found


# --- end to end on synthetic images -----------------------------------------

def _write_corpus(root: Path, source: str, n_groups: int, per_group: int) -> list[dict]:
    """Each group is one 'session': a base pattern plus imperceptible jitter, so
    within-group images are near-duplicates and across-group images are not."""
    rng = np.random.default_rng(1)
    rows = []
    for g in range(n_groups):
        base = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
        label = sources.CLASSES[g % len(sources.CLASSES)]
        d = root / source / label
        d.mkdir(parents=True, exist_ok=True)
        for k in range(per_group):
            arr = np.clip(base.astype(int) + rng.integers(-1, 2, base.shape), 0, 255)
            p = d / f"g{g}_{k}.png"
            Image.fromarray(arr.astype(np.uint8), "L").convert("RGB").save(p)
            rows.append({"path": str(p), "label": label,
                         "label_idx": sources.CLASS_TO_IDX[label],
                         "source": source, "group": f"{source}/g{g}", "split": ""})
    return rows


def test_group_split_never_lets_a_group_straddle(tmp_path):
    rows = _write_corpus(tmp_path, "synth", n_groups=87, per_group=6)
    data.split_by_group(rows, val=0.2, test=0.2, seed=3)
    seen = {}
    for r in rows:
        assert seen.setdefault(r["group"], r["split"]) == r["split"]
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_random_split_leaks_and_group_split_does_not(tmp_path):
    """The claim the whole project rests on, made falsifiable.

    Split the same corpus two ways. A random split scatters each session across
    train and test, so the audit should find near-duplicates for most test
    images. A group split keeps sessions whole, so it should find almost none.
    """
    rows = _write_corpus(tmp_path, "synth", n_groups=40, per_group=8)

    rng = random.Random(0)
    for r in rows:
        r["split"] = "test" if rng.random() < 0.25 else "train"
        r.pop("dhash", None)
    leaky = leakage.audit(rows, max_dist=6)["splits"]["test"]["leak_rate"]

    for r in rows:
        r["split"] = ""
    data.split_by_group(rows, val=0.0, test=0.25, seed=0)
    honest = leakage.audit(rows, max_dist=6)["splits"]["test"]["leak_rate"]

    assert leaky > 0.8, f"random split should be badly contaminated, got {leaky}"
    assert honest < 0.05, f"group split should be clean, got {honest}"


def test_recover_groups_finds_the_sessions(tmp_path):
    rows = _write_corpus(tmp_path, "synth", n_groups=12, per_group=5)
    truth = {r["path"]: r["group"] for r in rows}
    for r in rows:
        r["group"] = "synth/all"          # pretend the corpus shipped no ids
    leakage.recover_groups(rows, max_dist=6)

    by_recovered: dict[str, set[str]] = {}
    for r in rows:
        by_recovered.setdefault(r["group"], set()).add(truth[r["path"]])
    assert all(len(v) == 1 for v in by_recovered.values()), \
        "a recovered component merged two real sessions"
    assert len(by_recovered) == 12


def test_cross_source_split_refuses_overlapping_sources(tmp_path):
    rows = _write_corpus(tmp_path, "a", 4, 3) + _write_corpus(tmp_path, "b", 4, 3)
    with pytest.raises(ValueError):
        data.split_cross_source(rows, ["a"], ["a", "b"])
    kept = data.split_cross_source(rows, ["a"], ["b"], val=0.25)
    assert {r["source"] for r in kept if r["split"] == "test"} == {"b"}
    assert {r["source"] for r in kept if r["split"] in ("train", "val")} == {"a"}


def test_purge_drops_contaminated_eval_rows_only(tmp_path):
    rows = _write_corpus(tmp_path, "synth", 10, 6)
    rng = random.Random(1)
    for r in rows:
        r["split"] = "test" if rng.random() < 0.3 else "train"
    leakage.audit(rows, max_dist=6)
    kept = leakage.purge(rows)
    assert all(r["split"] == "train" for r in kept if "leak_dist" in r)
    assert sum(r["split"] == "train" for r in kept) == sum(r["split"] == "train" for r in rows)


# --- calibration / abstention ------------------------------------------------

def test_temperature_scaling_tames_an_overconfident_model():
    import torch

    torch.manual_seed(0)
    y = torch.randint(0, 5, (600,))
    logits = torch.randn(600, 5)
    logits[torch.arange(600), y] += 1.2       # right often, but not always
    logits *= 6.0                              # ... and wildly overconfident

    t = evaluate.fit_temperature(logits, y)
    assert t > 1.5, f"expected the fit to cool the logits, got T={t}"
    before = evaluate.expected_calibration_error(logits.softmax(1), y)
    after = evaluate.expected_calibration_error((logits / t).softmax(1), y)
    assert after < before


def test_risk_coverage_is_monotone():
    import torch

    torch.manual_seed(0)
    y = torch.randint(0, 5, (400,))
    logits = torch.randn(400, 5)
    logits[torch.arange(400), y] += 2.0
    rc = evaluate.risk_coverage(logits.softmax(1), y)
    # Answering fewer, more confident photos must not raise the error rate.
    errs = [r["error"] for r in rc]
    assert errs == sorted(errs, reverse=True), errs
    assert rc[0]["coverage"] == 1.0


# --- architecture search discipline ------------------------------------------

def test_every_preset_in_the_zoo_builds_and_reports_its_size():
    """A typo in a timm checkpoint name should fail here, not eight hours into a sweep."""
    from slr import model as M

    for name, p in M.PRESETS.items():
        net = M.build(name, len(sources.CLASSES), pretrained=False)
        got = M.n_params(net) / 1e6
        assert abs(got - p.approx_params_m) < 1.0, \
            f"{name}: PRESETS says {p.approx_params_m}M, actual {got:.1f}M"
        assert p.family in {"vit", "hybrid", "cnn"}
        assert p.tier in {"scale", "arch", "mobile"}


def test_named_sweeps_reference_real_presets():
    from slr import model as M

    for name, members in M.SWEEPS.items():
        assert members, f"sweep {name} is empty"
        unknown = set(members) - set(M.PRESETS)
        assert not unknown, f"sweep {name} names missing presets {unknown}"
    # The CNN control has to be in the headline sweeps, or the comparison is rigged.
    assert any(M.PRESETS[m].family == "cnn" for m in M.SWEEPS["quick"])
    assert any(M.PRESETS[m].family == "cnn" for m in M.SWEEPS["arch"])


def test_sweep_refuses_to_select_without_a_validation_split(tmp_path):
    """Selecting an architecture on test is leakage through the back door."""
    from slr import train

    rows = _write_corpus(tmp_path, "synth", n_groups=6, per_group=4)
    for r in rows:
        r["split"] = "train" if r["group"].endswith(("g0", "g1", "g2")) else "test"
    with pytest.raises(ValueError, match="validation"):
        train.sweep(rows, ["vit_tiny"], epochs=1, out_dir=tmp_path / "runs")


def test_sweep_resumes_from_a_partial_leaderboard(tmp_path, monkeypatch):
    """A full-zoo sweep outlives a Colab session, so a restart must not retrain
    candidates that already have a score."""
    import json

    from slr import train

    rows = _write_corpus(tmp_path, "synth", n_groups=9, per_group=4)
    for r in rows:
        r["split"] = ("train" if r["group"].endswith(("g0", "g1", "g2", "g3"))
                      else "val" if r["group"].endswith(("g4", "g5")) else "test")

    trained: list[str] = []

    def fake_train_one(_rows, preset, *a, **kw):
        trained.append(preset)
        score = {"vit_tiny": 0.9, "vit_small": 0.5}[preset]
        return {"preset": preset, "params_m": 1.0,
                "best": {"val_f1": score, "val_acc": score},
                "history": [{"secs": 1.0}], "test": {"acc": score}}

    monkeypatch.setattr(train, "train_one", fake_train_one)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "sweep_leaderboard.json").write_text(json.dumps({"leaderboard": [
        {"preset": "vit_tiny", "family": "vit", "tier": "scale", "params_m": 5.5,
         "img_size": 224, "val_f1": 0.9, "val_acc": 0.9, "secs_per_epoch": 1.0},
    ]}))

    res = train.sweep(rows, ["vit_tiny", "vit_small"], epochs=1, out_dir=runs)

    # vit_tiny appears once and only as the winner's final test run, never as a
    # search candidate: its score came off disk.
    assert trained == ["vit_small", "vit_tiny"]
    assert {b["preset"] for b in res["leaderboard"]} == {"vit_tiny", "vit_small"}
    assert res["winner"] == "vit_tiny"
    assert res["complete"] is True


# --- what group recovery can and cannot do -----------------------------------

def test_recovery_fragments_signers_when_each_shot_is_unique(tmp_path):
    """The documented limitation, pinned so nobody quietly forgets it.

    A corpus with one shot per person per class has no near-duplicates linking
    one person's images, so component recovery cannot see the person. It
    fragments them instead. Measured on the real sl_digits corpus this happens to
    211 of 217 signers; here it is reproduced in miniature.

    If this test ever starts failing because recovery got smarter, that is good
    news - but the README's claims need rewriting to match.
    """
    rng = np.random.default_rng(2)
    rows = []
    for signer in range(8):
        for cls_i in range(5):                 # one shot per class per signer
            label = sources.CLASSES[cls_i]
            d = tmp_path / "oneshot" / label
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"s{signer}_{cls_i}.png"
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
            rows.append({"path": str(p), "label": label,
                         "label_idx": sources.CLASS_TO_IDX[label],
                         "source": "oneshot", "group": f"oneshot/s{signer}",
                         "split": ""})

    truth = {r["path"]: r["group"] for r in rows}
    for r in rows:
        r["group"] = "oneshot/all"
    leakage.recover_groups(rows, max_dist=6)

    spread: dict[str, set[str]] = {}
    for r in rows:
        spread.setdefault(truth[r["path"]], set()).add(r["group"])
    fragmented = sum(1 for v in spread.values() if len(v) > 1)
    assert fragmented == 8, (
        "recovery is expected to fragment every signer here; if it no longer "
        "does, update recover_groups' docstring and the README"
    )


def test_audit_only_sources_cannot_be_trained_on(tmp_path):
    """sl_digits carries label_idx=-1. Feeding it to a trainer must fail loudly."""
    rows = _write_corpus(tmp_path, "synth", n_groups=2, per_group=2)
    for r in rows:
        r["label_idx"] = -1
        r["source"] = "sl_digits"
    with pytest.raises(ValueError, match="audit-only"):
        data.SignDataset(rows)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "dataset" / "manifests" / "sl_digits.csv").exists(),
    reason="run `python -m slr.sources download sl_digits && python -m slr.sources "
           "manifest sl_digits --out dataset/manifests/sl_digits.csv` first",
)
def test_signer_rule_holds_on_the_real_digits_corpus():
    """Ground truth against real data: the frame-counter rule really does split
    2062 images into the ~218 students the paper describes."""
    man = Path(__file__).resolve().parents[2] / "dataset" / "manifests" / "sl_digits.csv"
    rows = sources.read_manifest(man)
    by_signer: dict[str, list[str]] = {}
    for r in rows:
        by_signer.setdefault(r["group"], []).append(r["label"])

    assert 210 <= len(by_signer) <= 225, f"expected ~218 signers, got {len(by_signer)}"
    # A signer shooting each digit once must not show the same digit twice. A few
    # do, where the photographer retook a shot and slipped the counter.
    dupes = sum(1 for v in by_signer.values() if len(v) != len(set(v)))
    assert dupes / len(by_signer) < 0.05, f"{dupes}/{len(by_signer)} signers have a repeated digit"
    complete = sum(1 for v in by_signer.values() if set(v) == set("0123456789"))
    assert complete > 0.7 * len(by_signer), f"only {complete}/{len(by_signer)} signers are complete"


def test_subsample_honours_the_cap_even_when_one_session_exceeds_it():
    """A recovered `asl_alphabet` session runs to thousands of frames, so a cap
    of a few hundred is only meetable by keeping part of one session. The part
    kept must stay contiguous, so it is still a single unbroken run that any
    split deals to exactly one side."""
    from slr import experiment

    rows = [{"label": "A", "group": "g0", "path": f"a{i}.jpg"} for i in range(5000)]
    rows += [{"label": "B", "group": "g1", "path": f"b{i}.jpg"} for i in range(120)]
    rows += [{"label": "B", "group": "g2", "path": f"c{i}.jpg"} for i in range(120)]

    kept = experiment.subsample(rows, per_class=200)
    by_label = {}
    for r in kept:
        by_label.setdefault(r["label"], []).append(r)

    assert len(by_label["A"]) == 200, "one oversized session must be cut to the cap"
    assert [r["path"] for r in by_label["A"]] == [f"a{i}.jpg" for i in range(200)], \
        "the kept frames must be a contiguous prefix, not a scatter"
    # B's two sessions are small enough to deal whole: 120 fits, 120+120 exceeds
    # 200, so the second is trimmed to 80 rather than dropped or taken entire.
    assert len(by_label["B"]) == 200
    assert len({r["group"] for r in by_label["B"]}) == 2


def test_group_split_survives_one_session_holding_most_of_the_corpus():
    """`recover_groups` on `asl_alphabet` collapses 87k frames into 191 sessions,
    a few of which hold nearly all of it. A first-fit walk gave one such session
    to test, overshot the 15% budget by an order of magnitude, and left train
    empty - which is what actually happened on Colab."""
    rows = [{"path": f"big{i}.jpg", "label": "A", "group": "huge", "split": ""}
            for i in range(9000)]
    for g in range(60):
        rows += [{"path": f"s{g}_{i}.jpg", "label": "B", "group": f"small{g}",
                  "split": ""} for i in range(20)]

    data.split_by_group(rows, val=0.15, test=0.15, seed=0)
    counts = Counter(r["split"] for r in rows)
    for s in ("train", "val", "test"):
        assert counts[s] > 0, f"{s} is empty: {dict(counts)}"
    assert counts["train"] >= counts["test"], \
        f"the giant session belongs in train, not test: {dict(counts)}"
    # the invariant the whole split exists to preserve
    for g in {r["group"] for r in rows}:
        assert len({r["split"] for r in rows if r["group"] == g}) == 1


def _row(path, label, group, source="a"):
    return {"path": path, "label": label, "label_idx": 0,
            "source": source, "group": group, "split": ""}


def test_group_split_keeps_every_class_in_train_and_does_not_invert():
    """The failure that produced ~4% accuracy: a giant session of ten letters
    was dealt to val (11k images) and train kept 262 images of the other 19."""
    missing = list("ABIJKLOS UX".replace(" ", ""))
    others = [c for c in sources.CLASSES if c not in missing]
    rows = []
    for lab in missing:
        rows += [_row(f"{lab}_{i}.jpg", lab, "huge") for i in range(1000)]
    for i, lab in enumerate(others):
        rows += [_row(f"{lab}_{j}.jpg", lab, f"small{i}") for j in range(14)]

    data.split_by_group(rows, val=0.15, test=0.15, seed=0)
    n = Counter(r["split"] for r in rows)
    assert n["train"] > n["val"], dict(n)
    assert n["train"] > n["test"], dict(n)
    train_labels = {r["label"] for r in rows if r["split"] == "train"}
    assert train_labels == {r["label"] for r in rows}


def test_cross_source_val_does_not_swallow_the_training_corpus():
    """First-fit val carving assigned the first shuffled group to val. When that
    group was a giant recovered session, train lost ten classes."""
    rows = []
    for lab in sources.CLASSES:
        rows += [_row(f"tr_{lab}_{i}.jpg", lab, "huge_train", "a") for i in range(400)]
        rows += [_row(f"tr_{lab}_s{i}.jpg", lab, f"small_{lab}", "a") for i in range(20)]
        rows += [_row(f"te_{lab}_{i}.jpg", lab, f"hold_{lab}", "b") for i in range(10)]

    kept = data.split_cross_source(rows, ["a"], ["b"], val=0.15, seed=0)
    train = [r for r in kept if r["split"] == "train"]
    val = [r for r in kept if r["split"] == "val"]
    test = [r for r in kept if r["split"] == "test"]
    assert {r["source"] for r in test} == {"b"}
    assert len(train) > len(val), (len(train), len(val))
    assert {r["label"] for r in train} == set(sources.CLASSES)


def test_random_stratified_split_covers_every_class_70_15_15():
    rows = []
    for lab in sources.CLASSES:
        rows += [_row(f"{lab}_{i}.jpg", lab, f"g{lab}_{i}") for i in range(40)]
    data.split_random_stratified(rows, val=0.15, test=0.15, seed=0)
    n = Counter(r["split"] for r in rows)
    assert n["train"] > n["val"] >= n["test"] or n["train"] > n["test"]
    for split in ("train", "val", "test"):
        assert {r["label"] for r in rows if r["split"] == split} == set(sources.CLASSES)
    # ~70% train
    assert 0.62 < n["train"] / len(rows) < 0.78
def test_gradient_accumulation_preserves_final_partial_optimizer_step():
    """Catch dropping or underweighting the last microbatches of an epoch."""
    from slr import train

    assert [train._accumulation_group_size(i, total=8, accumulate=3)
            for i in range(8)] == [3, 3, 3, 3, 3, 3, 2, 2]
    assert [train._should_optimizer_step(i, total=8, accumulate=3)
            for i in range(8)] == [False, False, True, False, False, True, False, True]


# --- hand cropping -----------------------------------------------------------

def test_hand_box_is_square_margined_and_inside_the_frame():
    """The geometry runs on every cropped image and needs no detector to check.

    A box that leaves the frame produces a PIL crop padded with black, which is
    a background cue of exactly the kind this module exists to delete.
    """
    w, h = 200, 150
    for pts in ([(90, 70), (110, 80)],          # centred, small
                [(2, 2), (30, 40)],             # jammed into the top-left
                [(190, 140), (199, 149)],       # jammed into the bottom-right
                [(0, 0), (199, 149)]):          # fills the whole frame
        l, t, r, b = hands._box(pts, w, h)
        assert 0 <= l < r <= w and 0 <= t < b <= h, f"{pts} -> box escapes the frame"
        assert (r - l) == (b - t), f"{pts} -> box is not square"

    # The margin is real: a 20px-wide landmark cloud must yield a wider box.
    l, t, r, b = hands._box([(90, 70), (110, 80)], w, h)
    assert r - l > 20 * (1 + hands.MARGIN) - 1


def test_cropped_paths_mirror_the_corpus_tree_without_collisions():
    """Two images with the same filename in different corpora must not collide."""
    out = Path("/crops")
    a = {"path": str(sources.DEFAULT_ROOT / "asl_alphabet" / "A" / "A1.jpg"),
         "source": "asl_alphabet"}
    b = {"path": str(sources.DEFAULT_ROOT / "asl_alphabet_v2" / "A" / "A1.jpg"),
         "source": "asl_alphabet_v2"}
    pa, pb = hands._out_path(a, out), hands._out_path(b, out)
    assert pa != pb
    assert pa == out / "asl_alphabet" / "A" / "A1.jpg"
    # Paths outside the dataset root (a one-off photo) still land somewhere sane.
    odd = {"path": "/tmp/somewhere/x.png", "source": "mine"}
    assert hands._out_path(odd, out) == out / "mine" / "x.jpg"


@pytest.mark.skipif(not hands.available(),
                    reason="pip install 'slr[vision]' to exercise the detector")
def test_crop_finds_a_hand_and_falls_back_without_one():
    """End to end on one real photo, plus the miss case that must not raise.

    The fallback matters more than the hit: `slr.api` calls this on every
    incoming frame, and a detector that raised on a blank wall would take the
    service down instead of returning a low-confidence letter.
    """
    blank, _ = hands.crop(Image.new("RGB", (200, 200), (128, 128, 128)))
    assert blank.size == (200, 200), "a miss must return the image untouched"
    assert hands.crop(Image.new("RGB", (200, 200), (128, 128, 128)))[1] is False

    real = sorted((sources.DEFAULT_ROOT / "asl_alphabet_test").rglob("*.jpg"))[:1]
    if not real:
        pytest.skip("no downloaded corpus to detect a real hand in")
    img, found = hands.crop(Image.open(real[0]))
    if found:
        assert img.size[0] <= Image.open(real[0]).size[0]
        assert img.size[0] == img.size[1]


# --- landmark classifier -----------------------------------------------------

def test_landmark_normalisation_erases_camera_distance_and_position():
    """The property the cross-corpus result rests on.

    A landmark model transfers because the features cannot encode how far away
    the signer stood or where in frame their hand was. If this ever stops
    holding, the 90% cross-corpus number stops meaning what it claims.
    """
    from slr import landmarks as L

    rng = np.random.RandomState(0)
    lm = rng.rand(6, 21, 3).astype("float32") * 200

    n = L.normalise(lm)
    assert np.allclose(n[:, L.WRIST], 0), "wrist must sit at the origin"
    assert np.allclose(np.linalg.norm(n[:, L.MIDDLE_MCP], axis=-1), 1), \
        "palm length must normalise to 1"

    moved = lm * 3.7 + np.array([50, -20, 5], dtype="float32")
    assert np.allclose(L.features(lm), L.features(moved), atol=1e-4), \
        "features must survive a scale change and a translation unchanged"

    assert L.features(lm).shape == (6, L.FEATURE_DIM)
    assert L.features(lm[0]).shape == (1, L.FEATURE_DIM)


def test_rotation_normalisation_actually_removes_rotation():
    """`rotate=True` is cited in the docstring as measurably worse than leaving
    orientation in. That comparison only means anything if the switch does what
    it says, and a sign error here silently turns it into a no-op that gets
    written up as evidence.
    """
    from slr import landmarks as L

    lm = np.random.RandomState(1).rand(5, 21, 3).astype("float32") * 100
    spin = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype="float32")

    r = L.normalise(lm, rotate=True)
    assert np.allclose(r[:, L.MIDDLE_MCP, 0], 0, atol=1e-5), "palm axis is not on +y"
    assert (r[:, L.MIDDLE_MCP, 1] > 0).all(), "palm axis points the wrong way"

    assert np.allclose(L.features(lm, rotate=True),
                       L.features(lm @ spin, rotate=True), atol=1e-4), \
        "rotate=True must be invariant to spinning the hand in-plane"
    assert not np.allclose(L.features(lm), L.features(lm @ spin), atol=1e-4), \
        "rotate=False must keep orientation - it is what separates G/H from D/U"


def test_no_landmarks_predicts_nothing():
    """`nothing` is answered by the detector's silence, not by the classifier.

    The classifier has no `nothing` class to predict - those frames hold no hand
    and so produce no features - so this fallback is the only thing standing
    between an empty room and a confidently wrong letter.
    """
    from slr import landmarks as L

    classes = ["A", "B", "C"]
    net = L.build(len(classes))
    mu, sd = np.zeros(L.FEATURE_DIM), np.ones(L.FEATURE_DIM)
    lm = np.random.RandomState(0).rand(4, 21, 3).astype("float32") * 100
    found = np.array([True, False, True, False])

    pred = L.predict(net, mu, sd, classes, lm, found)
    assert list(pred[~found]) == ["nothing", "nothing"]
    assert all(p in classes for p in pred[found]), "found hands get a real class"


def test_score_matches_hand_computed_f1():
    from slr import landmarks as L

    truth = np.array(["A", "A", "B", "B", "nothing"])
    pred = np.array(["A", "B", "B", "B", "nothing"])
    s = L.score(pred, truth)
    assert s["acc"] == pytest.approx(4 / 5)
    # A: tp=1 fp=0 fn=1 -> 2/3.  B: tp=2 fp=1 fn=0 -> 4/5.  nothing: 1.0
    assert s["per_class_f1"]["A"] == pytest.approx(2 / 3)
    assert s["per_class_f1"]["B"] == pytest.approx(4 / 5)
    assert s["macro_f1"] == pytest.approx((2 / 3 + 4 / 5 + 1.0) / 3)


@pytest.mark.skipif(not hands.available(),
                    reason="pip install 'slr[vision]' to exercise the detector")
def test_landmarks_are_pixels_inside_the_frame_and_agree_with_the_box():
    """`detect` is defined as `_box(landmarks(...))`; check they cannot drift.

    Landmarks are returned in pixels rather than MediaPipe's normalised
    coordinates precisely so a non-square photo does not squash the hand, so the
    bounds check here is also the check on that decision.
    """
    from slr import landmarks as L

    real = sorted((sources.DEFAULT_ROOT / "asl_alphabet_test").rglob("*.jpg"))[:1]
    if not real:
        pytest.skip("no downloaded corpus to detect a real hand in")
    arr = np.asarray(Image.open(real[0]).convert("RGB"))
    lm = hands.landmarks(arr)
    if lm is None:
        pytest.skip("detector found no hand in the sampled image")

    h, w = arr.shape[:2]
    assert lm.shape == (21, 3)
    assert (lm[:, 0] > -w).all() and (lm[:, 0] < 2 * w).all(), "x is not in pixels"
    assert (lm[:, 1] > -h).all() and (lm[:, 1] < 2 * h).all(), "y is not in pixels"

    box = hands.detect(arr)
    assert box is not None, "landmarks found but detect returned no box"
    assert box == hands._box([(x, y) for x, y, _ in lm], w, h)
    assert L.features(lm).shape == (1, L.FEATURE_DIM)


def test_left_hands_fold_onto_the_right_handed_frame():
    """A left hand must produce the features of the right hand it mirrors.

    This is the whole fix for left-handed signers, and it is invisible to the
    benchmark: asl_alphabet is 91% right hands and asl_alphabet_test 98%, so
    both corpora agree with each other and disagree with half the population.
    Twelve photos from a left-handed signer scored 2/12 before the fold and
    9/12 after.
    """
    from slr import landmarks as L

    lm = np.random.RandomState(4).rand(5, 21, 3).astype("float32") * 100
    mirrored = lm.copy()
    mirrored[:, :, 0] *= -1

    assert np.allclose(L.features(lm, "Right"), L.features(mirrored, "Left"),
                       atol=1e-5), "a left hand must fold onto the right it mirrors"
    assert not np.allclose(L.features(lm, "Right"), L.features(mirrored, "Right"),
                           atol=1e-5), "without the fold the two must differ"

    # Per-row handedness, which is how a whole cache is folded at once.
    side = np.array(["Right", "Left", "Right", "Left", "Right"])
    mixed = np.where(side[:, None, None] == "Left", mirrored, lm)
    assert np.allclose(L.features(mixed, side), L.features(lm, "Right"), atol=1e-5)


def test_unknown_handedness_is_treated_as_right():
    """Plain arrays and pre-fold caches must behave exactly as they used to."""
    from slr import landmarks as L

    lm = np.random.RandomState(5).rand(3, 21, 3).astype("float32") * 100
    assert np.allclose(L.features(lm), L.features(lm, "Right"))
    assert not L.canonical_side(lm).any()


@pytest.mark.skipif(not hands.available(),
                    reason="pip install 'slr[vision]' to exercise the detector")
def test_detector_reports_which_hand_it_found():
    from slr import landmarks as L

    real = sorted((sources.DEFAULT_ROOT / "asl_alphabet_test").rglob("*.jpg"))[:1]
    if not real:
        pytest.skip("no downloaded corpus")
    got = hands.detect_hand(np.asarray(Image.open(real[0]).convert("RGB")))
    if got is None:
        pytest.skip("no hand in the sampled image")
    lm, side = got
    assert side in ("Left", "Right")
    assert lm.shape == (21, 3)
    # The pair must survive a pickle: it rides through a process pool into the
    # cache, and an attribute that silently reset to "Right" there would mark
    # every hand right-handed and undo the fold without any error.
    import pickle
    assert pickle.loads(pickle.dumps(got))[1] == side
    assert hands.landmarks(np.asarray(Image.open(real[0]).convert("RGB"))).shape == (21, 3)


# --- the validation split cannot come from a single-session corpus -----------

def test_recover_groups_refuses_a_collapsed_grouping(tmp_path):
    """A continuous capture chains into one component; returning it is the bug.

    On the real asl_alphabet this produced a single component holding 98.5% of
    87,000 images and spanning all 29 classes. A "group split" of that puts
    almost everything on one side, and the fusion weight chosen on the remainder
    came back as the worst of its candidates. Collapsing must be refused, not
    returned.
    """
    base = Image.new("L", (64, 64), 128)
    rows = []
    for i in range(24):
        p = tmp_path / f"f{i:03d}.png"
        # A gradient that shifts by one level per frame: every frame is a near
        # duplicate of its neighbour, which is what a continuous take looks like.
        im = base.copy()
        im.putpixel((i % 8, i // 8), 255)
        im.save(p)
        rows.append({"path": str(p), "label": "A", "source": "take",
                     "group": f"declared/{i // 6}", "split": ""})

    leakage.recover_groups(rows, max_dist=6, workers=0)
    groups = {r["group"] for r in rows}
    assert groups == {f"declared/{i}" for i in range(4)}, (
        "recovery collapsed and must have kept the caller's declared groups, "
        f"got {sorted(groups)}")


def test_recover_groups_still_works_when_it_does_not_collapse(tmp_path):
    """The guard must not fire on a corpus with real session structure."""
    rows = []
    for session in range(4):
        for i in range(5):
            p = tmp_path / f"s{session}_{i}.png"
            # Each session gets its own stripe period. dhash compares adjacent
            # pixels, so sessions must differ in *structure*: flat images of
            # different brightness hash identically and would look like one take.
            im = Image.new("L", (64, 64))
            period = 2 + session * 3
            im.putdata([255 if (x // period) % 2 else 0
                        for y in range(64) for x in range(64)])
            im.putpixel((i, i), 128)          # a little within-session variation
            im.save(p)
            rows.append({"path": str(p), "label": "A", "source": "multi",
                         "group": "unset", "split": ""})
    leakage.recover_groups(rows, max_dist=6, workers=0)
    found = {r["group"] for r in rows}
    assert "unset" not in found, "recovery should have replaced the placeholder"
    assert len(found) > 1, "distinct sessions must not merge into one component"


def test_tuning_split_is_group_disjoint_and_keeps_both_sides_complete(tmp_path):
    """Choosing a number needs a slice of the *holdout*, not of the training corpus."""
    rows = []
    for lab in ("A", "B", "C"):
        for g in range(8):
            for i in range(2):
                rows.append({"path": f"{lab}-{g}-{i}.jpg", "label": lab,
                             "source": "holdout", "group": f"{lab}/g{g}",
                             "split": "", "dhash": 0})
    # Pre-set groups and a hash so recovery is a no-op we do not need files for.
    import slr.leakage as lk
    real = lk.recover_groups
    try:
        lk.recover_groups = lambda r, **kw: r          # groups already declared
        tune, report = data.carve_tuning_split(rows, frac=0.5, seed=0)
    finally:
        lk.recover_groups = real

    assert len(tune) + len(report) == len(rows)
    assert {r["group"] for r in tune}.isdisjoint({r["group"] for r in report})
    assert {r["label"] for r in tune} == {"A", "B", "C"}
    assert {r["label"] for r in report} == {"A", "B", "C"}
