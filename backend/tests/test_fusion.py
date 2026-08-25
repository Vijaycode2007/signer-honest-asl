"""Behavioral checks for pixel/per-finger probability fusion."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from slr import fusion


def test_pixel_weight_controls_a_normalized_linear_probability_pool():
    """Catch reversed weights or a blend that stops being a distribution."""
    pixel = np.array([[0.8, 0.2]], dtype=np.float64)
    geometry = np.array([[0.1, 0.9]], dtype=np.float64)

    got = fusion.fuse_probabilities(pixel, geometry, pixel_weight=0.75)

    np.testing.assert_allclose(got, [[0.625, 0.375]])
    np.testing.assert_allclose(got.sum(axis=1), [1.0])


def test_probability_alignment_uses_class_names_not_column_positions():
    """Catch silently fusing A's pixel score with B's geometry score."""
    geometry = np.array([[0.7, 0.2, 0.1]], dtype=np.float64)

    got = fusion.align_probabilities(
        geometry,
        from_classes=["B", "nothing", "A"],
        to_classes=["A", "B", "nothing"],
    )

    np.testing.assert_allclose(got, [[0.1, 0.7, 0.2]])


def test_ratio_selection_prioritizes_c_but_caps_a_and_b_regressions():
    """Catch choosing the largest C score even when A regresses materially."""
    sweep = [
        {"pixel_weight": 1.0, "A": {"macro_f1": 0.90},
         "B": {"macro_f1": 0.80}, "C": {"macro_f1": 0.50}},
        {"pixel_weight": 0.7, "A": {"macro_f1": 0.895},
         "B": {"macro_f1": 0.795}, "C": {"macro_f1": 0.70}},
        {"pixel_weight": 0.5, "A": {"macro_f1": 0.87},
         "B": {"macro_f1": 0.80}, "C": {"macro_f1": 0.80}},
    ]

    winner = fusion.choose_ratio(sweep, max_ab_regression=0.01)

    assert winner["pixel_weight"] == 0.7
    assert winner["C"]["macro_f1"] == 0.70


def test_ratio_selection_requires_the_pixel_endpoint_as_baseline():
    """Catch reporting regressions relative to an arbitrary sweep row."""
    with pytest.raises(ValueError, match="pixel_weight=1.0"):
        fusion.choose_ratio([
            {"pixel_weight": 0.5, "A": {"macro_f1": 0.9},
             "B": {"macro_f1": 0.9}, "C": {"macro_f1": 0.9}},
        ])


def test_missing_hand_geometry_probability_is_exactly_nothing():
    """Catch letting an MLP classify an all-zero, detector-miss feature vector."""
    probs = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1]], dtype=np.float64)

    got = fusion.geometry_fallback(
        probs, found=np.array([True, False]), classes=["A", "B", "nothing"])

    np.testing.assert_allclose(got, [[0.8, 0.1, 0.1], [0.0, 0.0, 1.0]])


def test_metrics_use_macro_f1_over_truth_classes_not_all_model_classes():
    """Catch absent classes deflating a split such as Protocol B."""
    truth = np.array([0, 0, 1, 1])
    probs = np.array([
        [0.9, 0.1, 0.0],
        [0.2, 0.8, 0.0],
        [0.1, 0.9, 0.0],
        [0.1, 0.9, 0.0],
    ])

    got = fusion.classification_metrics(probs, truth)

    assert got["accuracy"] == pytest.approx(0.75)
    # class 0: 2/3; class 1: 4/5; absent class 2 must not enter the mean.
    assert got["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_cross_corpus_tuning_partition_keeps_groups_whole():
    """Catch a near-duplicate group leaking between ratio tuning and final test."""
    rows = [
        {"label": label, "group": f"{label}/g{g}", "path": f"{label}-{g}-{i}.jpg"}
        for label in ("A", "B") for g in range(4) for i in range(2)
    ]

    tune, final = fusion.partition_tuning_rows(rows, tune_fraction=0.25, seed=0)

    assert len(tune) + len(final) == len(rows)
    assert {r["group"] for r in tune}.isdisjoint({r["group"] for r in final})
    assert {r["label"] for r in tune} == {"A", "B"}
    assert {r["label"] for r in final} == {"A", "B"}


def test_per_finger_bundle_round_trip_and_detector_fallback(tmp_path):
    """Catch serving coordinates to a model trained on 35 geometry features."""
    from slr import landmarks

    classes = ["A", "B", "nothing"]
    net = landmarks.build(len(classes), in_dim=35, hidden=8)
    path = tmp_path / "geometry.pt"
    torch.save({
        "state_dict": net.state_dict(),
        "classes": classes,
        "mean": np.zeros(35),
        "std": np.ones(35),
        "hidden": 8,
        "feature": "per_finger_geometry_v1",
        "temperature": 1.0,
    }, path)

    bundle = fusion.load_geometry(path)
    lm = np.random.RandomState(7).rand(2, 21, 3).astype("float32") * 100
    got = fusion.geometry_probabilities(
        bundle, lm, found=np.array([True, False]), side=np.array(["Right", "Left"]))

    assert got.shape == (2, 3)
    np.testing.assert_allclose(got.sum(axis=1), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(got[1], [0.0, 0.0, 1.0])


def test_protocol_rows_reuse_the_same_source_pool_and_keep_b_groups_whole():
    """Catch changing the data pool while claiming only the split changed."""
    rows = [
        {"path": f"{label}-{group}-{i}.jpg", "label": label,
         "label_idx": 0 if label == "A" else 1, "source": "train",
         "group": f"{label}/g{group}", "split": "train"}
        for label in ("A", "B") for group in range(6) for i in range(2)
    ] + [
        {"path": f"holdout-{label}-{i}.jpg", "label": label,
         "label_idx": 0 if label == "A" else 1, "source": "holdout",
         "group": f"holdout/{label}/{i}", "split": "test"}
        for label in ("A", "B") for i in range(4)
    ]

    a = fusion.make_protocol_rows(rows, "A", "train", "holdout", seed=0)
    b = fusion.make_protocol_rows(rows, "B", "train", "holdout", seed=0)
    c = fusion.make_protocol_rows(rows, "C", "train", "holdout", seed=0)

    pool = {r["path"] for r in rows if r["source"] == "train"}
    assert {r["path"] for r in a} == pool
    assert {r["path"] for r in b} == pool
    assert {r["path"] for r in c if r["source"] == "train"} == pool
    assert {r["path"] for r in c if r["source"] == "holdout"} == {
        r["path"] for r in rows if r["source"] == "holdout"
    }
    for group in {r["group"] for r in b}:
        assert len({r["split"] for r in b if r["group"] == group}) == 1


def test_cached_landmarks_join_by_path_even_when_cache_order_differs():
    """Catch assigning one photo's hand geometry to another photo's pixel logits."""
    cache = {
        "path": np.array(["c.jpg", "a.jpg", "b.jpg"]),
        "lm": np.array([[[3.0]], [[1.0]], [[2.0]]]),
        "found": np.array([True, False, True]),
        "side": np.array(["Right", "Left", "Right"]),
    }
    rows = [{"path": "a.jpg"}, {"path": "b.jpg"}, {"path": "c.jpg"}]

    got = fusion.cached_landmarks_for_rows(cache, rows)

    np.testing.assert_allclose(got["lm"].reshape(-1), [1.0, 2.0, 3.0])
    assert got["found"].tolist() == [False, True, True]
    assert got["side"].tolist() == ["Left", "Right", "Right"]


def test_geometry_training_writes_a_servable_per_finger_bundle(tmp_path):
    """Catch saving a scaler or feature declaration that serving cannot replay."""
    rng = np.random.RandomState(11)
    rows = []
    for i in range(18):
        label = "A" if i % 2 == 0 else "B"
        split = "train" if i < 10 else ("val" if i < 14 else "test")
        rows.append({"path": f"{i}.jpg", "label": label,
                     "label_idx": 0 if label == "A" else 1,
                     "source": "train", "group": f"g{i}", "split": split})
    cache = {
        "path": np.array([r["path"] for r in reversed(rows)]),
        "lm": rng.rand(len(rows), 21, 3).astype("float32") * 100,
        "found": np.ones(len(rows), dtype=bool),
        "side": np.array(["Right"] * len(rows)),
    }
    out = tmp_path / "geometry.pt"

    report = fusion.train_geometry(rows, cache, out, epochs=1, seed=0, hidden=8)
    loaded = fusion.load_geometry(out)

    assert loaded.mean.shape == (35,)
    assert loaded.classes[0:2] == ["A", "B"]
    assert report["feature"] == "per_finger_geometry_v1"
    assert report["val"]["n"] == 4
    assert report["test"]["n"] == 4


def test_sweep_scores_both_endpoints_and_every_requested_ratio():
    """Catch omitting the standalone baselines needed to interpret a winner."""
    truth = np.array([0, 1])
    inputs = {
        protocol: {
            "pixel": np.array([[0.9, 0.1], [0.9, 0.1]]),
            "geometry": np.array([[0.6, 0.4], [0.1, 0.9]]),
            "truth": truth,
        }
        for protocol in ("A", "B", "C")
    }

    got = fusion.sweep_probabilities(inputs, ratios=[0.0, 0.5, 1.0])

    assert [row["pixel_weight"] for row in got] == [0.0, 0.5, 1.0]
    assert got[0]["C"]["accuracy"] == 1.0
    assert got[-1]["C"]["accuracy"] == 0.5
