"""Checks for the serving path: label handling, calibration, and the API.

The model itself is not under test here - `backend/tests` covers the research
code. What this covers is the layer between a trained checkpoint and a browser,
because that is the layer where a wrong answer is silent: an abstention that
never fires, a threshold read off the wrong split, a class list that drifts out
of order against the weights.

Run with:  PYTHONPATH=backend/src:app pytest app/tests -q
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT / "app"))

from server import engine  # noqa: E402


def _tiny_bundle(tmp: Path, *, calibrated: bool = True) -> Path:
    """A real checkpoint of a deliberately tiny backbone, in bundle format."""
    from slr import model as M

    classes = ["A", "B", "space", "del", "nothing"]
    net = M.build("vit_tiny", len(classes), pretrained=False)
    blob = {
        "preset": "vit_tiny",
        "classes": classes,
        "img_size": 224,
        "state_dict": {k: (v.half() if v.is_floating_point() else v)
                       for k, v in net.state_dict().items()},
    }
    if calibrated:
        blob |= {"temperature": 1.5, "threshold": 0.9, "coverage": 0.9}
    p = tmp / "bundle.pt"
    torch.save(blob, p)
    return p


def _tiny_fused_bundle(tmp: Path) -> Path:
    """A real pixel bundle whose geometry head deterministically answers B."""
    from slr import landmarks, model as M

    classes = ["A", "B", "space", "del", "nothing"]
    pixel = M.build("vit_tiny", len(classes), pretrained=False)
    geometry = landmarks.build(len(classes), in_dim=35, hidden=8)
    with torch.no_grad():
        for p in geometry.parameters():
            p.zero_()
        geometry[-1].bias[1] = 20.0
    path = tmp / "fused.pt"
    torch.save({
        "preset": "vit_tiny", "classes": classes, "img_size": 224,
        "state_dict": pixel.state_dict(), "temperature": 1.0,
        "threshold": 0.8, "coverage": 0.9,
        "fusion": {
            "pixel_weight": 0.0,
            "geometry": {
                "state_dict": geometry.state_dict(), "classes": classes,
                "mean": np.zeros(35), "std": np.ones(35), "hidden": 8,
                "feature": "per_finger_geometry_v1", "temperature": 1.0,
            },
        },
    }, path)
    return path


def _jpeg(size: int = 256) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (200, 170, 140)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("SLR_WEIGHTS", str(_tiny_bundle(tmp_path)))
    monkeypatch.delenv("SLR_THRESHOLD", raising=False)
    engine._STATE.clear()
    engine.load()
    yield
    engine._STATE.clear()


def test_bundle_carries_its_own_calibration(loaded):
    h = engine.health()
    assert h["calibrated"] is True
    assert h["temperature"] == 1.5
    # The threshold must come from the checkpoint, not default to zero. A
    # service that silently never abstains is the failure this guards.
    assert h["abstain_below"] == 0.9


def test_fp16_weights_are_widened_for_inference(loaded):
    p = next(engine._STATE["net"].parameters())
    assert p.dtype is torch.float32


def test_predict_shape_and_abstention(loaded):
    out = engine.predict(_jpeg(), top_k=3)
    assert out["label"] in engine.classes()
    assert 0.0 <= out["confidence"] <= 1.0
    assert len(out["top_k"]) == 3
    assert out["top_k"] == sorted(out["top_k"], key=lambda r: -r["p"])
    # Five classes and random weights put confidence near 0.2, far under the
    # 0.9 threshold, so this must abstain.
    assert out["abstain"] is (out["confidence"] < 0.9)
    assert out["abstain"] is True


def test_fused_bundle_drives_the_served_answer_and_health(tmp_path, monkeypatch):
    """Catch loading fusion metadata but still returning pixel-only probabilities."""
    import numpy as np
    from slr import hands

    monkeypatch.setenv("SLR_WEIGHTS", str(_tiny_fused_bundle(tmp_path)))
    monkeypatch.delenv("SLR_THRESHOLD", raising=False)
    monkeypatch.setattr(
        hands, "detect_hand",
        lambda _arr: (np.random.RandomState(3).rand(21, 3).astype("float32") * 100,
                      "Right"),
    )
    engine._STATE.clear()
    engine.load()
    try:
        out = engine.predict(_jpeg(), top_k=3)
        assert engine.health()["fusion"] == {"pixel_weight": 0.0,
                                               "geometry_weight": 1.0}
        assert out["model"] == "fused"
        assert out["label"] == "B"
        assert out["top_k"][0]["label"] == "B"
    finally:
        engine._STATE.clear()


def test_uncalibrated_checkpoint_never_abstains(tmp_path, monkeypatch):
    monkeypatch.setenv("SLR_WEIGHTS", str(_tiny_bundle(tmp_path, calibrated=False)))
    monkeypatch.delenv("SLR_THRESHOLD", raising=False)
    engine._STATE.clear()
    engine.load()
    try:
        assert engine.health()["calibrated"] is False
        assert engine.predict(_jpeg())["abstain"] is False
    finally:
        engine._STATE.clear()


def test_bad_uploads_are_rejected_not_crashed(loaded):
    with pytest.raises(engine.BadImage):
        engine.decode(b"")
    with pytest.raises(engine.BadImage):
        engine.decode(b"definitely not a jpeg")
    with pytest.raises(engine.BadImage):
        engine.decode(b"\xff" * (engine.MAX_UPLOAD_BYTES + 1))


def test_missing_weights_explains_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("SLR_WEIGHTS", str(tmp_path / "nope.pt"))
    monkeypatch.delenv("SLR_WEIGHTS_URL", raising=False)
    with pytest.raises(RuntimeError, match="SLR_WEIGHTS_URL"):
        engine.resolve_checkpoint()


def test_local_default_prefers_the_selected_fused_bundle(tmp_path, monkeypatch):
    """The finished ratio must be the local pipeline default, not an opt-in."""
    pixel = tmp_path / "deit3_base.pt"
    fused = tmp_path / "deit3_base_fused.pt"
    pixel.touch()
    fused.touch()
    monkeypatch.setattr(engine, "DEFAULT_WEIGHTS", pixel)
    monkeypatch.delenv("SLR_WEIGHTS", raising=False)
    monkeypatch.delenv("SLR_WEIGHTS_URL", raising=False)

    assert engine.resolve_checkpoint() == fused


def test_threshold_is_read_off_the_named_split():
    """The operating point must never be picked by peeking at test."""
    sys.path.insert(0, str(ROOT / "app"))
    import train as trainer

    report = {
        "val":  {"risk_coverage": [{"coverage": 0.9, "threshold": 0.61, "error": 0.11},
                                   {"coverage": 0.5, "threshold": 0.88, "error": 0.03}]},
        "test": {"risk_coverage": [{"coverage": 0.9, "threshold": 0.42, "error": 0.30}]},
    }
    thresh, row = trainer.pick_threshold(report, "val", 0.9)
    assert thresh == 0.61 and row["coverage"] == 0.9
    # Nearest coverage wins, not the first row.
    assert trainer.pick_threshold(report, "val", 0.55)[0] == 0.88


def test_model_card_matches_the_measured_fused_pipeline():
    """The card is hand-maintained, so guard the figures it asserts."""
    card = json.loads((ROOT / "app" / "model_card.json").read_text())
    assert card["model"]["preset"] == "deit3_base"
    assert card["model"]["val_f1"] == 0.9456
    assert card["leaderboard"][0]["preset"] == "deit3_base"
    assert all(card["leaderboard"][i]["val_f1"] >= card["leaderboard"][i + 1]["val_f1"]
               for i in range(len(card["leaderboard"]) - 1))
    # Every quoted protocol figure must say which model produced it, because
    # the honest numbers come from ViT-Small and DeiT III's are validation only.
    assert all(p["measured_on"] for p in card["protocols"])
    cross = next(p for p in card["protocols"] if p["id"] == "C")
    assert cross["measured_on"] == "deit3_base + per_finger_geometry"
    assert cross["test_f1"] == 0.797
    assert "not been through" not in card["attribution"]


# --- collecting photos to fine-tune on ---------------------------------------

def test_captures_are_off_without_somewhere_to_put_them(tmp_path, monkeypatch):
    """A deployed copy must not quietly bank its visitors' photographs."""
    from server import captures

    monkeypatch.delenv("SLR_CAPTURES", raising=False)
    monkeypatch.setattr(captures, "REPO", tmp_path)      # no dataset/raw beside it
    assert captures.root() is None
    assert captures.save(_jpeg()) is None
    assert captures.counts() == {"enabled": False}


def test_labelling_files_a_capture_where_the_scanner_will_find_it(tmp_path, monkeypatch):
    """Save, label, and then read it back through the real manifest builder.

    This is the whole feature in one assertion: a photo taken in the browser
    has to come out of `slr.sources` as a trainable row whose group is the
    capture session, because that group key is what keeps one sitting's
    near-identical frames from straddling a train/test split.
    """
    from slr import sources
    from server import captures

    monkeypatch.setenv("SLR_CAPTURES", str(tmp_path / "mine"))
    monkeypatch.setenv("SLR_SESSION", "s20260823")

    cid = captures.save(_jpeg())
    assert captures.counts()["unlabelled"] == 1

    out = captures.label(cid, "a")                       # typed lowercase
    assert out.parent.name == "A" and out.parent.parent.name == "s20260823"

    c = captures.counts()
    assert c["labelled"] == 1 and c["unlabelled"] == 0 and c["per_label"] == {"A": 1}

    rows = sources.scan("mine", root=tmp_path)
    assert len(rows) == 1
    assert rows[0]["label"] == "A"
    assert rows[0]["label_idx"] == sources.CLASS_TO_IDX["A"]
    assert rows[0]["group"] == "mine/s20260823"


def test_a_bad_label_is_refused_rather_than_made_into_a_class(tmp_path, monkeypatch):
    from server import captures

    monkeypatch.setenv("SLR_CAPTURES", str(tmp_path / "mine"))
    cid = captures.save(_jpeg())

    with pytest.raises(ValueError, match="not one of"):
        captures.label(cid, "banana")
    # A typo must not cost the photo: it is still there to label correctly.
    assert captures.counts()["unlabelled"] == 1
    assert captures.label(cid, "SPACE").parent.name == "space"


def test_a_capture_id_cannot_climb_out_of_the_inbox(tmp_path, monkeypatch):
    """`capture_id` arrives over the wire, so it is a path traversal waiting to happen."""
    from server import captures

    monkeypatch.setenv("SLR_CAPTURES", str(tmp_path / "mine"))
    captures.save(_jpeg())                               # make the inbox exist
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(_jpeg())

    with pytest.raises(ValueError, match="no capture"):
        captures.label("../../secret", "A")
    assert secret.exists()
