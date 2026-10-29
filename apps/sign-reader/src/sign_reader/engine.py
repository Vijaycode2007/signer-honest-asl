"""Load the DeiT III bundle and classify one photo.

The bundle from `train.py` holds the weights, class list, temperature and
abstain threshold. Transforms and the model come from `slr`.
"""

from __future__ import annotations

import base64
import io
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from slr import data as slr_data, landmarks as LM, model as M, safe_load

from . import APP

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_WEIGHTS = APP / "weights" / "deit3_base.pt"

_STATE: dict = {}


def resolve_checkpoint() -> Path:
    """Return a local checkpoint path, downloading `SLR_WEIGHTS_URL` once if needed."""
    configured = os.environ.get("SLR_WEIGHTS")
    fused_default = DEFAULT_WEIGHTS.with_name("deit3_base_fused.pt")
    local = Path(configured) if configured else (
        fused_default if fused_default.exists() else DEFAULT_WEIGHTS)
    if local.exists():
        return local

    url = os.environ.get("SLR_WEIGHTS_URL")
    if not url:
        raise RuntimeError(
            f"no checkpoint at {local} and SLR_WEIGHTS_URL is not set.\n"
            "Train one with `python -m sign_reader.train <manifest>`, "
            "then put it there or set SLR_WEIGHTS_URL."
        )

    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(".part")
    print(f"[engine] downloading weights from {url}")
    t0 = time.perf_counter()
    # Download then rename, so an interrupted boot never leaves a truncated cache.
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(local)
    print(f"[engine] weights cached at {local} "
          f"({local.stat().st_size / 1e6:.0f} MB in {time.perf_counter() - t0:.0f}s)")
    return local


def load() -> dict:
    """Build the network once and hold it. Returns the health payload."""
    ckpt = resolve_checkpoint()
    dev = M.device()
    blob = safe_load(ckpt)

    net = M.build(blob["preset"], len(blob["classes"]), pretrained=False)
    # Bundles ship fp16; torch's CPU half kernels are slower, so widen to fp32.
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in blob["state_dict"].items()}
    net.load_state_dict(state)
    net = net.to(dev).eval()

    temp = float(blob.get("temperature", 1.0))
    thresh = float(os.environ.get("SLR_THRESHOLD", blob.get("threshold", 0.0)))
    if "temperature" not in blob:
        print("[engine] no calibration in checkpoint; the service will never abstain")

    fused = None
    fusion_weight = None
    fusion_cal = None
    coverage = blob.get("coverage")
    if blob.get("fusion"):
        from slr import hands

        if not hands.available():
            raise RuntimeError("fused bundle needs mediapipe and opencv for geometry")
        meta = blob["fusion"]
        fusion_weight = float(meta["pixel_weight"])
        if not 0.0 <= fusion_weight <= 1.0:
            raise ValueError(f"fusion pixel_weight must be in [0,1], got {fusion_weight}")
        # The bundle records the pixel share; `fuse` multiplies the landmark share.
        fused = LM.model_from_blob(meta["geometry"],
                                   landmark_weight=1.0 - fusion_weight)
        if set(fused.classes) != set(blob["classes"]):
            raise ValueError("pixel and geometry class vocabularies differ")

        # The top-level threshold is for the pixel model; fused output has its own.
        fusion_cal = meta.get("calibration")
        if fusion_cal:
            temp_f = float(fusion_cal["temperature"])
            if not os.environ.get("SLR_THRESHOLD"):
                thresh = float(fusion_cal["threshold"])
            coverage = fusion_cal.get("measured_coverage", coverage)
            print(f"[engine] fused calibration: T={temp_f:.4f} "
                  f"threshold={thresh:.4f}, coverage "
                  f"{fusion_cal.get('measured_coverage')}")
        else:
            print("[engine] WARNING: fused bundle has no fused calibration; "
                  "falling back to the pixel model's threshold")

    _STATE.update(
        net=net, dev=dev, classes=blob["classes"], temp=temp, threshold=thresh,
        preset=blob["preset"], img_size=blob["img_size"],
        tf=slr_data.build_transforms(blob["img_size"], train=False),
        coverage=coverage, metrics=blob.get("metrics", {}),
        ckpt=str(ckpt), calibrated="temperature" in blob,
        geometry=fused, fusion_weight=fusion_weight, fusion_cal=fusion_cal,
    )
    print(f"[engine] {blob['preset']} on {dev}: {len(blob['classes'])} classes, "
          f"T={temp:.3f}, abstain below {thresh:.3f}")
    return health()


def health() -> dict:
    if not _STATE:
        return {"ok": False}
    return {
        "ok": True,
        "preset": _STATE["preset"],
        "device": str(_STATE["dev"]),
        "n_classes": len(_STATE["classes"]),
        "img_size": _STATE["img_size"],
        "temperature": round(_STATE["temp"], 4),
        "abstain_below": round(_STATE["threshold"], 4),
        "calibrated": _STATE["calibrated"],
        "coverage": _STATE["coverage"],
        "fusion": None if _STATE["geometry"] is None else {
            "pixel_weight": _STATE["fusion_weight"],
            "geometry_weight": 1.0 - _STATE["fusion_weight"],
            "calibration": _STATE["fusion_cal"],
        },
    }


def classes() -> list[str]:
    return list(_STATE.get("classes", []))


class BadImage(Exception):
    """The upload was not a decodable image, or was too large."""


def decode(raw: bytes) -> Image.Image:
    if not raw:
        raise BadImage("empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise BadImage(f"image exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    try:
        Image.open(io.BytesIO(raw)).verify()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise BadImage(f"not a decodable image: {e}") from e


def _png_b64(arr: np.ndarray, mode: str) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _heatmap_png_b64(mask: torch.Tensor, size: int) -> str:
    """Attention rollout as a translucent PNG the browser overlays on the crop."""
    m = mask.squeeze(0).cpu().numpy()
    m = (m * 255).astype("uint8")
    heat = Image.fromarray(m, "L").resize((size, size), Image.Resampling.BILINEAR)
    a = np.asarray(heat).astype("float32") / 255.0
    # Rollout is min-max normalised, so even a flat map spans 0..1. The gamma
    # keeps the weak tail transparent instead of tinting the entire photo.
    a = a ** 1.7
    rgba = np.zeros((size, size, 4), dtype="uint8")
    rgba[..., :3] = (214, 84, 45)  # the page's clay accent
    rgba[..., 3] = (a * 165).astype("uint8")
    return _png_b64(rgba, "RGBA")


def _input_png_b64(x: torch.Tensor) -> str:
    """The centre-cropped input as a PNG, so it lines up with the attention map."""
    mean = torch.tensor(slr_data.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(slr_data.IMAGENET_STD).view(3, 1, 1)
    img = (x[0].detach().cpu() * std + mean).clamp(0, 1)
    return _png_b64((img.permute(1, 2, 0).numpy() * 255).astype("uint8"), "RGB")


def predict(raw: bytes, top_k: int = 3, explain: bool = False) -> dict:
    if not _STATE:
        raise RuntimeError("model not loaded")
    t0 = time.perf_counter()
    img = decode(raw)
    x = _STATE["tf"](img).unsqueeze(0).to(_STATE["dev"])
    with torch.no_grad():
        pixel_probs = (_STATE["net"](x).float() / _STATE["temp"]).softmax(1)[0]

    probs = pixel_probs
    served_model = "pixel"
    if _STATE["geometry"] is not None:
        geometry_probs, _found, _side = LM.probs(
            _STATE["geometry"], np.asarray(img), _STATE["classes"])
        fused_probs = LM.fuse(pixel_probs.detach().cpu().numpy(), geometry_probs,
                              _STATE["geometry"].landmark_weight)
        probs = torch.from_numpy(fused_probs).float()
        cal = _STATE["fusion_cal"]
        if cal:
            # Changes the confidence, not the top label.
            logp = probs.clamp_min(float(cal.get("eps", 1e-6))).log()
            probs = (logp / float(cal["temperature"])).softmax(0)
        served_model = "fused"

    k = min(top_k, probs.numel())
    vals, idx = probs.topk(k)
    conf = float(vals[0])
    label = _STATE["classes"][int(idx[0])]

    heat = crop = None
    if explain:
        try:
            # Gradient-weighted, so the map explains this letter, not any class.
            heat = _heatmap_png_b64(
                M.grad_attention(_STATE["net"], x, int(idx[0])), _STATE["img_size"])
            crop = _input_png_b64(x)
        except TypeError:
            # Non-ViT backbones have no attention to roll up; answer without it.
            heat = crop = None

    return {
        "label": label,
        "confidence": round(conf, 4),
        "abstain": conf < _STATE["threshold"] or label in {"J", "Z"},
        "reason": (f"{label} needs movement. Use Record a moving sign."
                   if label in {"J", "Z"} else None),
        "threshold": round(_STATE["threshold"], 4),
        "top_k": [{"label": _STATE["classes"][int(i)], "p": round(float(v), 4)}
                  for v, i in zip(vals, idx)],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "attention_png_b64": heat,
        "input_png_b64": crop,
        "model": served_model,
    }
