"""Loading the trained DeiT III checkpoint, and answering one photo with it.

Everything that touches the model lives here so the HTTP layer stays a thin
translation of JSON to tensors and back. The rest of the pipeline - the
backbone zoo, the eval transforms, attention rollout - is imported from the
research package (`slr`) rather than copied, so the thing served is the thing
that was measured.

The serving artefact is a single self-contained bundle written by `train.py`:
weights, class list, and the calibration constants that were fitted on the
validation split. Keeping the temperature and the abstention threshold *inside*
the checkpoint means the number the service abstains on is the same number
whose error rate was measured, rather than one somebody typed into an env var
later.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from slr import data as slr_data, landmarks as LM, model as M

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "deit3_base.pt"

_STATE: dict = {}


# --- getting the weights onto the box ----------------------------------------

def resolve_checkpoint() -> Path:
    """Return a local checkpoint path, downloading it once if configured.

    Render's filesystem is ephemeral and an 86M-parameter checkpoint is far too
    big for git, so the deploy path is: build the image without weights, and
    fetch them on first boot from `SLR_WEIGHTS_URL` (a release asset, an HF Hub
    file, a signed bucket URL). The download is cached on disk, so a restart
    inside the same container is free; a fresh container pays for it again.
    """
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
            "Train one with `python app/train.py <manifest>` (see app/README.md), "
            "then either place it at that path or host it and set SLR_WEIGHTS_URL."
        )

    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(".part")
    print(f"[engine] downloading weights from {url}")
    t0 = time.perf_counter()
    # Download to a temp name and rename, so an interrupted boot cannot leave a
    # truncated file that looks like a valid cache on the next start.
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
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)

    net = M.build(blob["preset"], len(blob["classes"]), pretrained=False)
    # Bundles ship fp16 to halve the download; CPU inference wants fp32 back
    # (torch's CPU kernels for half are patchy and slower, not faster).
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in blob["state_dict"].items()}
    net.load_state_dict(state)
    net = net.to(dev).eval()

    temp = float(blob.get("temperature", 1.0))
    thresh = float(os.environ.get("SLR_THRESHOLD", blob.get("threshold", 0.0)))
    if "temperature" not in blob:
        print("[engine] checkpoint carries no calibration; confidences are RAW "
              "and the service will never abstain")

    fused = None
    fusion_weight = None
    if blob.get("fusion"):
        from slr import hands

        if not hands.available():
            raise RuntimeError("fused bundle needs mediapipe and opencv for geometry")
        meta = blob["fusion"]
        fusion_weight = float(meta["pixel_weight"])
        if not 0.0 <= fusion_weight <= 1.0:
            raise ValueError(f"fusion pixel_weight must be in [0,1], got {fusion_weight}")
        # `weight` on the model is the geometry share; the bundle records the
        # pixel share, because that is the number the sweep reports against.
        fused = LM.model_from_blob(meta["geometry"], weight=1.0 - fusion_weight)
        if set(fused.classes) != set(blob["classes"]):
            raise ValueError("pixel and geometry class vocabularies differ")

    _STATE.update(
        net=net, dev=dev, classes=blob["classes"], temp=temp, threshold=thresh,
        preset=blob["preset"], img_size=blob["img_size"],
        tf=slr_data.build_transforms(blob["img_size"], train=False),
        coverage=blob.get("coverage"), metrics=blob.get("metrics", {}),
        ckpt=str(ckpt), calibrated="temperature" in blob,
        geometry=fused, fusion_weight=fusion_weight,
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
        },
    }


def classes() -> list[str]:
    return list(_STATE.get("classes", []))


# --- one photo in, one letter out --------------------------------------------

class BadImage(Exception):
    """The upload was not a decodable image, or was too large."""


def decode(raw: bytes) -> Image.Image:
    if not raw:
        raise BadImage("empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise BadImage(f"image exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    try:
        Image.open(io.BytesIO(raw)).verify()          # cheap structural check
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise BadImage(f"not a decodable image: {e}") from e


def _heatmap_png_b64(mask: torch.Tensor, size: int) -> str:
    """Attention rollout as a standalone translucent PNG the browser overlays.

    Returned as its own image rather than composited onto the photo, so the UI
    can fade it in and out over the frame the user already has on screen
    instead of swapping between two full-size pictures.
    """
    import base64

    import numpy as np

    m = mask.squeeze(0).cpu().numpy()
    m = (m * 255).astype("uint8")
    heat = Image.fromarray(m, "L").resize((size, size), Image.Resampling.BILINEAR)
    a = np.asarray(heat).astype("float32") / 255.0
    # Rollout is min-max normalised, so even a flat map spans 0..1. The gamma
    # keeps the weak tail transparent instead of tinting the entire photo.
    a = a ** 1.7
    rgba = np.zeros((size, size, 4), dtype="uint8")
    rgba[..., 0] = 214            # the canvas's warm clay, so the overlay reads
    rgba[..., 1] = 84             # as part of the UI rather than a jet colormap
    rgba[..., 2] = 45
    rgba[..., 3] = (a * 165).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _input_png_b64(x: torch.Tensor) -> str:
    """The exact tensor the model saw, turned back into a picture.

    The eval transform is Resize(255) then CenterCrop(224), so the model reads
    only the central 87.8% of the photo. Painting the attention map over the
    full frame displaced every feature by up to 6% of the frame. Returning the
    crop itself removes the question: the map and the picture are the same
    224x224 grid by construction.
    """
    import base64

    import numpy as np

    mean = torch.tensor(slr_data.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(slr_data.IMAGENET_STD).view(3, 1, 1)
    img = (x[0].detach().cpu() * std + mean).clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


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
        # Detection, the chirality fold, the class re-ordering and the
        # no-hand-means-`nothing` rule all live in `slr.landmarks.probs`, so the
        # served path and the measured path are the same code.
        geometry_probs, _found, _side = LM.probs(
            _STATE["geometry"], np.asarray(img), _STATE["classes"])
        fused_probs = LM.fuse(pixel_probs.detach().cpu().numpy(), geometry_probs,
                              _STATE["geometry"].weight)
        probs = torch.from_numpy(fused_probs).float()
        served_model = "fused"

    k = min(top_k, probs.numel())
    vals, idx = probs.topk(k)
    conf = float(vals[0])
    label = _STATE["classes"][int(idx[0])]

    heat = crop = None
    if explain:
        try:
            # Gradient-weighted, so the map explains THIS letter rather than
            # reporting where attention happens to pool regardless of class.
            heat = _heatmap_png_b64(
                M.grad_attention(_STATE["net"], x, int(idx[0])), _STATE["img_size"])
            crop = _input_png_b64(x)
        except TypeError:
            # Non-ViT backbones have no .blocks to roll up. The prediction is
            # still perfectly good, so degrade instead of failing the request.
            heat = crop = None

    return {
        "label": label,
        "confidence": round(conf, 4),
        "abstain": conf < _STATE["threshold"],
        "threshold": round(_STATE["threshold"], 4),
        "top_k": [{"label": _STATE["classes"][int(i)], "p": round(float(v), 4)}
                  for v, i in zip(vals, idx)],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "attention_png_b64": heat,
        "input_png_b64": crop,
        "model": served_model,
    }
