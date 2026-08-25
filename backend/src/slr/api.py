"""HTTP inference service - the surface the React Native app talks to.

One meaningful endpoint, `POST /predict`, taking a photo and returning a letter,
a calibrated confidence, and an explicit `abstain` flag. The app should render
the abstain case as "hold steady, try again" rather than showing a letter it has
no business showing.

The temperature and the confidence threshold are read from the evaluation report
produced by `slr.evaluate`, so the threshold the service uses is the same one
whose error rate was measured, rather than a number somebody guessed.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from . import data, hands, landmarks as LM, model as M

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STATE: dict = {}


class Prediction(BaseModel):
    label: str
    confidence: float
    abstain: bool
    top_k: list[dict]
    latency_ms: float
    hand_detected: bool | None = None
    model: str = "pixel"
    landmark: dict | None = None
    attention_png_b64: str | None = None


def _load() -> None:
    """Read the checkpoint plus, if present, the calibration report beside it."""
    ckpt = Path(os.environ.get("SLR_CHECKPOINT", "runs/best.pt"))
    if not ckpt.exists():
        raise RuntimeError(
            f"no checkpoint at {ckpt}. Train one first, or point SLR_CHECKPOINT at it."
        )
    dev = M.device()
    blob = torch.load(ckpt, map_location=dev)
    net = M.build(blob["preset"], len(blob["classes"]), pretrained=False).to(dev).eval()
    net.load_state_dict(blob["state_dict"])

    temp, thresh = 1.0, float(os.environ.get("SLR_THRESHOLD", 0.0))
    report = Path(os.environ.get("SLR_EVAL_REPORT", ckpt.parent / "eval.json"))
    if report.exists():
        r = json.loads(report.read_text())
        temp = r.get("temperature", 1.0)
        if not os.environ.get("SLR_THRESHOLD"):
            # Default to the threshold measured at 90% coverage on the honest
            # test split: answer nine photos in ten, at a known error rate.
            for row in r.get("test", {}).get("risk_coverage", []):
                if abs(row["coverage"] - 0.9) < 0.03:
                    thresh = row["threshold"]
        print(f"[api] calibration from {report}: T={temp:.3f} threshold={thresh:.3f}")
    else:
        print(f"[api] no eval report at {report}; confidences are UNCALIBRATED")

    # Serve the way the model was trained. A crop-trained model fed a whole
    # room is a different problem than the one it was fitted to, and the gap
    # shows up as confident nonsense rather than as an error.
    crop = bool(blob.get("hand_crop"))
    if crop and not hands.available():
        print("[api] WARNING: checkpoint was trained on hand crops but mediapipe "
              "is not installed here. Serving uncropped photos to a crop-trained "
              "model - accuracy will be materially worse. pip install 'slr[vision]'")
        crop = False
    elif crop:
        hands.model_path()                 # fetch the landmarker before traffic
        print("[api] hand cropping enabled")

    # The landmark model is optional and tiny (368 KB), and it is loaded beside
    # the pixel model rather than instead of it so the same photo can be put to
    # both. On the cross-corpus test it scores 0.902 macro-F1 against the pixel
    # model's 0.568, which is a large enough gap that it should be easy to check
    # on real input rather than taken on trust.
    lm_path = Path(os.environ.get("SLR_LANDMARK", ckpt.parent.parent / "landmark.pt"))
    lm_model = None
    if lm_path.exists() and hands.available():
        lm_model = LM.load_model(lm_path)
        print(f"[api] landmark model from {lm_path}: {len(lm_model.classes)} classes, "
              f"fusing at w={lm_model.weight:.3f}")
    elif lm_path.exists():
        print(f"[api] landmark model at {lm_path} needs mediapipe; not loaded")

    STATE.update(lm=lm_model,
                 net=net, dev=dev, classes=blob["classes"], temp=temp,
                 threshold=thresh, preset=blob["preset"], crop=crop,
                 tf=data.build_transforms(blob["img_size"], train=False),
                 img_size=blob["img_size"], ckpt=str(ckpt))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    STATE.clear()


app = FastAPI(title="Sign Language Recognition API", version="0.1.0", lifespan=lifespan)
# ponytail: wide-open CORS. Fine while the only client is the RN app over a LAN;
# pin allow_origins to the deployed host before this faces the internet.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {
        "ok": bool(STATE),
        "preset": STATE.get("preset"),
        "checkpoint": STATE.get("ckpt"),
        "device": str(STATE.get("dev")),
        "n_classes": len(STATE.get("classes", [])),
        "temperature": STATE.get("temp"),
        "abstain_below": STATE.get("threshold"),
        "hand_crop": STATE.get("crop"),
        "landmark_model": STATE.get("lm") is not None,
        "fusion_weight": None if STATE.get("lm") is None else STATE["lm"].weight,
    }


def _decode(raw: bytes) -> Image.Image:
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()                       # cheap structural check
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(415, f"not a decodable image: {e}") from e


def _heatmap_png(mask: torch.Tensor, base: Image.Image) -> str:
    """Overlay the attention rollout on the photo and return it base64-encoded."""
    import numpy as np

    m = (mask.squeeze(0).cpu().numpy() * 255).astype("uint8")
    heat = Image.fromarray(m, "L").resize(base.size, Image.Resampling.BILINEAR)
    # Simple red-channel overlay; avoids pulling matplotlib in for a colourmap.
    arr = np.asarray(base.convert("RGB")).astype("float32")
    h = np.asarray(heat).astype("float32")[..., None] / 255.0
    arr = arr * (1 - 0.5 * h) + np.array([255.0, 40.0, 40.0]) * (0.5 * h)
    buf = io.BytesIO()
    Image.fromarray(arr.astype("uint8")).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/predict", response_model=Prediction)
async def predict(
    image: UploadFile = File(..., description="a photo of one hand making one sign"),
    top_k: int = Query(3, ge=1, le=10),
    explain: bool = Query(False, description="also return an attention heatmap"),
    model: str = Query("fused", pattern="^(fused|pixel|landmark|both)$",
                       description="'fused' (default) blends the two at the "
                                   "measured weight; 'pixel' and 'landmark' "
                                   "isolate one; 'both' answers with pixel and "
                                   "attaches the landmark result for comparison"),
) -> Prediction:
    if not STATE:
        raise HTTPException(503, "model not loaded")
    t0 = time.perf_counter()
    img = _decode(await image.read())

    if model != "pixel" and STATE["lm"] is None:
        raise HTTPException(
            503, "no landmark model loaded. Train one with `python -m slr.landmarks "
                 "train <cache>` and point SLR_LANDMARK at it, or install mediapipe.")

    lm_out = None
    if model in ("landmark", "both", "fused"):
        import numpy as np
        lm_out = LM.predict_one(STATE["lm"], np.asarray(img), top_k)
        if model == "landmark":
            return Prediction(
                label=lm_out["label"],
                confidence=lm_out["confidence"] or 0.0,
                # Uncalibrated, so there is no measured threshold to abstain on.
                # Saying "not sure" needs a number whose error rate is known, and
                # this model does not have one yet.
                abstain=False,
                top_k=lm_out["top_k"],
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                hand_detected=lm_out["hand_detected"],
                model="landmark",
            )

    found = None
    if STATE["crop"]:
        img, found = hands.crop(img)
        # Deliberately not an abstain. Detection misses roughly one honest photo
        # in seven on tightly framed input, and the training set carries the same
        # uncropped minority, so a miss is a degraded prediction rather than an
        # invalid one. The flag lets the app say "move your hand into frame"
        # without the API refusing to answer.

    x = STATE["tf"](img).unsqueeze(0).to(STATE["dev"])
    with torch.no_grad():
        logits = STATE["net"](x).float()
    probs = (logits / STATE["temp"]).softmax(1)[0]

    if model == "fused":
        # Both models vote on the same 29 slots, at the weight measured on a
        # leakage-free half of the holdout corpus. The blend is worth little
        # accuracy over landmarks alone (0.9034 -> 0.9080) and a great deal of
        # log-loss (0.6985 -> 0.4263), which is the half that matters here: the
        # abstention threshold is read off a confidence, so a better-calibrated
        # confidence is a better-placed refusal.
        import numpy as np
        lm_p, _, _ = LM.probs(STATE["lm"], np.asarray(img), STATE["classes"])
        blended = LM.fuse(probs.cpu().numpy(), lm_p, STATE["lm"].weight)
        probs = torch.tensor(blended, dtype=torch.float32)

    k = min(top_k, probs.numel())
    vals, idx = probs.topk(k)
    conf = float(vals[0])

    heat = None
    if explain:
        mask = M.attention_rollout(STATE["net"], x)
        heat = _heatmap_png(mask, img.resize((STATE["img_size"], STATE["img_size"])))

    return Prediction(
        label=STATE["classes"][int(idx[0])],
        confidence=round(conf, 4),
        abstain=conf < STATE["threshold"],
        top_k=[{"label": STATE["classes"][int(i)], "p": round(float(v), 4)}
               for v, i in zip(vals, idx)],
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        hand_detected=found if found is not None else (
            lm_out["hand_detected"] if lm_out else None),
        model=model if lm_out else "pixel",
        landmark=lm_out,
        attention_png_b64=heat,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="serve the recognition API")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args(argv)
    if a.checkpoint:
        os.environ["SLR_CHECKPOINT"] = a.checkpoint
    uvicorn.run(app, host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
