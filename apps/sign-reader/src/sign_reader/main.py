"""FastAPI app: the JSON API and the static site.

If the checkpoint is missing the site still starts and `/api/health` says why.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.types import Receive, Scope, Send

from . import APP, captures, engine, motion

WEB = APP / "web"
MODEL_CARD = APP / "model_card.json"

BOOT: dict = {"ready": False, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        engine.load()
        BOOT.update(ready=True, error=None)
    except Exception as e:
        BOOT.update(ready=False, error=str(e))
        print(f"[boot] model unavailable: {e}")
    motion.load()
    yield


app = FastAPI(title="Sign Reader", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {**engine.health(), "ready": BOOT["ready"], "error": BOOT["error"],
            "motion": motion.status()}


@app.get("/api/model")
def model_card() -> dict:
    """Model card plus runtime state, for the 'How it works' panel."""
    card = json.loads(MODEL_CARD.read_text()) if MODEL_CARD.exists() else {}
    return {**card, "runtime": engine.health(), "classes": engine.classes()}


# Hand sharpness on the training and test images, measured the same way the
# tracker does it (256px JPEG q70 -> hand crop -> 224px). Re-measure if that changes.
REFERENCE_SHARPNESS = {
    "soft_below": 45.0,       # training corpus 25th percentile
    "typical": 84.0,          # asl_alphabet median
    "good_above": 105.0,      # holdout corpus median
}


@app.get("/api/capture-target")
def capture_target() -> dict:
    """Sharpness bands for the focus meter."""
    return REFERENCE_SHARPNESS


@app.post("/api/landmarks")
async def landmarks(
    image: UploadFile = File(..., description="one video frame, already square"),
) -> JSONResponse:
    """Hand landmarks and sharpness for one preview frame.

    `hand` is 0-1 coordinates for drawing. `xyz` and `side` are what the page
    records and later posts to `/api/motion`. No retry, to keep it fast.
    """
    from slr import hands

    if not hands.available():
        raise HTTPException(503, "hand tracking needs mediapipe and opencv")

    raw = await image.read()
    if not raw or len(raw) > engine.MAX_UPLOAD_BYTES:
        raise HTTPException(400, "empty or oversized frame")
    try:
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "not a decodable frame") from None

    got = hands.detect_hand(arr, retry=False)
    if got is None:
        return JSONResponse({"hand": None, "xyz": None, "side": None,
                             "sharpness": None})
    lm, side = got
    h, w = arr.shape[:2]

    # Laplacian variance depends on size, so measure at 224px like REFERENCE_SHARPNESS.
    import cv2

    x0, y0, x1, y1 = hands._box([(x, y) for x, y, _ in lm], w, h)
    crop = cv2.resize(arr[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

    return JSONResponse({
        "hand": [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y, _ in lm],
        # Pixel coordinates, which is what the clip model was trained on.
        "xyz": [[round(float(x), 2), round(float(y), 2), round(float(z), 2)]
                for x, y, z in lm],
        "side": side,
        "sharpness": round(sharp, 1),
    })


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(..., description="one photo of one hand making one sign"),
    top_k: int = Query(3, ge=1, le=10),
    explain: bool = Query(False, description="also return an attention heatmap"),
) -> JSONResponse:
    if not BOOT["ready"]:
        raise HTTPException(503, BOOT["error"] or "model not loaded")
    raw = await image.read()
    try:
        out = engine.predict(raw, top_k, explain)
    except engine.BadImage as e:
        raise HTTPException(415, str(e)) from e

    # A full disk or read-only mount loses the capture, not the prediction.
    try:
        out["capture_id"] = captures.save(raw)
    except OSError as e:
        out["capture_id"] = None
        print(f"[capture] not kept: {e}")
    return JSONResponse(out)


@app.post("/api/motion")
def read_motion(
    frames: list = Body(..., embed=True,
                        description="(N, 21, 3) landmarks, detected frames only"),
    sides: list | None = Body(None, embed=True,
                              description="per-frame 'Left'/'Right' from the detector"),
    top_k: int = Query(3, ge=1, le=10),
) -> dict:
    """Recorded landmarks in, one sign out (including J, Z and words)."""
    try:
        return motion.read(frames, sides, top_k)
    except motion.BadClip as e:
        raise HTTPException(400 if motion.status()["ok"] else 503, str(e)) from e


@app.post("/api/label")
def set_label(capture_id: str = Body(...), label: str = Body(...)) -> dict:
    """Save a captured photo under the given label."""
    try:
        path = captures.label(capture_id, label)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"path": str(path), **captures.counts()}


@app.get("/api/captures")
def capture_counts() -> dict:
    return captures.counts()


class RevalidatingStatics(StaticFiles):
    """Static files with `cache-control: no-cache`, so CSS and JS are never stale."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def tag(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", [])
                           if k.lower() != b"cache-control"]
                headers.append((b"cache-control", b"no-cache"))
                message["headers"] = headers
            await send(message)

        await super().__call__(scope, receive, tag)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html", headers={"cache-control": "no-cache"})


app.mount("/", RevalidatingStatics(directory=WEB), name="web")


def main() -> int:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
