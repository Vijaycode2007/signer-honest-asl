"""The HTTP surface: three JSON endpoints and the static site in front of them.

One process serves both the API and the page that calls it, because that is the
shape Render (and every other one-service host) makes cheapest: no CORS, no
second deploy, no origin to keep in sync.

Booting never fails on a missing checkpoint. A crash-looping container tells
whoever deployed it nothing, so instead the site comes up, `/api/health`
reports why it is not ready, and the page says so in plain words.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send

from . import captures, engine

WEB = Path(__file__).resolve().parent.parent / "web"
MODEL_CARD = Path(__file__).resolve().parent.parent / "model_card.json"

BOOT: dict = {"ready": False, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        engine.load()
        BOOT.update(ready=True, error=None)
    except Exception as e:                      # noqa: BLE001 - reported, not swallowed
        BOOT.update(ready=False, error=str(e))
        print(f"[boot] model unavailable: {e}")
    yield


app = FastAPI(title="Sign Reader", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {**engine.health(), "ready": BOOT["ready"], "error": BOOT["error"]}


@app.get("/api/model")
def model_card() -> dict:
    """What the UI's 'How it works' panel renders.

    The measured evaluation figures are a checked-in file rather than something
    computed here: they come from split protocols this service never runs, and
    inventing them at request time would be exactly the sort of unsourced
    number this project exists to argue against.
    """
    card = json.loads(MODEL_CARD.read_text()) if MODEL_CARD.exists() else {}
    return {**card, "runtime": engine.health(), "classes": engine.classes()}


@app.post("/api/landmarks")
async def landmarks(
    image: UploadFile = File(..., description="one video frame, already square"),
) -> JSONResponse:
    """The 21 hand landmarks for one frame, for the live overlay.

    Deliberately separate from `/api/predict`: this one runs many times a second
    on a preview, touches no torch model, keeps nothing, and is allowed to fail
    quietly. Prediction runs once per photo, is calibrated, and files a capture.

    Coordinates come back normalised to the frame that was posted, so the page
    can scale them to whatever size it is drawing at without knowing what
    resolution the detector saw.

    The single-pass detector is used here rather than the retry: at preview rate
    the second look costs more than the frames it would rescue are worth, and a
    frame that misses is simply a frame with no skeleton on it.
    """
    try:
        from slr import hands
    except ImportError:                          # noqa: F401 - reported, not raised
        raise HTTPException(503, "hand tracking needs the slr package") from None
    if not hands.available():
        raise HTTPException(503, "hand tracking needs mediapipe and opencv")

    import numpy as np
    from PIL import Image, UnidentifiedImageError

    raw = await image.read()
    if not raw or len(raw) > engine.MAX_UPLOAD_BYTES:
        raise HTTPException(400, "empty or oversized frame")
    try:
        import io
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "not a decodable frame") from None

    lm = hands.landmarks(arr, retry=False)
    if lm is None:
        return JSONResponse({"hand": None})
    h, w = arr.shape[:2]
    return JSONResponse({"hand": [[round(float(x) / w, 5), round(float(y) / h, 5)]
                                  for x, y, _ in lm]})


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

    # Banking the photo is not allowed to cost anyone their answer: a full disk
    # or a read-only mount loses the training example, not the prediction.
    try:
        out["capture_id"] = captures.save(raw)
    except OSError as e:                        # noqa: BLE001 - reported, not swallowed
        out["capture_id"] = None
        print(f"[capture] not kept: {e}")
    return JSONResponse(out)


@app.post("/api/label")
def set_label(capture_id: str = Body(...), label: str = Body(...)) -> dict:
    """Name what a banked photo actually was, and file it under that class.

    The label is validated against the same 29-class vocabulary the model was
    trained on, by the same function that read the class names out of every
    other corpus. A typo is a 400 rather than a folder called `Q ` that quietly
    becomes a thirtieth class the next time a manifest is built.
    """
    try:
        path = captures.label(capture_id, label)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"path": str(path), **captures.counts()}


@app.get("/api/captures")
def capture_counts() -> dict:
    return captures.counts()


class RevalidatingStatics(StaticFiles):
    """Serve the page assets with `no-cache`.

    Not "do not cache" - the browser still keeps the file and still gets a cheap
    304 when nothing changed. What it stops is the browser using a stale copy
    *without asking*, which is what let an old stylesheet render new markup: the
    HTML came back fresh through the route below while the CSS was served from
    memory cache, and the page looked broken in a way no reload would explain.
    """

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
