"""Save the app's photos and file them under the label you type.

Photos go to `_inbox/` first, then move to

    dataset/raw/mine/<session>/<LABEL>/<id>.jpg

once labelled, which is the layout `slr.sources.scan` reads.

Off unless `SLR_CAPTURES` is set or `dataset/raw` exists, so a deployed copy
never keeps visitors' photos.
"""

from __future__ import annotations

import io
import os
import secrets
import time
from pathlib import Path

from PIL import Image

from slr.sources import CLASSES, REPO, STAGING, normalise_label

INBOX = STAGING


def root() -> Path | None:
    """Where captures go, or None when collection is switched off."""
    env = os.environ.get("SLR_CAPTURES")
    if env:
        return Path(env)
    default = REPO / "dataset" / "raw" / "mine"
    return default if default.parent.exists() else None


def session() -> str:
    """`SLR_SESSION`, or today's date."""
    return os.environ.get("SLR_SESSION") or time.strftime("s%Y%m%d")


def save(raw: bytes) -> str | None:
    """Save one unlabelled photo and return its id."""
    r = root()
    if r is None:
        return None
    # Re-encode so every capture is an RGB JPEG.
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    cid = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    box = r / INBOX
    box.mkdir(parents=True, exist_ok=True)
    img.save(box / f"{cid}.jpg", quality=95)
    return cid


def label(cid: str, raw_label: str) -> Path:
    """Move a saved photo into `<session>/<LABEL>/`. Raises ValueError."""
    r = root()
    if r is None:
        raise ValueError("capture collection is off - set SLR_CAPTURES")

    lab = normalise_label(raw_label)
    if lab is None:
        raise ValueError(f"{raw_label!r} is not one of the {len(CLASSES)} signs")

    box = (r / INBOX).resolve()
    src = r / INBOX / f"{cid}.jpg"
    # `cid` came off the wire: refuse anything that resolves outside the inbox.
    if src.resolve().parent != box or not src.is_file():
        raise ValueError(f"no capture {cid!r} waiting to be labelled")

    dest = r / session() / lab
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / src.name
    src.rename(out)
    return out


def counts() -> dict:
    """What has been collected so far."""
    r = root()
    if r is None:
        return {"enabled": False}

    per: dict[str, int] = {}
    for d in r.glob("*/*"):
        if d.is_dir():
            per[d.name] = per.get(d.name, 0) + len(list(d.glob("*.jpg")))
    inbox = r / INBOX
    return {
        "enabled": True,
        "root": str(r),
        "session": session(),
        "sessions": sorted(p.name for p in r.iterdir()
                           if p.is_dir() and p.name != INBOX) if r.exists() else [],
        "labelled": sum(per.values()),
        "unlabelled": len(list(inbox.glob("*.jpg"))) if inbox.exists() else 0,
        "per_label": dict(sorted(per.items())),
    }
