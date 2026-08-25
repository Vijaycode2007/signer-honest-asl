"""Keeping the photos you take, and filing them under the label you type.

This model reads a stranger's hand at 60.6% (Protocol C in the model card).
The cheapest thing that moves that number for *your* hand is more of your
hand, so every photo taken here is banked, and the folder it lands in is
already the layout `slr.sources.scan` walks:

    dataset/raw/mine/<session>/<LABEL>/<id>.jpg

which puts the collection one `python -m slr.sources manifest mine` away from
a training manifest, with no import step in between.

The session directory is the part that matters. A session is one sitting: one
light, one shirt, one camera distance, and its frames are near-copies of each
other. `slr.sources` reads that directory as the group key, so whole sittings
land on one side of a split - which is what stops a fine-tune on this data
from reproducing the leakage this whole project exists to expose. Collect
across several days and change something each time, or the group key is
telling the truth about a split that has nothing to separate.

A photo is banked the moment it is read, unlabelled, and moves into its class
folder when you name it. That order is deliberate: the photo exists before you
have decided what it was, and a capture you never got round to labelling is
still sitting in `_inbox/` to be dragged into place later.

Collection is off unless there is somewhere to collect to - an explicit
`SLR_CAPTURES`, or a `dataset/raw` next to the source tree. A deployed
instance has neither, so a public copy of this service never quietly banks its
visitors' photographs.
"""

from __future__ import annotations

import io
import os
import secrets
import time
from pathlib import Path

from PIL import Image

from slr.sources import CLASSES, STAGING, normalise_label

REPO = Path(__file__).resolve().parents[2]
# Named by the scanner, which has to know to walk past it.
INBOX = STAGING


def root() -> Path | None:
    """Where captures go, or None when collection is switched off.

    Read per call rather than at import so a test (or a second sitting) can
    point it somewhere else without reloading the module.
    """
    env = os.environ.get("SLR_CAPTURES")
    if env:
        return Path(env)
    default = REPO / "dataset" / "raw" / "mine"
    return default if default.parent.exists() else None


def session() -> str:
    """The current sitting. One per capture session, not one per day.

    The date is a safe default and a poor one: two sittings on the same
    afternoon share a light and a shirt, and merging them into one group tells
    the splitter they are the same session when they are. Set `SLR_SESSION`
    when you start a genuinely new one.
    """
    return os.environ.get("SLR_SESSION") or time.strftime("s%Y%m%d")


def save(raw: bytes) -> str | None:
    """Bank one photo unlabelled. Returns the id that labels it later."""
    r = root()
    if r is None:
        return None
    # Re-encoded rather than written through: uploads arrive as PNG, HEIC or
    # whatever the file picker handed over, and a corpus of one format in one
    # colour space is one less thing for the training loader to be clever about.
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    cid = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    box = r / INBOX
    box.mkdir(parents=True, exist_ok=True)
    img.save(box / f"{cid}.jpg", quality=95)
    return cid


def label(cid: str, raw_label: str) -> Path:
    """Move a banked photo into `<session>/<LABEL>/`. Raises ValueError."""
    r = root()
    if r is None:
        raise ValueError("capture collection is off - set SLR_CAPTURES")

    lab = normalise_label(raw_label)
    if lab is None:
        raise ValueError(f"{raw_label!r} is not one of the {len(CLASSES)} signs")

    box = (r / INBOX).resolve()
    src = (r / INBOX / f"{cid}.jpg")
    # `cid` came off the wire, so it could be `../../../etc/passwd`. Resolve it
    # and check it did not climb out of the inbox before anything is moved.
    if src.resolve().parent != box or not src.is_file():
        raise ValueError(f"no capture {cid!r} waiting to be labelled")

    dest = r / session() / lab
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / src.name
    src.rename(out)
    return out


def counts() -> dict:
    """What has been collected so far, for the page to report back."""
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
