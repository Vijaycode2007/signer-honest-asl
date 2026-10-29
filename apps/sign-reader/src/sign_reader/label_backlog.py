#!/usr/bin/env python
"""Label the `_inbox/` backlog by keyboard, one photo per keystroke.

    python -m sign_reader.label_backlog
    # then open http://localhost:8010

The app can only label the photo still on screen, so anything not named in the
moment waits in `_inbox/` with no label anywhere. This walks the inbox and files
each photo through `captures.label`. A letter labels and advances; `.` is
`nothing`, space is `space`, backspace is `del`, and `[` / `]` move without
labelling. Skip rather than guess: a wrong label is worse than none.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import captures

app = FastAPI(title="Backlog labeller")


def _inbox() -> Path:
    root = captures.root()
    if root is None:
        raise HTTPException(503, "capture collection is off - set SLR_CAPTURES")
    return root / captures.INBOX


@app.get("/api/queue")
def queue() -> dict:
    box = _inbox()
    ids = sorted(p.stem for p in box.glob("*.jpg")) if box.exists() else []
    return {"ids": ids, "session": captures.session(),
            "counts": captures.counts()}


@app.get("/api/photo/{cid}")
def photo(cid: str):
    box = _inbox()
    src = box / f"{cid}.jpg"
    # `cid` came off the wire: refuse anything that resolves outside the inbox.
    if src.resolve().parent != box.resolve() or not src.is_file():
        raise HTTPException(404, f"no capture {cid!r}")
    return FileResponse(src, media_type="image/jpeg")


class Naming(BaseModel):
    capture_id: str
    label: str


@app.post("/api/label")
def label(n: Naming) -> dict:
    try:
        out = captures.label(n.capture_id, n.label)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "path": str(out), "counts": captures.counts()}


PAGE = """
<title>Backlog labeller</title>
<style>
  :root { color-scheme: light dark; }
  body { margin:0; font:15px/1.5 ui-sans-serif,system-ui,sans-serif;
         display:flex; flex-direction:column; align-items:center; gap:14px;
         padding:24px; background:Canvas; color:CanvasText; }
  img { width:min(420px,80vw); aspect-ratio:1; object-fit:cover; border-radius:6px;
        border:1px solid rgba(128,128,128,.35); background:rgba(128,128,128,.12); }
  #bar { font-variant-numeric:tabular-nums; opacity:.75; }
  #hint { max-width:52ch; text-align:center; opacity:.7; font-size:13.5px; }
  #last { min-height:1.5em; font-variant-numeric:tabular-nums; }
  kbd { font:12px ui-monospace,monospace; border:1px solid rgba(128,128,128,.5);
        border-radius:3px; padding:1px 5px; }
  b.big { font-size:30px; }
</style>
<div id="bar">loading…</div>
<img id="shot" alt="the photo waiting to be labelled">
<div id="last"></div>
<div id="hint">
  <kbd>a</kbd>–<kbd>z</kbd> label and advance ·
  <kbd>.</kbd> nothing · <kbd>space</kbd> space · <kbd>⌫</kbd> del ·
  <kbd>]</kbd> skip · <kbd>[</kbd> back<br>
  Skip anything you are not sure of. A wrong label is worse than no label.
</div>
<script>
let ids = [], i = 0, done = 0;
const bar = document.getElementById('bar'), shot = document.getElementById('shot'),
      last = document.getElementById('last');

async function boot() {
  const q = await (await fetch('/api/queue')).json();
  ids = q.ids; i = 0; done = 0; show();
}
function show() {
  if (!ids.length) { bar.textContent = 'inbox empty - nothing left to label';
                     shot.removeAttribute('src'); return; }
  i = Math.max(0, Math.min(i, ids.length - 1));
  bar.textContent = `${i + 1} / ${ids.length} waiting · ${done} filed this session`;
  shot.src = `/api/photo/${ids[i]}?t=${Date.now()}`;
}
async function file(label) {
  const cid = ids[i];
  if (!cid) return;
  const r = await fetch('/api/label', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ capture_id: cid, label }) });
  if (!r.ok) { last.textContent = `refused: ${(await r.json()).detail}`; return; }
  ids.splice(i, 1); done++;
  last.innerHTML = `filed as <b class="big">${label}</b>`;
  show();
}
addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  if (k === ']') { i++; show(); return; }
  if (k === '[') { i--; show(); return; }
  if (k === '.') { e.preventDefault(); return file('nothing'); }
  if (k === ' ') { e.preventDefault(); return file('space'); }
  if (k === 'Backspace') { e.preventDefault(); return file('del'); }
  if (/^[a-zA-Z]$/.test(k)) { e.preventDefault(); return file(k.toUpperCase()); }
});
boot();
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


if __name__ == "__main__":
    import uvicorn

    box = captures.root()
    print(f"[label] inbox: {box / captures.INBOX if box else 'COLLECTION OFF'}")
    print(f"[label] filing into session {captures.session()!r} "
          f"(set SLR_SESSION to change it)")
    uvicorn.run(app, host="127.0.0.1", port=8010)
