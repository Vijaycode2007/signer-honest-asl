/* Sign Reader - camera, one POST per photo, message built from the letters.
   No framework and no build step: the page is served as-is by the same process
   that runs the model, which is what makes this a one-service deploy. */

const $ = (id) => document.getElementById(id);

const els = {
  video: $('video'), shot: $('shot'), scratch: $('scratch'),
  overlay: $('overlay'), tracker: $('tracker'),
  trackChip: $('track-chip'), trackText: $('track-text'),
  useAnyway: $('use-anyway'),
  frameEmpty: $('frame-empty'), frameHint: $('frame-hint'),
  camDot: $('cam-dot'), camText: $('cam-text'),
  evidence: $('evidence'), evShot: $('ev-shot'), evHeat: $('ev-heat'),
  message: $('message'), undo: $('undo'), clear: $('clear'), copy: $('copy'),
  resultIdle: $('result-idle'), result: $('result'),
  glyph: $('glyph'), phrase: $('phrase'), sub: $('sub'), meter: $('meter'),
  confText: $('conf-text'), ms: $('ms'), alts: $('alts'), altsRow: $('alts-row'),
  collect: $('collect'), collectCount: $('collect-count'),
  collectNote: $('collect-note'), truth: $('truth'), saveLabel: $('save-label'),
  shutter: $('shutter'), shutterText: $('shutter-text'),
  uploadBtn: $('upload-btn'), file: $('file'),
  bootError: $('boot-error'), about: $('about'), aboutBody: $('about-body'),
};

const state = { message: '', stream: null, last: null, busy: false, hasCamera: false,
                shownUrl: null, collecting: false, captureId: null, session: '' };

const QUESTION_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"
  stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18z"/>
  <path d="M9.4 9.3a2.7 2.7 0 0 1 5.2.9c0 1.8-2.6 2.3-2.6 2.3"/><path d="M12 16.6h.01"/></svg>`;

/* --- the message ---------------------------------------------------------- */

// `space`, `del` and `nothing` are real classes in the 29-way vocabulary, and
// they are what makes photographing signs one at a time add up to a sentence.
function applyLabel(label) {
  if (label === 'nothing') return null;
  if (label === 'space') { state.message += ' '; return 'space'; }
  if (label === 'del') {
    state.message = state.message.slice(0, -1);
    return 'delete';
  }
  state.message += label;
  return 'letter';
}

// How a label reads on screen, with no side effect. `applyLabel` also appends to
// the message, so it cannot be used to show a guess the model is withholding.
function glyphFor(label) {
  if (label === 'nothing') return { text: 'no sign', word: true };
  if (label === 'space') return { text: 'space', word: true };
  if (label === 'del') return { text: 'delete', word: true };
  return { text: label, word: false };
}

function renderMessage() {
  if (!state.message) {
    els.message.innerHTML = '<span class="message-empty">Nothing spelled yet</span>';
    return;
  }
  const span = document.createElement('span');
  span.className = 'message-text';
  span.textContent = state.message.split('').join(' ');
  els.message.replaceChildren(span);
}

/* --- camera --------------------------------------------------------------- */

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) return camOff('No camera here');
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 1280 } },
      audio: false,
    });
    els.video.srcObject = state.stream;
    state.hasCamera = true;
    els.frameEmpty.hidden = true;
    els.camDot.className = 'dot live';
    els.camText.textContent = 'Camera on';
    trackLoop();
  } catch {
    camOff('Camera blocked');
  }
}

function camOff(text) {
  state.hasCamera = false;
  track.stop = true;
  els.trackChip.hidden = true;
  els.camDot.className = 'dot off';
  els.camText.textContent = text;
  els.video.hidden = true;
  els.frameHint.querySelector('.hint-sub').textContent =
    'No camera, so use "Upload instead" to pick a photo.';
}

// The model is trained on centre-cropped squares, so send it a square: cropping
// here rather than letting the server letterbox keeps what the user framed and
// what the model sees the same picture.
function grabSquare() {
  const v = els.video;
  const side = Math.min(v.videoWidth, v.videoHeight);
  if (!side) return null;
  const c = els.scratch;
  c.width = c.height = Math.min(side, 640);
  c.getContext('2d').drawImage(
    v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side,
    0, 0, c.width, c.height);
  return new Promise((res) => c.toBlob(res, 'image/jpeg', 0.92));
}

/* --- asking the model ----------------------------------------------------- */

async function send(blob, previewUrl) {
  if (state.busy) return;
  state.busy = true;
  els.shutter.disabled = true;
  els.shutterText.textContent = 'Reading...';

  const body = new FormData();
  body.append('image', blob, 'frame.jpg');

  try {
    const r = await fetch('/api/predict?top_k=3&explain=true', { method: 'POST', body });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `${r.status}`);
    const prev = state.shownUrl;
    showPhoto(previewUrl);
    render(data);
    goLive();
    state.shownUrl = previewUrl;
    // Only now is the old frame referenced by nothing: not the still, not the
    // evidence panel. Revoking it any earlier blanks the evidence mid-capture.
    if (prev && prev !== previewUrl) URL.revokeObjectURL(prev);
  } catch (e) {
    showError(e.message);
    if (previewUrl !== state.shownUrl) URL.revokeObjectURL(previewUrl);
  } finally {
    state.busy = false;
    els.shutter.disabled = false;
    els.shutterText.textContent = 'Take the next photo';
  }
}

function goLive() {
  if (!state.hasCamera) return;   // an upload has no live view to return to
  els.shot.hidden = true;
  els.overlay.hidden = false;
}

function showPhoto(url) {
  els.shot.src = url;
  els.shot.hidden = false;
  els.overlay.hidden = true;      // the bones described the live frame, not this one
  els.frameEmpty.hidden = true;
  els.frameHint.hidden = true;
}

function render(d) {
  state.last = d;
  showEvidence(d, els.shot.src);

  els.resultIdle.hidden = true;
  els.result.hidden = false;

  const pct = Math.round(d.confidence * 100);
  els.confText.textContent = `${pct}% sure`;
  els.ms.textContent = `${Math.round(d.latency_ms)} ms`;
  els.meter.style.width = `${pct}%`;

  if (d.abstain) {
    // Show the guess, and be plain that it is one. Withholding the letter
    // entirely was worse than useless: the "Could also be" row below still
    // listed the second and third guesses, so the reading was missing exactly
    // the one candidate the model rated highest. On a set of 22 photos the
    // hidden guess was the correct letter 3 times.
    const g = glyphFor(d.label);
    els.glyph.className = `glyph unsure${g.word ? ' word' : ''}`;
    els.glyph.textContent = g.text;
    els.phrase.className = 'result-phrase unsure';
    els.phrase.textContent = `Not sure - best guess ${g.text}`;
    els.sub.textContent =
      'Below the confidence needed to add it, so your message is unchanged. '
      + 'Tap "Use it anyway" if the guess is right, or retake with your hand '
      + 'closer and better lit.';
    els.meter.className = 'meter-fill unsure';
    els.useAnyway.hidden = false;
  } else {
    els.useAnyway.hidden = true;
    const kind = applyLabel(d.label);
    renderMessage();
    const shown = { space: 'space', delete: 'delete' }[kind];
    els.glyph.className = shown || d.label === 'nothing' ? 'glyph word' : 'glyph';
    els.glyph.textContent = d.label === 'nothing' ? 'no sign' : (shown || d.label);
    els.phrase.className = 'result-phrase';
    els.phrase.textContent = phraseFor(d.confidence);
    els.sub.textContent = {
      letter: 'Added to your message',
      space: 'Added a space',
      delete: 'Removed the last letter',
    }[kind] || 'No hand sign in this photo, so nothing was added';
    els.meter.className = 'meter-fill';
  }

  if (state.collecting) {
    state.captureId = d.capture_id || null;
    els.saveLabel.disabled = !state.captureId;
    if (state.captureId) idleNote();
    else collectNote('This photo was not kept - check the server log.', 'bad');
  }

  const alts = (d.top_k || []).slice(1);
  els.alts.hidden = alts.length === 0;
  els.altsRow.replaceChildren(...alts.map((a) => {
    const el = document.createElement('div');
    el.className = 'alt';
    el.innerHTML = `<span class="alt-letter"></span><span class="alt-p"></span>`;
    el.querySelector('.alt-letter').textContent = a.label;
    el.querySelector('.alt-p').textContent = `${Math.round(a.p * 100)}%`;
    return el;
  }));
}

// Accepting a withheld guess is the user's call, not a lower threshold. The
// threshold stays where its error rate was measured; this just lets a person who
// can see their own hand overrule it for one photo.
function useAnyway() {
  const d = state.last;
  if (!d || !d.abstain) return;
  applyLabel(d.label);
  renderMessage();
  els.useAnyway.hidden = true;
  els.sub.textContent = 'Added on your say-so, not the model\'s.';
}

function phraseFor(c) {
  if (c >= 0.86) return 'Pretty confident';
  if (c >= 0.72) return 'Fairly sure';
  return 'A bit unsure';
}

function showError(msg) {
  els.resultIdle.hidden = true;
  els.result.hidden = false;
  els.glyph.className = 'glyph unsure';
  els.glyph.innerHTML = QUESTION_SVG;
  els.phrase.className = 'result-phrase unsure';
  els.phrase.textContent = "That didn't work";
  els.sub.textContent = msg;
  els.meter.style.width = '0%';
  els.confText.textContent = '';
  els.ms.textContent = '';
  els.alts.hidden = true;
  els.evidence.hidden = true;
  state.captureId = null;
  els.saveLabel.disabled = true;
}

function showEvidence(d, photoUrl) {
  if (!d.attention_png_b64) {
    els.evidence.hidden = true;
    return;
  }
  // Prefer the server's own 224px crop: it is the exact grid the attention map
  // is computed on, so the two line up pixel for pixel. The raw capture is a
  // fallback and is very slightly wider than what the model actually read.
  els.evShot.src = d.input_png_b64
    ? `data:image/png;base64,${d.input_png_b64}`
    : photoUrl;
  els.evHeat.src = `data:image/png;base64,${d.attention_png_b64}`;
  els.evidence.hidden = false;
}

/* --- keeping the photo ---------------------------------------------------- */

/* Collecting is a local errand. The server only banks photos when it has a
   dataset directory to bank them in, so the page asks once at boot and, if the
   answer is no, never shows any of this - a deployed copy of the app has no
   such directory and offers nobody's visitors a way to fill it. */

function renderCollect(c) {
  if (!c || !c.enabled) return;
  state.collecting = true;
  state.session = c.session || '';
  els.collect.hidden = false;
  const kept = c.labelled || 0, waiting = c.unlabelled || 0;
  els.collectCount.textContent = (kept || waiting)
    ? `${kept} labelled${waiting ? ` \u00b7 ${waiting} waiting` : ''}`
    : 'nothing kept yet';
  // Callers that have something more specific to say set their own note after
  // this; on a cold page this is the only thing explaining what the card is.
  collectNote(`Every photo you take is kept. Name the sign it really was and it `
    + `lands in that class folder, ready to fine-tune on. Session ${state.session}.`);
}

function collectNote(text, cls = '') {
  els.collectNote.className = `note collect-note ${cls}`.trim();
  els.collectNote.textContent = text;
}

function idleNote() {
  collectNote(`Type the sign you actually made, not the one it guessed. The box `
    + `keeps its value, so a run of one letter is the shutter and Enter. `
    + `Filing under session ${state.session}.`);
}

async function saveLabel() {
  const label = els.truth.value.trim();
  if (!label) return collectNote('Type the sign first: a letter, or space, del, nothing.', 'bad');
  if (!state.captureId) return;

  els.saveLabel.disabled = true;
  try {
    const r = await fetch('/api/label', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ capture_id: state.captureId, label }),
    });
    const c = await r.json();
    if (!r.ok) throw new Error(c.detail || `${r.status}`);
    state.captureId = null;              // one photo files exactly once
    renderCollect(c);
    collectNote(`Kept as ${label.toUpperCase()}. Take the next one.`, 'saved');
  } catch (e) {
    els.saveLabel.disabled = false;
    collectNote(e.message, 'bad');
  }
}

/* --- wiring --------------------------------------------------------------- */

els.shutter.addEventListener('click', async () => {
  if (state.busy) return;
  if (!state.hasCamera) return els.file.click();
  els.shot.hidden = true;
  const blob = await grabSquare();
  if (!blob) return showError('The camera has not produced a frame yet. Give it a moment.');
  const url = URL.createObjectURL(blob);
  showPhoto(url);                 // hold the captured frame while the model reads it
  send(blob, url);
});

els.useAnyway.addEventListener('click', useAnyway);
els.saveLabel.addEventListener('click', saveLabel);
els.truth.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); saveLabel(); }
});

els.uploadBtn.addEventListener('click', () => els.file.click());
els.file.addEventListener('change', () => {
  const f = els.file.files[0];
  if (f) send(f, URL.createObjectURL(f));
  els.file.value = '';
});

els.undo.addEventListener('click', () => {
  state.message = state.message.slice(0, -1);
  renderMessage();
});
els.clear.addEventListener('click', () => {
  state.message = '';
  renderMessage();
});
els.copy.addEventListener('click', async () => {
  if (!state.message) return;
  try {
    await navigator.clipboard.writeText(state.message);
    els.copy.textContent = 'Copied';
    setTimeout(() => { els.copy.textContent = 'Copy'; }, 1400);
  } catch { /* clipboard blocked; the text is on screen to read anyway */ }
});

$('open-about').addEventListener('click', () => els.about.showModal());
$('close-about').addEventListener('click', () => els.about.close());

/* --- boot ----------------------------------------------------------------- */

async function boot() {
  renderMessage();
  startCamera();

  try {
    const h = await (await fetch('/api/health')).json();
    if (!h.ready) {
      els.bootError.hidden = false;
      els.bootError.textContent = `The model is not loaded, so reading is switched off.\n\n${h.error || ''}`;
      els.shutter.disabled = true;
      els.uploadBtn.disabled = true;
    }
  } catch { /* health is a nicety; the predict call reports its own failures */ }

  try {
    renderAbout(await (await fetch('/api/model')).json());
  } catch { /* leave the dialog empty rather than block the app */ }

  try {
    renderCollect(await (await fetch('/api/captures')).json());
  } catch { /* no collection dir, no card - the app reads photos either way */ }
}

function renderAbout(m) {
  const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const model = m.model || {};
  const rt = m.runtime || {};
  const pct = (x) => `${Math.round(x * 100)}%`;
  // "Running on mps" means nothing to someone holding a phone.
  const hardware = { cuda: 'a GPU', mps: 'an Apple GPU', cpu: 'a CPU' }[rt.device] || 'a CPU';

  const protos = (m.protocols || []).map((p) => {
    const cls = p.is_headline_number ? 'proto key' : (p.id === 'A' ? 'proto muted' : 'proto');
    return `<div class="${cls}">
      <div class="proto-n">${pct(p.test_acc)}</div>
      <div class="proto-b">
        <p class="proto-h">${esc(p.headline)}</p>
        <p class="proto-x">${esc(p.explain)}</p>
      </div></div>`;
  }).join('');

  const board = (m.leaderboard || []).map((b) => {
    // Scale the bars across the range actually spanned, not 0-100: every
    // candidate scores above 0.80, so a zero-based bar would show six
    // near-identical stripes and hide the thing the chart exists to show.
    const w = Math.max(4, Math.round(((b.val_f1 - 0.80) / 0.16) * 100));
    return `<div class="board-row ${b.winner ? 'win' : ''}">
      <span class="board-name">${esc(b.name)}</span>
      <div class="board-bar"><div class="board-fill" style="width:${w}%"></div></div>
      <span class="board-score">${(b.val_f1 * 100).toFixed(1)}</span>
    </div>`;
  }).join('');

  const letters = (m.classes || []).filter((c) => c.length === 1);
  const special = (m.classes || []).filter((c) => c.length > 1);
  const thresh = rt.abstain_below ? pct(rt.abstain_below) : 'the measured cut-off';

  els.aboutBody.innerHTML = `
    <p class="about-lede">This app reads fingerspelling, one letter at a time.</p>
    <p class="about-sub">Photograph a hand shape and it guesses which of ${model.n_classes || 29} signs
      you made: the letters A to Z, plus space, delete, and nothing.</p>

    <div class="about-block">
      <p class="about-h">How often is it right?</p>
      <p class="about-claim">It depends entirely on who is signing - and most apps don't tell you that.</p>
      ${protos}
      <p class="note">${esc(m.attribution)}</p>
    </div>

    <div class="about-block">
      <p class="about-h">Why it sometimes refuses</p>
      <p class="proto-x">${esc(m.abstention)} Right now that cut-off sits at <strong>${thresh}</strong>.</p>
    </div>

    <div class="about-block">
      <p class="about-h">What's under the hood</p>
      <p class="about-claim" style="margin-bottom:4px">${esc(model.name)}</p>
      <p class="proto-x" style="margin-bottom:14px">A vision transformer, ${Math.round(model.params_m)} million
        parameters, reading a ${model.img_size}-pixel square. Running on ${hardware}.</p>
      ${board}
      <p class="note">${esc(m.leaderboard_note)}</p>
    </div>

    <div class="about-block">
      <p class="about-h">The ${(m.classes || []).length} signs it knows</p>
      <div class="alpha">${letters.map((c) => `<span>${esc(c)}</span>`).join('')}</div>
      <div class="alpha-special">${special.map((c) => `<span>${esc(c)}</span>`).join('')}</div>
    </div>

    <ul class="limits">${(m.limits || []).map((l) => `<li>${esc(l)}</li>`).join('')}</ul>`;
}

boot();

/* --- live hand tracking ---------------------------------------------------

   The skeleton is drawn from the same detector the classifier is fed, and from
   the same centre-square crop `grabSquare` sends, so the overlay is evidence
   rather than decoration: if a knuckle is not marked, that joint is not in the
   vector the model will be given.

   Detection runs on the server rather than in the browser. That costs a round
   trip per frame and caps the preview at whatever the box can do (~20/s on
   localhost), and it buys exact agreement with what `/api/predict` sees, no
   second copy of the model to keep in step, and no build step - which is the
   trade this app has made everywhere else. Moving it into WASM on the client is
   the upgrade when this is served over a real network.                       */

const BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],               // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],               // index
  [5, 9], [9, 10], [10, 11], [11, 12],          // middle
  [9, 13], [13, 14], [14, 15], [15, 16],        // ring
  [13, 17], [17, 18], [18, 19], [19, 20],       // pinky
  [0, 17],                                      // palm edge
];
// Per-finger colour, by the landmark a bone ends at. Reading which finger the
// detector thinks is where is the whole point of drawing this.
const FINGER_COLOUR = [
  '#F2F0EC',                                    // 0 wrist
  '#E8833F', '#E8833F', '#E8833F', '#E8833F',   // thumb
  '#3F8F5E', '#3F8F5E', '#3F8F5E', '#3F8F5E',   // index
  '#3B7EA1', '#3B7EA1', '#3B7EA1', '#3B7EA1',   // middle
  '#8A5FBF', '#8A5FBF', '#8A5FBF', '#8A5FBF',   // ring
  '#C2445E', '#C2445E', '#C2445E', '#C2445E',   // pinky
];
const TRACK_PX = 256;          // what the detector sees; it downsamples anyway
const TRACK_FLOOR_MS = 40;     // don't spin faster than the eye needs

const track = { on: false, inflight: false, pts: null, stop: false };

function trackerBlob() {
  const v = els.video;
  const side = Math.min(v.videoWidth, v.videoHeight);
  if (!side) return null;
  const c = els.tracker;
  c.width = c.height = TRACK_PX;
  c.getContext('2d').drawImage(
    v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side,
    0, 0, TRACK_PX, TRACK_PX);
  return new Promise((res) => c.toBlob(res, 'image/jpeg', 0.7));
}

/* Normalised coordinates in the posted square -> CSS pixels in the frame.

   Two mappings compose here and both matter. The square is a centre crop of the
   video, so it has to be put back where it came from; and #video is drawn with
   object-fit: cover, which scales by the LARGER ratio and centres the overflow.
   Get either wrong and the skeleton sits next to the hand instead of on it. */
function toFrame(pts, box) {
  const v = els.video;
  const side = Math.min(v.videoWidth, v.videoHeight);
  const cropX = (v.videoWidth - side) / 2, cropY = (v.videoHeight - side) / 2;
  const scale = Math.max(box.w / v.videoWidth, box.h / v.videoHeight);
  const offX = (box.w - v.videoWidth * scale) / 2;
  const offY = (box.h - v.videoHeight * scale) / 2;
  return pts.map(([nx, ny]) => [
    (cropX + nx * side) * scale + offX,
    (cropY + ny * side) * scale + offY,
  ]);
}

function drawSkeleton(pts) {
  const c = els.overlay;
  const box = { w: c.clientWidth, h: c.clientHeight };
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(box.w * dpr) || c.height !== Math.round(box.h * dpr)) {
    c.width = Math.round(box.w * dpr);
    c.height = Math.round(box.h * dpr);
  }
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, box.w, box.h);
  if (!pts) return;

  const p = toFrame(pts, box);
  const r = Math.max(2.5, Math.min(box.w, box.h) * 0.007);

  // Every bone twice: a dark casing first, so the skeleton stays readable over
  // a bright palm or a white wall, then the finger's own colour on top.
  g.lineCap = 'round';
  for (const pass of [{ w: r * 1.9, col: 'rgba(24,18,14,.55)' }, { w: r * 0.9, col: null }]) {
    for (const [a, b] of BONES) {
      g.beginPath();
      g.moveTo(p[a][0], p[a][1]);
      g.lineTo(p[b][0], p[b][1]);
      g.lineWidth = pass.w;
      g.strokeStyle = pass.col || FINGER_COLOUR[b];
      g.stroke();
    }
  }
  p.forEach(([x, y], i) => {
    g.beginPath();
    g.arc(x, y, i === 0 ? r * 1.5 : r, 0, Math.PI * 2);
    g.fillStyle = FINGER_COLOUR[i];
    g.strokeStyle = 'rgba(24,18,14,.55)';
    g.lineWidth = r * 0.5;
    g.fill();
    g.stroke();
  });
}

function trackChip(found) {
  els.trackChip.hidden = false;
  els.trackChip.classList.toggle('searching', !found);
  els.trackText.textContent = found ? 'Tracking hand' : 'Looking for a hand';
}

async function trackOnce() {
  const blob = await trackerBlob();
  if (!blob) return;
  const body = new FormData();
  body.append('image', blob, 'f.jpg');
  const r = await fetch('/api/landmarks', { method: 'POST', body });
  if (r.status === 503) {                 // no mediapipe on this box: stop asking
    track.stop = true;
    els.trackChip.hidden = true;
    return;
  }
  if (!r.ok) return;
  const d = await r.json();
  track.pts = d.hand;
  trackChip(!!d.hand);
  drawSkeleton(els.shot.hidden ? d.hand : null);
}

async function trackLoop() {
  if (track.on) return;
  track.on = true;
  while (!track.stop && state.hasCamera) {
    if (document.hidden || els.video.readyState < 2) {
      await new Promise((res) => setTimeout(res, 200));
      continue;
    }
    const t0 = performance.now();
    try {
      await trackOnce();
    } catch {
      // A dropped frame is not worth a message. A dead server is caught by the
      // predict path, which the user is actually waiting on.
      await new Promise((res) => setTimeout(res, 500));
    }
    const spent = performance.now() - t0;
    if (spent < TRACK_FLOOR_MS) {
      await new Promise((res) => setTimeout(res, TRACK_FLOOR_MS - spent));
    }
  }
  track.on = false;
}
