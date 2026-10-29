/* Sign Reader - camera, one POST per photo, message built from the letters.
   No framework and no build step: the model's process serves this file as-is. */

const $ = (id) => document.getElementById(id);

const els = {
  video: $('video'), shot: $('shot'), scratch: $('scratch'),
  overlay: $('overlay'), tracker: $('tracker'),
  trackChip: $('track-chip'), trackText: $('track-text'),
  focusChip: $('focus-chip'), focusFill: $('focus-fill'),
  focusMark: $('focus-mark'), focusText: $('focus-text'),
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
  record: $('record'), recordText: $('record-text'),
  uploadBtn: $('upload-btn'), file: $('file'),
  bootError: $('boot-error'), about: $('about'), aboutBody: $('about-body'),
};

// `message` is a list of what was read, so a word sign like HELLO stays one
// item instead of reading as fingerspelled H-E-L-L-O.
const state = { message: [], stream: null, last: null, busy: false, hasCamera: false,
                shownUrl: null, collecting: false, captureId: null, session: '',
                motion: null };

const QUESTION_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"
  stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18z"/>
  <path d="M9.4 9.3a2.7 2.7 0 0 1 5.2.9c0 1.8-2.6 2.3-2.6 2.3"/><path d="M12 16.6h.01"/></svg>`;

/* --- the message ---------------------------------------------------------- */

// `space`, `del` and `nothing` are classes in the 29-way vocabulary.
function applyLabel(label) {
  if (label === 'nothing') return null;
  if (label === 'space') { state.message.push(' '); return 'space'; }
  if (label === 'del') { state.message.pop(); return 'delete'; }
  state.message.push(label);
  return label.length > 1 ? 'word' : 'letter';
}

// How a label reads on screen, without appending it to the message.
function glyphFor(label) {
  if (label === 'nothing') return { text: 'no sign', word: true };
  if (label === 'space') return { text: 'space', word: true };
  if (label === 'del') return { text: 'delete', word: true };
  return { text: label, word: label.length > 1 };
}

// Word signs get spaces around them so they never run into adjacent letters.
const padWord = (t) => (t.length > 1 ? ` ${t} ` : t);

function messageText() {
  return state.message.map(padWord).join('').replace(/ {2,}/g, ' ');
}

function renderMessage() {
  if (!state.message.length) {
    els.message.innerHTML = '<span class="message-empty">Nothing spelled yet</span>';
    return;
  }
  const span = document.createElement('span');
  span.className = 'message-text';
  // Letters are spaced apart, as fingerspelling reads; word signs stay whole.
  span.textContent = state.message.map(padWord).join(' ').replace(/ {2,}/g, ' ');
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
    loadFocusBands();
    trackLoop();
  } catch {
    camOff('Camera blocked');
  }
}

function camOff(text) {
  state.hasCamera = false;
  track.stop = true;
  els.trackChip.hidden = true;
  els.focusChip.hidden = true;
  els.camDot.className = 'dot off';
  els.camText.textContent = text;
  els.video.hidden = true;
  els.frameHint.querySelector('.hint-sub').textContent =
    'No camera, so use "Upload instead" to pick a photo.';
}

// Send the centre square, so what the user framed is what the model sees.
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
    // Revoke the old frame only now: the evidence panel showed it until render.
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

/* --- reading a moving sign ------------------------------------------------

   J and Z need movement, so this records the landmarks the tracking loop
   already gets back from /api/landmarks and posts them to the clip model.    */

const SAMPLE_MS = 1000 / 12;   // the clip model was trained at 12 fps
const RECORD_MS = 3000;        // training clips are ~3.5s; 2s was noticeably worse

const rec = { on: false, frames: [], sides: [], last: 0, enabled: false };

function bufferFrame(d) {
  if (!rec.on || !d.xyz) return;
  const now = performance.now();
  // Keep to the sample rate the model was fitted at.
  if (now - rec.last < SAMPLE_MS) return;
  rec.last = now;
  rec.frames.push(d.xyz);
  rec.sides.push(d.side || 'Right');
}

async function recordSign() {
  if (state.busy || rec.on || !rec.enabled) return;
  if (!state.hasCamera) {
    return showError('Recording a movement needs the live camera. '
      + 'A still photo can be uploaded, but a clip cannot be reconstructed from one.');
  }
  state.busy = true;
  rec.on = true;
  rec.frames = [];
  rec.sides = [];
  rec.last = 0;
  els.record.classList.add('recording');
  els.shutter.disabled = true;
  els.shot.hidden = true;                 // watch your own hand while signing
  els.overlay.hidden = false;

  const t0 = performance.now();
  while (performance.now() - t0 < RECORD_MS) {
    const left = Math.ceil((RECORD_MS - (performance.now() - t0)) / 1000);
    els.recordText.textContent = `Signing... ${left}`;
    await new Promise((res) => setTimeout(res, 100));
  }
  rec.on = false;
  els.record.classList.remove('recording');
  els.recordText.textContent = 'Reading...';

  try {
    const r = await fetch('/api/motion?top_k=3', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ frames: rec.frames, sides: rec.sides }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || `${r.status}`);
    render(d);
  } catch (e) {
    showError(e.message);
  } finally {
    state.busy = false;
    els.shutter.disabled = false;
    els.recordText.textContent = 'Record a moving sign';
  }
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
  // The clip model is uncalibrated, so its score must not read as a probability.
  els.confText.textContent = d.calibrated === false
    ? `${pct}% raw score (uncalibrated)` : `${pct}% sure`;
  els.ms.textContent = d.frames
    ? `${d.frames} frames · ${Math.round(d.latency_ms)} ms`
    : `${Math.round(d.latency_ms)} ms`;
  els.meter.style.width = `${pct}%`;

  if (d.abstain) {
    // Show the guess, labelled as one, so the alternatives row is never
    // missing the top candidate.
    const g = glyphFor(d.label);
    els.glyph.className = `glyph unsure${g.word ? ' word' : ''}`;
    els.glyph.textContent = g.text;
    els.phrase.className = 'result-phrase unsure';
    els.phrase.textContent = `Not sure - best guess ${g.text}`;
    els.sub.textContent = d.reason ||
      'Below the confidence needed to add it, so your message is unchanged. '
      + 'Tap "Use it anyway" if the guess is right, or retake with your hand '
      + 'closer and better lit.';
    els.meter.className = 'meter-fill unsure';
    els.useAnyway.hidden = false;
  } else {
    els.useAnyway.hidden = true;
    const kind = applyLabel(d.label);
    renderMessage();
    const g = glyphFor(d.label);
    els.glyph.className = `glyph${g.word ? ' word' : ''}`;
    els.glyph.textContent = g.text;
    els.phrase.className = 'result-phrase';
    els.phrase.textContent = d.calibrated === false
      ? 'Read from the movement' : phraseFor(d.confidence);
    els.sub.textContent = {
      letter: 'Added to your message',
      word: 'A whole-word sign, added to your message',
      space: 'Added a space',
      delete: 'Removed the last letter',
    }[kind] || 'No hand sign in this photo, so nothing was added';
    els.meter.className = 'meter-fill';
  }

  // A recorded clip banks nothing, so it leaves the collection card alone.
  if (state.collecting && d.model !== 'clip_gru') {
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

// Overrule the threshold for one photo, without moving the threshold.
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
  // Prefer the server's 224px crop: it is the grid the attention map is on.
  els.evShot.src = d.input_png_b64
    ? `data:image/png;base64,${d.input_png_b64}`
    : photoUrl;
  els.evHeat.src = `data:image/png;base64,${d.attention_png_b64}`;
  els.evidence.hidden = false;
}

/* --- keeping the photo ---------------------------------------------------- */

/* Shown only when the server reports a capture directory, which a deployed
   copy does not have. */

function renderCollect(c) {
  if (!c || !c.enabled) return;
  state.collecting = true;
  state.session = c.session || '';
  els.collect.hidden = false;
  const kept = c.labelled || 0, waiting = c.unlabelled || 0;
  els.collectCount.textContent = (kept || waiting)
    ? `${kept} labelled${waiting ? ` \u00b7 ${waiting} waiting` : ''}`
    : 'nothing kept yet';
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

els.record.addEventListener('click', recordSign);
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
  state.message.pop();
  renderMessage();
});
els.clear.addEventListener('click', () => {
  state.message = [];
  renderMessage();
});
els.copy.addEventListener('click', async () => {
  if (!state.message.length) return;
  try {
    await navigator.clipboard.writeText(messageText());
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
    // Shown only with a clip checkpoint loaded. Not gated on the camera: the
    // permission prompt may still be open, so `recordSign` checks it instead.
    if (h.motion?.ok) {
      rec.enabled = true;
      els.record.hidden = false;
      els.record.title =
        `Records ${RECORD_MS / 1000}s and reads the movement, for J and Z. `
        + `Also knows `
        + `${(h.motion.classes || []).filter((c) => c.length > 1).join(', ')}.`;
      state.motion = h.motion;
    }
  } catch { /* health is a nicety; the predict call reports its own failures */ }

  try {
    renderAbout(await (await fetch('/api/model')).json());
  } catch { /* leave the dialog empty rather than block the app */ }

  try {
    renderCollect(await (await fetch('/api/captures')).json());
  } catch { /* no collection dir, no card - the app reads photos either way */ }
}

const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function motionBlock() {
  const mo = state.motion;
  if (!mo?.ok) return '';
  const words = (mo.classes || []).filter((c) => c.length > 1);
  const baseline = mo.frame_classifier_plurality;
  return `
    <div class="about-block">
      <p class="about-h">Reading a movement, not a pose</p>
      <p class="about-claim">J and Z are motion letters, and a photo cannot hold one.</p>
      <p class="proto-x">J is I with a hook and Z is D with a zigzag. The photo corpus
        labels a frozen mid-trace frame "Z", so the still model can only ever answer
        that pose. "Record a moving sign" watches ${RECORD_MS / 1000} seconds of hand
        movement instead and asks a separate recurrent model, which also knows
        ${words.length} whole-word signs: ${esc(words.join(', '))}.</p>
      ${baseline != null ? `<p class="proto-x">For comparison, the photo model voting
        frame by frame on the same clips gets ${Math.round(baseline * 100)}%.</p>` : ''}
      <p class="note warn">${esc(mo.caveat)}</p>
      <p class="note">Its confidence is a raw score, not a calibrated probability.
        Unlike the photo model, it has no refusal threshold.</p>
    </div>`;
}

function renderAbout(m) {
  const model = m.model || {};
  const rt = m.runtime || {};
  const pct = (x) => `${Math.round(x * 100)}%`;
  // "Running on mps" means nothing to someone holding a phone.
  const hardware = { cuda: 'a GPU', mps: 'an Apple GPU', cpu: 'a CPU' }[rt.device] || 'a CPU';

  const protos = (m.protocols || []).map((p) => {
    const cls = p.is_headline_number ? 'proto key' : (p.id === 'A' ? 'proto muted' : 'proto');
    // A and B were measured on a superseded pipeline; say so on the card.
    const tag = p.comparable_to_headline === false
      ? ' <em class="proto-tag">(older pipeline, not comparable)</em>' : '';
    return `<div class="${cls}">
      <div class="proto-n">${pct(p.test_acc)}</div>
      <div class="proto-b">
        <p class="proto-h">${esc(p.headline)}${tag}</p>
        <p class="proto-x">${esc(p.explain)}</p>
      </div></div>`;
  }).join('');

  const arch = m.architecture_search || {};
  const board = (arch.leaderboard || []).map((b) => {
    // Every candidate scores above 0.80, so scale bars over 0.80-0.96, not 0-1.
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
      <p class="about-claim">It depends a lot on who is signing.</p>
      ${protos}
      <p class="note">${esc(m.attribution)}</p>
    </div>

    <div class="about-block">
      <p class="about-h">Why it sometimes refuses</p>
      <p class="proto-x">${esc((m.abstention || {}).explain)} Right now that cut-off sits at
        <strong>${thresh}</strong>, and on held-out data it answered
        <strong>${pct((m.abstention || {}).measured_coverage || 0)}</strong> of photos and was right on
        <strong>${pct((m.abstention || {}).accuracy_among_accepted || 0)}</strong> of those.</p>
    </div>

    <div class="about-block">
      <p class="about-h">What's under the hood</p>
      <p class="about-claim" style="margin-bottom:4px">${esc(model.name)}</p>
      <p class="proto-x" style="margin-bottom:14px">A vision transformer, ${Math.round(model.params_m)} million
        parameters, reading a ${model.img_size}-pixel square. Running on ${hardware}.</p>
      ${board}
      <p class="note">${esc(arch.leaderboard_note)}</p>
      ${arch.valid_evidence === false
        ? `<p class="note warn">${esc(arch.explain)}</p>` : ''}
    </div>

    ${motionBlock()}

    <div class="about-block">
      <p class="about-h">The ${(m.classes || []).length} signs it knows</p>
      <div class="alpha">${letters.map((c) => `<span>${esc(c)}</span>`).join('')}</div>
      <div class="alpha-special">${special.map((c) => `<span>${esc(c)}</span>`).join('')}</div>
    </div>

    <ul class="limits">${(m.limits || []).map((l) => `<li>${esc(l)}</li>`).join('')}</ul>`;
}

boot();

/* --- live hand tracking ---------------------------------------------------

   The skeleton comes from the server's detector on the same centre square
   `grabSquare` sends, so it shows exactly what the model is given. Detection
   in the browser (WASM) is the upgrade if a round trip per frame gets slow.  */

const BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],               // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],               // index
  [5, 9], [9, 10], [10, 11], [11, 12],          // middle
  [9, 13], [13, 14], [14, 15], [15, 16],        // ring
  [13, 17], [17, 18], [18, 19], [19, 20],       // pinky
  [0, 17],                                      // palm edge
];
// Per-finger colour, by the landmark a bone ends at.
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

const track = { on: false, stop: false };

/* Focus: the server's hand sharpness, banded against the training corpora and
   smoothed with a 5-frame median because single frames are noisy. */
const focus = { band: null, recent: [] };

async function loadFocusBands() {
  try {
    focus.band = await (await fetch('/api/capture-target')).json();
    // The tick sits at the corpus median on a scale that runs to twice it.
    els.focusMark.style.left = `${(focus.band.typical / (focus.band.good_above * 1.6)) * 100}%`;
  } catch { /* the readout simply stays hidden */ }
}

function renderFocus(value) {
  if (value == null || !focus.band) { els.focusChip.hidden = true; focus.recent = []; return; }
  focus.recent.push(value);
  if (focus.recent.length > 5) focus.recent.shift();
  const sorted = [...focus.recent].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];

  const full = focus.band.good_above * 1.6;
  els.focusChip.hidden = false;
  els.focusFill.style.width = `${Math.min(100, (median / full) * 100)}%`;
  els.focusChip.classList.toggle('soft', median < focus.band.soft_below);
  els.focusChip.classList.toggle('ok', median >= focus.band.typical);
  const word = median < focus.band.soft_below ? 'soft'
             : median >= focus.band.good_above ? 'sharp'
             : median >= focus.band.typical ? 'good' : 'ok';
  els.focusText.textContent = `${Math.round(median)} ${word}`;
  els.focusChip.title =
    `Hand sharpness ${Math.round(median)}. Training corpus sits at ${focus.band.typical}, `
    + `the holdout corpus at ${focus.band.good_above}. Below ${focus.band.soft_below} is `
    + `softer than a quarter of the training data, and the pixel half of the model `
    + `contributes least there.`;
}

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
   Undo the centre crop, then apply object-fit: cover, which scales by the
   larger ratio and centres the overflow. */
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
    els.focusChip.hidden = true;
    return;
  }
  if (!r.ok) return;
  const d = await r.json();
  bufferFrame(d);
  trackChip(!!d.hand);
  renderFocus(d.sharpness);
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
      // A dropped frame is not worth a message.
      await new Promise((res) => setTimeout(res, 500));
    }
    const spent = performance.now() - t0;
    if (spent < TRACK_FLOOR_MS) {
      await new Promise((res) => setTimeout(res, TRACK_FLOOR_MS - spent));
    }
  }
  track.on = false;
}
