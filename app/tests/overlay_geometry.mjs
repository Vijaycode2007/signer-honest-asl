/* Does the skeleton land on the hand?

   `toFrame` composes two mappings and both are easy to get subtly wrong: the
   posted frame is a centre crop of the video, and #video is painted with
   object-fit: cover, which scales by the LARGER ratio and centres the overflow.
   Get either backwards and the overlay sits beside the hand instead of on it -
   which looks like a broken detector rather than a broken transform.

   No test runner and no dependencies, because the app has neither:
       node app/tests/overlay_geometry.mjs
*/
import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const m = src.match(/function toFrame\(pts, box\) \{[\s\S]*?\n\}/);
if (!m) {
  console.error('could not find toFrame in app/web/app.js - was it renamed?');
  process.exit(1);
}

let els = { video: {} };
const toFrame = eval(`(${m[0].replace('function toFrame', 'function')})`);

let fails = 0;
const check = (name, got, want) => {
  const ok = Math.abs(got[0] - want[0]) < 1e-4 && Math.abs(got[1] - want[1]) < 1e-4;
  if (ok) console.log(`ok   ${name}`);
  else { fails++; console.log(`FAIL ${name}: got [${got}] want [${want}]`); }
};

// A landscape webcam in a square frame. Cover scales by height and lets the
// sides overflow, and the posted square is the middle of the sensor.
els.video = { videoWidth: 1280, videoHeight: 720 };
check('landscape: square centre lands on frame centre',
      toFrame([[0.5, 0.5]], { w: 600, h: 600 })[0], [300, 300]);
check('landscape: square corners span the frame',
      toFrame([[1, 1]], { w: 600, h: 600 })[0], [600, 600]);

// A portrait phone camera in a wide frame: cover now scales by width instead.
els.video = { videoWidth: 720, videoHeight: 1280 };
check('portrait: centre stays centred',
      toFrame([[0.5, 0.5]], { w: 800, h: 400 })[0], [400, 200]);

// Square video in a square frame is the degenerate case and must be identity.
els.video = { videoWidth: 480, videoHeight: 480 };
check('square: identity', toFrame([[0.25, 0.75]], { w: 480, h: 480 })[0], [120, 360]);
check('square: corner', toFrame([[1, 0]], { w: 480, h: 480 })[0], [480, 0]);

console.log(fails ? `\n${fails} FAILED` : '\nall coordinate mappings correct');
process.exit(fails ? 1 : 0);
