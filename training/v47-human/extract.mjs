// V47 step 1: keyframes from a phone recording of the official Threes app (night colours, 588x1280).
// Crops y 150..1040 (preview + board), halves it to 294x445, and keeps one frame per still period:
// a frame is still when under 1% of its 6x6 blocks changed (luma > 20) since the previous frame;
// a run of >= MIN still frames (env MIN, default 2) emits its middle frame, unless it matches the last emitted keyframe.
//  node extract.mjs VIDEO OUTDIR [FPS=30] [START_S] [DURATION_S]
import { spawn } from 'node:child_process';
import { writeFileSync, openSync, writeSync, closeSync, mkdirSync } from 'node:fs';
const [video, out, FPS = '30', start, dur] = process.argv.slice(2);
export const W = 294, H = 445, MIN = +(process.env.MIN || 2);
mkdirSync(out, { recursive: true });
const args = ['-v', 'error', ...(start ? ['-ss', start] : []), ...(dur ? ['-t', dur] : []), '-i', video,
  '-vf', `fps=${FPS},crop=588:890:0:150,scale=${W}:${H}:flags=area`, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'];
const ff = spawn('ffmpeg', args, { stdio: ['ignore', 'pipe', 'inherit'] });
const FR = W * H * 3, BW = Math.floor(W / 6), BH = Math.floor(H / 6);
function blocks(f) {
  const b = new Float32Array(BW * BH);
  for (let by = 0; by < BH; by++) for (let bx = 0; bx < BW; bx++) {
    let s = 0;
    for (let y = 0; y < 6; y++) for (let x = 0; x < 6; x++) { const i = ((by * 6 + y) * W + bx * 6 + x) * 3; s += 0.3 * f[i] + 0.59 * f[i + 1] + 0.11 * f[i + 2]; }
    b[by * BW + bx] = s / 36;
  }
  return b;
}
const changed = (a, b) => { let n = 0; for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 20) n++; return n / a.length; };
const fd = openSync(`${out}/keyframes.bin`, 'w'); const index = []; const diffs = [];
let buf = Buffer.alloc(0), n = 0, prev = null, run = [], last = null;
function flush() {
  if (run.length >= MIN) {
    const mid = run[Math.floor(run.length / 2)];
    if (!last || changed(last.b, mid.b) > 0.003) { writeSync(fd, mid.f); index.push({ frame: mid.n, t: +(mid.n / +FPS + (+start || 0)).toFixed(3), still: run.length }); last = mid; }
  }
  run = [];
}
ff.stdout.on('data', d => {
  buf = Buffer.concat([buf, d]);
  while (buf.length >= FR) {
    const f = Buffer.from(buf.subarray(0, FR)); buf = buf.subarray(FR);
    const b = blocks(f), c = prev ? changed(prev, b) : 1; diffs.push(+c.toFixed(4));
    if (c < 0.01) run.push({ f, b, n }); else flush();
    prev = b; n++;
  }
});
ff.on('close', () => { flush(); closeSync(fd); writeFileSync(`${out}/keyframes.json`, JSON.stringify({ W, H, fps: +FPS, start: +start || 0, frames: n, keyframes: index })); writeFileSync(`${out}/diffs.json`, JSON.stringify(diffs)); console.log(`${n} frames, ${index.length} keyframes`); });
