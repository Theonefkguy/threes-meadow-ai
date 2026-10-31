// V47 step 2: follow the game through the keyframes. From the last accepted board, every legal
// continuation (direction x entry lane x new card, also two moves in a row) is scored against the
// next keyframe with per-value cell templates learned on the way; the best one is accepted when it
// fits clearly better than "nothing happened". Partial drags, animations and menus fit nothing and
// are skipped. The preview is only logged: it can lag behind the board.
//  node track.mjs KEYFRAME_DIR START_INDEX 'START_BOARD_JSON' OUT.json [SEEDS.json]
// SEEDS: {dir, frames: [[index, board], ...]}, hand-labelled keyframes (any keyframe dir) that give every card value a template up front
// (a bonus card can show a value never made by a merge yet, which would otherwise be guesswork).
import { readFileSync, writeFileSync } from 'node:fs';
import { projectMove, entryCell, DIRECTIONS } from '../../dist/engine.js';
const [dir, startArg, boardArg, out, seedFile] = process.argv.slice(2);
const meta = JSON.parse(readFileSync(`${dir}/keyframes.json`)); const { W, H } = meta; const KF = meta.keyframes;
const FR = W * H * 3; const bins = new Map(); const binOf = d => bins.get(d) ?? bins.set(d, readFileSync(`${d}/keyframes.bin`)).get(d);
// Cell features: a quarter-resolution grey patch (inner TW x TH plus a margin M) and the mean colour
// of the inner area. Cards bob by a few pixels, so patches are compared at the best shift within
// +-S quarter-res pixels; the mean colour separates empty / 1 / 2 / grey card / monster.
export const CX = [54, 114, 174, 234], CY = [156, 236, 316, 396];
const TW = 12, TH = 16, M = 2, S = 2, PW = TW + 2 * M, PH = TH + 2 * M;
function cells(k, d = dir) {
  const f = binOf(d).subarray(k * FR, (k + 1) * FR), res = [];
  for (let c = 0; c < 16; c++) {
    const g = new Float32Array(PW * PH), x0 = CX[c % 4] - PW, y0 = CY[c >> 2] - PH, col = [0, 0, 0]; let i = 0;
    for (let y = 0; y < PH; y++) for (let x = 0; x < PW; x++) {
      let s = 0; const inner = y >= M && y < PH - M && x >= M && x < PW - M;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const p = ((y0 + 2 * y + dy) * W + x0 + 2 * x + dx) * 3; s += 0.3 * f[p] + 0.59 * f[p + 1] + 0.11 * f[p + 2];
        if (inner) { col[0] += f[p]; col[1] += f[p + 1]; col[2] += f[p + 2]; }
      }
      g[i++] = s / 4;
    }
    res.push({ g, col: col.map(v => v / (TW * TH * 4)) });
  }
  return res;
}
// Preview: pixel classes in the box x 92..202, y 8..54 (keyframe coords).
export function preview(k) {
  const f = binOf(dir).subarray(k * FR, (k + 1) * FR); let blue = 0, red = 0, greyMid = 0, greySide = 0;
  for (let y = 8; y < 54; y++) for (let x = 92; x < 202; x++) {
    const p = (y * W + x) * 3, r = f[p], g = f[p + 1], b = f[p + 2];
    if (b > 180 && r < 130 && g > 140) blue++; else if (r > 200 && g < 130 && b < 150) red++;
    else if (r > 105 && Math.abs(r - g) < 25 && Math.abs(r - b) < 35) (x > 130 && x < 165 ? greyMid++ : greySide++);
  }
  if (greySide > 400) return 'bonus';
  if (blue > 150) return 1; if (red > 150) return 2; if (greyMid > 150) return 3;
  return null;
}
// Templates: per value a few stored examples (monster cards change face, neighbours overlap); a cell
// is as close to a value as to its nearest example.
const T = new Map(), PROTOS = v => typeof v === 'number' && v >= 192 ? 12 : 6;
function dist(x, t) {
  let best = Infinity;
  for (let sy = -S; sy <= S; sy++) for (let sx = -S; sx <= S; sx++) {
    let s = 0;
    for (let y = M; y < M + TH; y++) { const a = y * PW, b = (y + sy) * PW + sx; for (let i = M; i < M + TW; i++) s += Math.abs(t.g[a + i] - x.g[b + i]); }
    if (s < best) best = s;
  }
  return best / (TW * TH) + 0.5 * (Math.abs(x.col[0] - t.col[0]) + Math.abs(x.col[1] - t.col[1]) + Math.abs(x.col[2] - t.col[2])) / 3;
}
function learn(v, x) {
  const list = T.get(v) ?? T.set(v, []).get(v);
  if (list.length < PROTOS(v) && (!list.length || near(x, v) > 6)) list.push(x);
}
const near = (x, v) => Math.min(...T.get(v).map(t => dist(x, t)));
// A just-merged card is drawn blank (no number) for a moment, and fast players move on before the
// number appears: a cell merged by the last move may also match the blank template ('B') at a cost.
const BLANK_COST = 8, LATE_CARD_COST = 8, CAP = 40;
function table(obs, values) {
  return obs.map(x => {
    const known = new Map(); for (const v of T.keys()) known.set(v, near(x, v));
    const gen = Math.min(...[...known].filter(([v]) => typeof v === 'number' && v >= 3).map(([, d]) => d)) + 10;
    const row = new Map(); for (const v of values) row.set(v, Math.min(known.get(v) ?? (v >= 3 ? gen : 99), 99));
    row.blank = (known.get('B') ?? 99) + BLANK_COST; return row;
  });
}
const flat = b => b.flat();
const maxOf = b => Math.max(...flat(b));
function bonusValues(b) { const r = []; for (let n = 6; n <= maxOf(b) / 8; n *= 2) r.push(n); return r; }
function moves(b, pv) {
  const res = [];
  const vals = pv === 'bonus' ? bonusValues(b) : pv ? [pv] : [1, 2, 3, ...bonusValues(b)];
  DIRECTIONS.forEach((d, di) => {
    const p = projectMove(b, d); if (!p.changed) return;
    if (flat(p.board).includes(12288)) { res.push({ board: p.board, dir: di, cell: -1, card: 0, merged: p.merged }); return; }
    for (const lane of p.lanes) { const [y, x] = entryCell(d, lane);
      for (const v of vals) { const nb = p.board.map(r => [...r]); nb[y][x] = v; res.push({ board: nb, dir: di, cell: y * 4 + x, card: v, merged: p.merged }); } }
  });
  return res;
}
// The card entering on the last move may not have slid in yet: its cell may still look empty.
// A cell may be mid-animation (a squashed merge, a card sliding in): each cell's cost is capped at
// CAP, so one such cell costs at most CAP/16 while real value differences (10-30) still count.
function score(tab, b, entry = -1, merged = [], info = {}) {
  let sum = 0; info.hidden = false;
  flat(b).forEach((v, c) => { let x = tab[c].get(v) ?? 99;
    if (c === entry) { const empty = tab[c].get(0) + LATE_CARD_COST; if (empty < x) { x = empty; info.hidden = true; } }
    else if (merged.includes(c)) x = Math.min(x, tab[c].blank);
    sum += Math.min(x, CAP); });
  return sum / 16;
}
// Priors: a bonus card comes once per 21 draws, and fewer moves between two keyframes are likelier.
// A last card that has not slid in yet gets no prior, so its possible values tie and the frame is skipped.
function seqScore(tab, seq) {
  const info = {}, last = seq.at(-1), sc = score(tab, last.board, last.cell, last.merged, info);
  return sc + 0.4 * seq.filter(m => m.card > 3 && !(m === last && info.hidden)).length + 0.2 * (seq.length - 1);
}
const allValues = b => { const s = new Set([0, 1, 2, 3, ...flat(b)]); for (let n = 6; n <= 12288; n *= 2) if (n <= 2 * maxOf(b)) s.add(n); return [...s]; };

let state = JSON.parse(boardArg); const start = +startArg;
cells(start).forEach((x, c) => learn(flat(state)[c], x));
if (seedFile) { const sd = JSON.parse(readFileSync(seedFile)); for (const [k, b] of sd.frames) cells(k, sd.dir).forEach((x, c) => learn(b.flat()[c], x));
  for (const [d, k, c] of sd.blank ?? []) learn('B', cells(k, d)[c]); }
let pv = preview(start);
const steps = [], skipped = [], log = [];
const ACCEPT = (process.env.ACCEPT || '5,0.3,8,1.5').split(',').map(Number);
for (let k = start + 1; k < KF.length; k++) {
  const obs = cells(k), tab = table(obs, allValues(state)), stay = score(tab, state);
  // Beam over up to three moves; the preview is not trusted (it can lag behind the board).
  let level = moves(state, null).map(m => ({ seq: [m], s: seqScore(tab, [m]) })).sort((a, b) => a.s - b.s);
  let all = level.slice();
  for (let depth = 2; depth <= 3 && (!all[0] || all[0].s > 3); depth++) {
    const nextLevel = [];
    for (const a of level.slice(0, depth === 2 ? 400 : 150)) for (const m of moves(a.seq.at(-1).board, null)) { const seq = [...a.seq, m]; nextLevel.push({ seq, s: seqScore(tab, seq) }); }
    nextLevel.sort((a, b) => a.s - b.s); level = nextLevel;
    if (level[0] && (!all[0] || level[0].s < all[0].s - 1)) all = [...level, ...all].sort((a, b) => a.s - b.s);
  }
  const best = all[0], second = all.find(x => x.seq.at(-1).board.flat().join() !== best.seq.at(-1).board.flat().join());
  const p = preview(k);
  if (+process.env.DEBUGK === k) for (const c of all.slice(0, 6)) console.error(c.s.toFixed(2), JSON.stringify(c.seq.map(m => [m.dir, m.cell, m.card])), flat(c.seq.at(-1).board).map((v, i) => `${v}:${Math.round(tab[i].get(v) ?? 99)}`).join(' '));
  // Accept a good fit with any clear runner-up gap, or a looser fit when every alternative is far worse.
  const gap = second ? second.s - best.s : 9;
  // The winning move makes 12288, which has no template: accept it when nothing else comes close.
  const final = best && best.seq.at(-1).cell === -1 && gap > 5 && best.s < stay - 5;
  const accept = final || best && best.s < stay - 1.5 && (best.s < ACCEPT[0] && gap > ACCEPT[1] || best.s < ACCEPT[2] && gap > ACCEPT[3]);
  log.push({ k, t: KF[k].t, stay: +stay.toFixed(2), best: best ? +best.s.toFixed(2) : null, gap: second ? +(second.s - best.s).toFixed(2) : null, n: best ? best.seq.length : 0, accept, pv: p,
    ...(accept || !best ? {} : { cand: best.seq.map(m => [m.dir, m.cell, m.card]), cost: flat(best.seq.at(-1).board).map((v, c) => Math.round(tab[c].get(v) ?? 99)) }) });
  if (!accept) { if (stay < 8 && p !== null) pv = p; skipped.push(k); continue; }
  // Self-check: cells where a direct per-cell read clearly beats the accepted board's value.
  const lastM = best.seq.at(-1), fb = flat(lastM.board);
  const disagree = fb.filter((v, c) => c !== lastM.cell && !lastM.merged.includes(c) && (tab[c].get(v) ?? 99) > 25 && [...tab[c].values()].some(x => x < 10)).length;
  for (const m of best.seq) steps.push({ disagree, k, t: KF[k].t, dir: m.dir, cell: m.cell, card: m.card, preview: m === best.seq[0] ? pv : null, board: m.board, fit: +best.s.toFixed(2), gap: second ? +(second.s - best.s).toFixed(2) : null, multi: best.seq.length });
  state = best.seq.at(-1).board;
  // Add examples only from clear frames and cells that already fit, so animation never becomes a template.
  if (gap > 0.8 && best.s < ACCEPT[0]) obs.forEach((x, c) => { const v = flat(state)[c]; if ((tab[c].get(v) ?? 99) <= 30) learn(v, x); });
  pv = p;
  if (best.seq.at(-1).cell === -1) break; // 12288: game over
}
writeFileSync(out, JSON.stringify({ start, startBoard: JSON.parse(boardArg), steps, skipped, log }));
const lowGap = steps.filter(s => s.gap !== null && s.gap < 0.8).length;
console.log(`moves ${steps.length}, keyframes used ${new Set(steps.map(s => s.k)).size}, skipped ${skipped.length}, multi-move frames ${new Set(steps.filter(s => s.multi > 1).map(s => s.k)).size}, low-gap ${lowGap}, last t ${steps.at(-1)?.t}, max ${maxOf(state)}`);
