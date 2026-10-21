// V47 step 3: the website AI on every position the human faced. Replays game.json with the same
// public state the website keeps (card-memory.js: bag counts and bonus cycle, updated after each
// move with the new preview and board), then runs the website search (six-ply / 0.005, clamp,
// cycle-aware, float16 weights, no time limit) and records the four root values.
// The preview before move i is the card that entered on move i (bonus: the window read by hand).
//  node eval.mjs FROM TO OUT.jsonl
import { readFileSync, writeFileSync } from 'node:fs';
import { chooseFastMove } from '../../dist/fast-search.js';
import { initialCounts, observePreview, initialCycle, observeCycle } from '../../dist/card-memory.js';
import { loadWebModel } from '../v45-explain/explain.mjs';
const L = new URL('./', import.meta.url);
const game = JSON.parse(readFileSync(new URL('game.json', L)));
const windows = JSON.parse(readFileSync(new URL('bonus-windows.json', L))).windows;
const [from, to, out] = [+process.argv[2], +process.argv[3], process.argv[4]];
const steps = game.steps, model = loadWebModel('f16');
export function preview(i) {
  const s = steps[i]; if (!s || s.cell < 0) return null;
  return s.card <= 3 ? { bonus: false, candidates: [s.card] } : { bonus: true, candidates: windows[i].window, probabilities: [1 / 3, 1 / 3, 1 / 3] };
}
let board = game.startBoard, counts = initialCounts(board, preview(0)), cycle = initialCycle();
const lines = [];
for (let i = 0; i < Math.min(to, steps.length); i++) {
  const next = preview(i);
  if (i >= from && next) {
    const stats = { values: null }, t0 = performance.now();
    const r = chooseFastMove(board, next, model, { maxDepth: 6, threshold: 0.005, clampLeaf: true, budgetMs: 1e9, maxNodes: 1e12, remaining: counts, cycle, stats });
    lines.push(JSON.stringify({ i, t: steps[i].t, board, next, counts, cycle, human: steps[i].dir, ai: ['left', 'right', 'up', 'down'].indexOf(r.direction), values: stats.values, depth: r.depth, nodes: r.nodes, ms: Math.round(performance.now() - t0), multi: steps[i].multi }));
  }
  board = steps[i].board;
  const after = preview(i + 1);
  if (after) { counts = observePreview(counts, after); cycle = observeCycle(cycle, after, board); }
}
writeFileSync(out, lines.join('\n') + '\n');
