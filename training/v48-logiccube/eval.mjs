// V48 step 2: the website AI on every position of a decoded LogicCube game (as V47 eval.mjs).
// PKU 2048 Threes draws a bonus with a constant 1/21 chance once the highest card is >= 48 (no
// 21-draw block), so the search runs without `cycle` (constant 1/21, the V31 rule). The replay keeps
// only the bonus card, not the 3-card window the player saw: with several possible windows the
// search runs once per window and the root values are averaged (each window equally likely).
// The website game ends at 12288 (the search treats it as terminal), so positions with 12288 on the
// board are not evaluated.
//  node eval.mjs games/ID.json OUT.jsonl
import { readFileSync, writeFileSync } from 'node:fs';
import { chooseFastMove } from '../../dist/fast-search.js';
import { initialCounts, observePreview } from '../../dist/card-memory.js';
import { loadWebModel } from '../v45-explain/explain.mjs';
const [file, out] = process.argv.slice(2);
const game = JSON.parse(readFileSync(file)), steps = game.steps, model = loadWebModel('f16');
const previews = i => { const s = steps[i]; if (!s) return null;
  return s.card <= 3 ? [{ bonus: false, candidates: [s.card] }] : s.windows.map(w => ({ bonus: true, candidates: w, probabilities: w.map(() => 1 / w.length) })); };
let board = game.startBoard, counts = initialCounts(board, previews(0)[0]);
const lines = [];
for (let i = 0; i < steps.length; i++) {
  if (Math.max(...board.flat()) >= 12288) break;
  const nexts = previews(i), t0 = performance.now(), runs = [];
  for (const next of nexts) {
    const stats = { values: null };
    const r = chooseFastMove(board, next, model, { maxDepth: 6, threshold: 0.005, clampLeaf: true, budgetMs: 1e9, maxNodes: 1e12, remaining: counts, stats });
    runs.push({ values: stats.values, depth: r.depth, nodes: r.nodes });
  }
  const values = [0, 1, 2, 3].map(d => runs[0].values[d] == null ? null : runs.reduce((s, r) => s + r.values[d], 0) / runs.length);
  const legal = values.map((v, d) => v == null ? -1 : d).filter(d => d >= 0), ai = legal.reduce((a, d) => values[d] > values[a] ? d : a, legal[0]);
  lines.push(JSON.stringify({ i, t: steps[i].t, ms: steps[i].ms, board, next: nexts.length === 1 ? nexts[0] : { bonus: true, windows: nexts.map(n => n.candidates) }, counts,
    human: steps[i].dir, ai, values, windows: nexts.length, nodes: runs.reduce((s, r) => s + r.nodes, 0), ms_ai: Math.round(performance.now() - t0), multi: 1 }));
  board = steps[i].board;
  const after = previews(i + 1);
  if (after) counts = observePreview(counts, after[0]);
}
writeFileSync(out, lines.join('\n') + '\n');
