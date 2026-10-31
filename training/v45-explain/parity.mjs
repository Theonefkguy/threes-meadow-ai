// The traced copy must choose the same moves with the same root values as dist/fast-search.js (6/0.005).
import { readFileSync } from 'node:fs';
import { chooseFastMove as web } from '../../dist/fast-search.js';
import { chooseFastMove as traced } from './trace-search.mjs';
import { loadWebModel, explainMove } from './explain.mjs';
const model = loadWebModel();
const states = JSON.parse(readFileSync(new URL('../v25-adaptive-cutoff/browser/states.json', import.meta.url)));
let bad = 0, maxCheck = 0;
states.forEach((s, i) => {
  const cycle = { pos: i % 21, used: i % 3 === 0 ? 1 : 0, excl: (i % 21) >> 1 }, o = { remaining: s.remaining, maxDepth: 6, threshold: 0.005, clampLeaf: true, budgetMs: 1e9, maxNodes: 1e12, cycle };
  const a = {}, b = { trace: null }, ra = web(s.board, s.next, model, { ...o, stats: a }), rb = traced(s.board, s.next, model, { ...o, stats: b });
  if (ra.direction !== rb.direction || JSON.stringify(a.values) !== JSON.stringify(b.values)) bad++;
  // the traced root terms must reproduce the root values: value(d) = merge reward + mean over outcomes
  if (i % 20 === 0) { const e = explainMove({ ...s, cycle }, model); for (const d of Object.values(e.perDirection)) for (const x of d.outcomes) maxCheck = Math.max(maxCheck, Math.abs(x.check) / Math.max(1, Math.abs(x.value))); }
});
console.log(JSON.stringify({ states: states.length, mismatches: bad, maxRelativeBranchCheck: maxCheck }));
if (bad) process.exit(1);
