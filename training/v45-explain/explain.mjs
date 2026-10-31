// V45 explanation tools for the website AI (V31 float16 weights, six-ply / 0.005, cycle-aware).
//  decompose(board, model): stage, total value, the 64 four-cell and 32 five-cell feature terms, and
//    a per-cell attribution (each term split evenly over its cells).
//  explainMove(state, model): root values of the four directions, the margin between the best two,
//    and per direction the spawn outcomes and next-card branches that drive the value.
import { readFileSync } from 'node:fs';
import { decodeModel, FEATURES, FEATURES5, PATTERNS, TABLE, TABLE5, stageForRanks } from '../../dist/rl-model.js';
import { chooseFastMove } from './trace-search.mjs';
export const toRank = n => n < 3 ? n : Math.log2(n / 3) + 3;
const ROOT = new URL('../../', import.meta.url);
export function loadWebModel(precision = 'f16') {
  const buf = p => { const b = readFileSync(new URL(p, ROOT)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
  const res = precision === 'f16' ? [1, 2, 3].map(s => `training/web-release/models/ntuple-v31-s${s}.f16`)
    : ['training/v31-stages/models/S1-final.res', 'training/v30-parallel/models/P-final.res', 'training/v31-stages/models/S3-final.res'];
  return decodeModel(buf('training/web-release/models/ntuple-v31.bin'), res.map(buf));
}
export function decompose(board, model) {
  const b = board.flat().map(toRank), stage = stageForRanks(b, model.stages), off = stage * PATTERNS.length * TABLE;
  const terms = [], cell = Array(16).fill(0);
  FEATURES.forEach((p, f) => {
    const key = p.reduce((k, c, j) => k | Math.min(15, b[c]) << (4 * j), 0), w = model.weights[off + (f >> 3) * TABLE + key];
    terms.push({ kind: '4-cell', pattern: f >> 3, symmetry: f & 7, cells: p, value: w }); p.forEach(c => cell[c] += w / p.length);
  });
  const R = model.residuals[stage];
  if (R) FEATURES5.forEach((p, f) => {
    const key = p.reduce((k, c, j) => k | Math.min(15, b[c]) << (4 * j), 0), w = R[(f >> 3) * TABLE5 + key];
    terms.push({ kind: '5-cell', pattern: f >> 3, symmetry: f & 7, cells: p, value: w }); p.forEach(c => cell[c] += w / p.length);
  });
  const total = terms.reduce((s, t) => s + t.value, 0);
  return { stage, total, check: model.value(b, stage), terms, cellAttribution: [0, 1, 2, 3].map(y => cell.slice(y * 4, y * 4 + 4)) };
}
export const SEARCH = { maxDepth: 6, threshold: 0.005, clampLeaf: true, budgetMs: 1e9, maxNodes: 1e12 };
export function explainMove(state, model, opts = {}) {
  const stats = { trace: null };
  const r = chooseFastMove(state.board, state.next, model, { ...SEARCH, ...opts, remaining: state.remaining, cycle: state.cycle, stats });
  const dirs = ['left', 'right', 'up', 'down'], vals = stats.values.map((v, i) => ({ direction: dirs[i], value: v })).filter(x => x.value !== null).sort((a, b) => b.value - a.value);
  const per = {};
  for (const t of stats.trace) {
    const o = per[t.direction] ??= { outcomes: [] };
    const nextMean = t.branches.length ? t.branches.reduce((s, x) => s + x.probability * x.value, 0) : 0;
    o.outcomes.push({ cell: t.cell, card: t.card, probability: t.probability, value: t.points + t.future, nextBranches: t.branches.map(x => ({ next: x.next, bonus: x.bonus, probability: x.probability, value: x.value, cut: x.cut })), check: nextMean - t.future });
  }
  return { direction: r.direction, depth: r.depth, nodes: r.nodes, ranking: vals, margin: vals.length > 1 ? vals[0].value - vals[1].value : null, perDirection: per };
}
