// Human-readable explanation of one saved state (training/v38-progress/cgame.h line format).
//  node training/v45-explain/show.mjs POOL_FILE INDEX
import { readFileSync } from 'node:fs';
import { loadWebModel, decompose, explainMove, toRank } from './explain.mjs';
const [file, idx] = process.argv.slice(2);
const f = readFileSync(file, 'utf8').trim().split('\n')[+idx].split(/\s+/).map(Number);
let k = 0; const seed = f[k++], turns = f[k++], left = f[k++]; k += left;
const size = f[k++], bonus = !!f[k++], cards = [], probs = [];
for (let i = 0; i < 3; i++) { const c = f[k++], p = f[k++]; if (i < size) { cards.push(c < 3 ? c : 3 * 2 ** (c - 3)); probs.push(p); } }
const remaining = [f[k++], f[k++], f[k++]], ranks = f.slice(k, k + 16); k += 16; k += 2;
const cycle = { pos: f[k++], used: f[k++], excl: f[k++] };
const board = [0, 1, 2, 3].map(y => ranks.slice(y * 4, y * 4 + 4).map(r => r < 3 ? r : 3 * 2 ** (r - 3)));
const state = { board, next: { bonus, candidates: cards, probabilities: probs }, remaining, cycle };
const model = loadWebModel(), fmt = n => Math.round(n).toLocaleString('en-US'), pad = (s, w) => String(s).padStart(w);
console.log(`state ${idx} (seed ${seed}, move ${turns}), next ${bonus ? 'bonus ' : ''}${cards.join('/')}, bag 1/2/3 left ${remaining.join('/')}, cycle ${JSON.stringify(cycle)}`);
board.forEach(r => console.log('  ' + r.map(v => pad(v || '.', 5)).join('')));
const d = decompose(board, model);
console.log(`\nevaluation of this board as it stands (stage ${d.stage}): ${fmt(d.total)}  [model.value ${fmt(d.check)}]`);
console.log('per-cell attribution (each pattern value split over its cells):');
d.cellAttribution.forEach(r => console.log('  ' + r.map(v => pad(fmt(v), 9)).join('')));
const top = [...d.terms].sort((a, b) => b.value - a.value);
console.log('largest positive terms: ' + top.slice(0, 3).map(t => `${t.kind} p${t.pattern}s${t.symmetry} cells[${t.cells}] ${fmt(t.value)}`).join('; '));
console.log('largest negative terms: ' + top.slice(-3).reverse().map(t => `${t.kind} p${t.pattern}s${t.symmetry} cells[${t.cells}] ${fmt(t.value)}`).join('; '));
const e = explainMove(state, model);
console.log(`\nsearch (6-ply, cutoff 0.005): plays ${e.direction}; margin over the runner-up ${fmt(e.margin)} (${(100 * e.margin / Math.abs(e.ranking[0].value)).toFixed(2)}%)`);
for (const r of e.ranking) {
  const o = e.perDirection[r.direction].outcomes, worst = [...o].sort((a, b) => a.value - b.value)[0], bestO = [...o].sort((a, b) => b.value - a.value)[0];
  console.log(`  ${r.direction.padEnd(5)} ${pad(fmt(r.value), 10)}  outcomes ${o.length}; worst: ${worst.card} at cell ${worst.cell} -> ${fmt(worst.value)}; best: ${bestO.card} at cell ${bestO.cell} -> ${fmt(bestO.value)}`);
}
