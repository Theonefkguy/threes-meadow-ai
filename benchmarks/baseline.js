import { DIRECTIONS, projectMove, entryCell, canMerge, hasMoves, bonusPreviews, cardProbabilities } from '../dist/engine.js';

const rank = n => n < 3 ? (n ? 1 : 0) : Math.log2(n / 3) + 2;
function evaluate(board) {
  const flat = board.flat(), empty = flat.filter(n => !n).length;
  if (!empty && !hasMoves(board)) return -100000;
  let value = empty * 260, roughness = 0, disorder = 0;
  const highest = Math.max(...flat);
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      const a = board[y][x];
      value += rank(a) * 3;
      for (const b of [x < 3 ? board[y][x + 1] : 0, y < 3 ? board[y + 1][x] : 0]) {
        if (canMerge(a,b)) value += 28 * rank(a);
        if (a && b) roughness += Math.abs(rank(a) - rank(b));
      }
    }
  }
  for (let i = 0; i < 4; i++) {
    for (const line of [board[i], board.map(row => row[i])]) {
      let up = 0, down = 0;
      for (let j = 1; j < 4; j++) {
        const diff = rank(line[j]) - rank(line[j - 1]);
        up += Math.max(0, diff); down += Math.max(0, -diff);
      }
      disorder += Math.min(up, down);
    }
  }
  if ([board[0][0],board[0][3],board[3][0],board[3][3]].includes(highest)) value += rank(highest) * 65;
  return value - roughness * 7 - disorder * 22;
}

// Only visible information is supplied. Future ordinary cards use their
// long-run 1/3 probabilities; the hidden deck order is never inspected.
function upcoming(board) {
  const previews = bonusPreviews(Math.max(...board.flat()));
  const bonus = previews.length ? 1 / 21 : 0;
  return [
    ...[1,2,3].map(n => ({ candidates:[n], probability:(1 - bonus) / 3 })),
    ...previews.map(preview => ({ ...preview, probability:bonus * preview.probability }))
  ];
}

export function chooseMove(board, next, { maxDepth = 3, budgetMs = 160, maxNodes = 24000 } = {}) {
  if (!next?.candidates?.length) return { direction:null, depth:0, nodes:0 };
  const legal = DIRECTIONS.map(direction => ({ direction, projection:projectMove(board,direction) })).filter(m => m.projection.changed);
  if (!legal.length) return { direction:null, depth:0, nodes:0 };
  const deadline = performance.now() + budgetMs;
  const timeout = Symbol('budget');
  let nodes = 0, completed = 0, best = legal[0].direction;
  const cache = new Map();
  function tick(depth) {
    nodes++;
    if (completed && (nodes > maxNodes || (nodes % 64 === 0 && performance.now() >= deadline))) throw timeout;
  }
  function decision(state, preview, depth) {
    tick(depth);
    const key = `${depth}:${state.flat().join(',')}:${preview.candidates.join(',')}:${cardProbabilities(preview).join(',')}`;
    if (cache.has(key)) return cache.get(key);
    let value = -100000;
    for (const direction of DIRECTIONS) {
      const projection = projectMove(state, direction);
      if (projection.changed) value = Math.max(value, afterMove(projection, direction, preview, depth));
    }
    cache.set(key, value);
    return value;
  }
  function afterMove(projection, direction, preview, depth) {
    let total = 0;
    const probabilities = cardProbabilities(preview);
    for (const lane of projection.lanes) {
      const [y,x] = entryCell(direction,lane);
      for (const [index, card] of preview.candidates.entries()) {
        tick(depth);
        const state = projection.board.map(row => [...row]);
        state[y][x] = card;
        if (depth === 1) total += probabilities[index] * evaluate(state);
        else total += probabilities[index] * upcoming(state).reduce((sum, future) => sum + future.probability * decision(state, future, depth - 1), 0);
      }
    }
    return total / projection.lanes.length;
  }
  for (let depth = 1; depth <= maxDepth; depth++) {
    try {
      let value = -Infinity, choice = best;
      for (const move of legal) {
        const candidate = afterMove(move.projection, move.direction, next, depth);
        if (candidate > value) { value = candidate; choice = move.direction; }
      }
      best = choice; completed = depth;
    } catch (error) { if (error !== timeout) throw error; break; }
  }
  return { direction:best, depth:completed, nodes };
}
