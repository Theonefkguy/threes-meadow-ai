import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame, projectMove, bonusGroups, bonusPreviews, canMerge, score, hasMoves } from '../dist/engine.js';

const empty = () => Array.from({ length: 4 }, () => Array(4).fill(0));
const withRow = row => [row, ...empty().slice(1)];
const seed = n => () => ((n = (Math.imul(n, 1664525) + 1013904223) >>> 0) / 4294967296);

test('nine-card opening and preview are drawn from one balanced deck', () => {
  const patterns = new Set();
  for (let n = 1; n <= 128; n++) {
    const game = createGame(seed(n));
    const cards = game.board.flat().filter(Boolean);
    assert.equal(cards.length, 9);
    assert.equal(game.deck.length, 2);
    assert.equal(game.next.bonus, false);
    const all = [...cards, ...game.deck, ...game.next.candidates];
    for (const value of [1,2,3]) assert.equal(all.filter(n => n === value).length, 4);
    patterns.add([1,2,3].map(v => cards.filter(n => n === v).length).join());
  }
  assert(patterns.size > 1, 'opening must not always be three of each');
});

test('ordinary deck continues through remaining cards before refill', () => {
  const game = createGame(seed(34));
  const initial = [...game.board.flat().filter(Boolean), ...game.next.candidates];
  for (let i = 0; i < 2; i++) {
    game.board = withRow([0,1,0,0]);
    game.move('left');
    initial.push(...game.next.candidates);
  }
  assert.equal(game.deck.length, 0);
  for (const n of [1,2,3]) assert.equal(initial.filter(v => v === n).length, 4);
  game.board = withRow([0,1,0,0]);
  game.move('left');
  assert.equal(game.deck.length, 11);
});

test('single-cell sliding, edge-first merge, and no cascade', () => {
  for (const [before, after] of [
    [[0,0,0,3],[0,0,3,0]], [[0,0,3,3],[0,3,3,0]],
    [[1,2,1,2],[3,1,2,0]], [[3,3,3,3],[6,3,3,0]],
    [[6,3,3,3],[6,6,3,0]], [[1,1,2,3],[1,3,3,0]],
    [[1,1,1,1],[1,1,1,1]]
  ]) assert.deepEqual(projectMove(withRow(before), 'left').board[0], after);
  assert(!canMerge(1,1)); assert(!canMerge(2,2));
  assert(canMerge(2,1)); assert(canMerge(6,6)); assert(!canMerge(0,3));
});

test('projection is pure and blocked moves do not consume random numbers', () => {
  let calls = 0;
  const game = createGame(() => { calls++; return .5; });
  game.board = withRow([1,1,1,1]);
  const snapshot = JSON.stringify(game);
  const beforeCalls = calls;
  for (let i = 0; i < 8; i++) {
    projectMove(game.board, 'up');
    assert.equal(game.move('left').changed, false);
  }
  assert.equal(JSON.stringify(game), snapshot);
  assert.equal(calls, beforeCalls);
});

test('one tile enters the opposite edge in the only lane that moved', () => {
  const examples = [
    ['left',[2,1],[2,3]], ['right',[1,2],[1,0]],
    ['up',[2,1],[3,1]], ['down',[1,2],[0,2]]
  ];
  for (const [dir, [y,x], [sy,sx]] of examples) {
    const game = createGame(seed(7));
    game.board = empty(); game.board[y][x] = 3;
    game.next = { bonus:false, candidates:[2] };
    const result = game.move(dir);
    assert.equal(result.spawned, sy * 4 + sx);
    assert.equal(game.board[sy][sx], 2);
    assert.equal(game.board.flat().filter(Boolean).length, 2);
    assert.equal(game.board.flat().reduce((a,b) => a+b), 5);
  }
});

test('bonus windows are at most three consecutive legal values', () => {
  assert.deepEqual(bonusGroups(24), []);
  assert.deepEqual(bonusGroups(48), [[6]]);
  assert.deepEqual(bonusGroups(96), [[6,12]]);
  assert.deepEqual(bonusGroups(192), [[6,12,24]]);
  assert.deepEqual(bonusGroups(384), [[6,12,24],[12,24,48]]);
  assert.deepEqual(bonusGroups(768), [[6,12,24],[12,24,48],[24,48,96]]);
});

test('bonus trial boundary and no ordinary-deck consumption on bonus', () => {
  let draws = [];
  const game = createGame(() => draws.length ? draws.shift() : .5);
  for (const [trial, bonus] of [[0,true], [1/21 - 1e-8,true], [1/21,false], [.99,false]]) {
    game.board = withRow([0,384,0,0]); game.over = false;
    game.next = { bonus:false, candidates:[1] }; game.deck = [1,2,3];
    draws = [0, trial, .99]; // entry lane, bonus trial, candidate window
    game.move('left');
    assert.equal(game.next.bonus, bonus);
    assert.equal(game.deck.length, bonus ? 3 : 2);
    if (bonus) assert.deepEqual(game.next.candidates, [12,24,48]);
  }
  game.board = withRow([0,24,0,0]); game.next = { bonus:false, candidates:[1] };
  draws = [0,0,0];
  game.move('left');
  assert.equal(game.next.bonus, false);
});

test('a committed bonus always belongs to the displayed candidates', () => {
  let draws = [];
  const game = createGame(() => draws.length ? draws.shift() : .5);
  for (const [roll, expected] of [[0,12],[.34,24],[.99,48]]) {
    game.board = withRow([0,3,0,0]);
    game.next = { bonus:true, candidates:[12,24,48] };
    draws = [0,roll,.99];
    assert.equal(game.move('left').value, expected);
  }
});

test('score, terminal state, and reset after the terminal move', () => {
  assert.equal(score(withRow([1,2,3,6])), 12);
  assert.equal(score(withRow([12,24,48,96])), 1080);
  const game = createGame(seed(5));
  game.board = Array.from({ length:4 }, () => [1,1,1,1]);
  game.board[0][1] = 2;
  game.next = { bonus:false, candidates:[1] };
  const deck = [...game.deck];
  assert(hasMoves(game.board));
  game.move('left');
  assert.equal(game.over, true);
  assert.equal(game.next, null);
  assert.deepEqual(game.deck, deck);
  assert.equal(game.move('right').changed, false);
  game.reset();
  assert(!game.over); assert.equal(game.turns, 0); assert(game.next);
});

test('official-modern bonus: uniform window, then uniform card within the window', () => {
  // Expected marginals (x 1/denominator) for each maximum card, window-first.
  const expected = {48:[[6],[1],1],96:[[6,12],[1,1],2],192:[[6,12,24],[1,1,1],3],
    384:[[6,12,24,48],[1,2,2,1],6],768:[[6,12,24,48,96],[1,2,3,2,1],9],
    1536:[[6,12,24,48,96,192],[1,2,3,3,2,1],12]};
  for (const highest of [48,96,192,384,768,1536,3072,6144]) {
    const previews = bonusPreviews(highest), groups = bonusGroups(highest);
    assert.deepEqual(previews.map(p => p.candidates), groups);
    for (const p of previews) {
      assert(Math.abs(p.probability - 1/groups.length) < 1e-15, `window weight max=${highest}`);
      for (const q of p.probabilities) assert(Math.abs(q - 1/p.candidates.length) < 1e-15);
    }
    const marginals = new Map();
    for (const p of previews) p.candidates.forEach((c,i) => marginals.set(c,(marginals.get(c)||0)+p.probability*p.probabilities[i]));
    assert(Math.abs([...marginals.values()].reduce((a,b)=>a+b,0)-1) < 1e-12);
    if (expected[highest]) {
      const [cards, weights, den] = expected[highest];
      assert.deepEqual([...marginals.keys()], cards);
      cards.forEach((c,i) => assert(Math.abs(marginals.get(c) - weights[i]/den) < 1e-12, `max=${highest} card=${c}`));
    }
  }
});

test('actual engine bonus frequencies preserve 1/21 total and the window-first marginals', () => {
  const game = createGame(seed(12345));
  const counts = new Map([6,12,24,48].map(n => [n,0]));
  for (let i=0;i<84000;i++) {
    game.board = withRow([0,384,0,0]);
    const result = game.move('left');
    if (counts.has(result.value)) counts.set(result.value,counts.get(result.value)+1);
  }
  const total = [...counts.values()].reduce((a,b) => a+b,0);
  assert(Math.abs(total-4000) < 250, `bonus total ${total}`);
  // At max 384: windows [6,12,24],[12,24,48] -> marginals 1/6, 1/3, 1/3, 1/6.
  for (const [card,share] of [[6,1/6],[12,1/3],[24,1/3],[48,1/6]]) {
    const n = counts.get(card); assert(Math.abs(n-4000*share) < 170, `${card}: ${n}`);
  }
});

test('two 6144s make the final 12288 and end the game at once', () => {
  const game = createGame(seed(7));
  game.board = [[6144,6144,3,1],[2,6,12,24],[48,96,192,384],[768,1536,3072,1]];
  game.next = { bonus:false, candidates:[2] };
  const deck = [...game.deck], before = score(game.board);
  const result = game.move('left');
  assert.equal(result.changed, true); assert.equal(result.cleared, true);
  assert.equal(game.board[0][0], 12288);
  assert.equal(game.over, true); assert.equal(game.cleared, true); assert.equal(game.next, null);
  assert.equal(game.board.flat().filter(v => v === 2).length, 1, 'no card enters after 12288');
  assert.deepEqual(game.deck, deck);
  assert.equal(score(game.board) - before, 3**13 - 2*3**12);
  assert.equal(game.move('right').changed, false);
  game.reset(); assert.equal(game.over, false); assert.equal(game.cleared, false);
});
