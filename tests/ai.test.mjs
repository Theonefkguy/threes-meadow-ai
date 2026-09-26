import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseMove } from '../dist/ai.js';
import { createGame, projectMove } from '../dist/engine.js';

test('AI respects visible bonus candidates, preserves inputs and returns legal moves under a tight budget', () => {
  const game = createGame(() => .5);
  const next = { bonus:true,candidates:[6,12,24] };
  const before = JSON.stringify([game.board,next]);
  const result = chooseMove(game.board,next,{budgetMs:0,maxNodes:1});
  assert(projectMove(game.board,result.direction).changed);
  assert.equal(result.depth,1);
  assert.equal(JSON.stringify([game.board,next]),before);
});

test('AI reports no move on a terminal board', () => {
  const board = [[3,6,12,24],[6,12,24,48],[12,24,48,96],[24,48,96,192]];
  assert.equal(chooseMove(board,{candidates:[1]}).direction,null);
});

test('AI can complete a legal multi-turn simulation', () => {
  let seed = 17;
  const rng = () => ((seed = (Math.imul(seed,1664525) + 1013904223) >>> 0) / 4294967296);
  const game = createGame(rng);
  for (let step = 0; step < 100 && !game.over; step++) {
    const choice = chooseMove(game.board,game.next,{maxDepth:2,budgetMs:20,maxNodes:6000});
    assert(choice.depth >= 1);
    assert(game.move(choice.direction).changed);
  }
  assert(game.turns > 20);
});

test('AI uses the conditional weights of a visible bonus preview', () => {
  for (let seed=1;seed<=12;seed++) {
    let state=seed;
    const game=createGame(()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/4294967296));
    const weighted=chooseMove(game.board,{candidates:[6,12,24],probabilities:[0,0,1]},{maxDepth:1});
    const certain=chooseMove(game.board,{candidates:[24]},{maxDepth:1});
    assert.equal(weighted.direction,certain.direction);
  }
});
