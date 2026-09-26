import {phaseForRanks} from './rl-model.js?v=20';

export function cornerPotential(board) {
  const max=Math.max(...board);
  return max>=13 && [0,3,12,15].some(i=>board[i]===max) ? 1 : 0;
}

// Stored weights estimate shaped returns U. Search uses original game rewards,
// so its leaves must restore V = U + coefficient * potential exactly once.
export function withCornerPotential(base,coefficient) {
  if(!Number.isFinite(coefficient)||coefficient<0)throw new Error('角落策略参数无效');
  return {value(board,stage=phaseForRanks(board)) {
    return base.value(board,stage)+(stage===2?coefficient*cornerPotential(board):0);
  }};
}
