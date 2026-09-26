import { chooseMove as searchMove } from './ai-search.js?v=20';

// Selected after the six-policy pilot and independent-seed validation.
// Keep the original 160 ms / 24,000-node / three-move limits.
export function chooseMove(board, next, options = {}) {
  return searchMove(board, next, { mode:'chain', remember:true, ...options });
}
