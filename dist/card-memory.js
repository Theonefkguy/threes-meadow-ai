import { bonusPreviews, CYCLE } from './engine.js?v=28';

// Public-card accounting: never reads the shuffled deck or its order.
export function initialCounts(board,next) {
  const counts=[4,4,4];
  for(const n of board.flat()) if(n>=1 && n<=3) counts[n-1]--;
  return observePreview(counts,next);
}
export function observePreview(counts,next) {
  const result=counts.slice();
  if(next && !next.bonus) {
    if(result.every(n=>n===0)) result.fill(4);
    result[next.candidates[0]-1]--;
  }
  return result;
}


// Public bonus-cycle state (see engine.js) for the draw after the next move: its position in the
// block, whether the block's bonus has been shown, and how many earlier positions of the block
// are excluded (a bonus was possible there but not shown). Built from previews only, never from
// the hidden slot. Start of a game: draw 1 is next, at position 0.
export const initialCycle = () => ({ pos:0, used:0, excl:0 });
export function observeCycle(cycle,next,board) {
  if(!next) return cycle;
  const possible=bonusPreviews(Math.max(...board.flat())).length>0;
  const result={ ...cycle, pos:cycle.pos+1 };
  if(next.bonus) result.used=1; else if(possible) result.excl++;
  return result.pos===CYCLE ? initialCycle() : result;
}
