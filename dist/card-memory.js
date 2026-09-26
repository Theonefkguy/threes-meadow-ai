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

