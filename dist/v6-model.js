// Stage 0 stores logits only for a reach1536 policy. Stages 1/2 stay V4 scores.
export function reachProbability(base,board) {
  if(Math.max(...board)>=12)return 1;
  const z=base.value(board,0);
  return z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));
}
