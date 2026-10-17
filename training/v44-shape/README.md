# V44 shape replay (descriptive)

Question from Discord: does the AI keep 6144+3072+1536+768 in a 2x2 square, or another formation?
Replays the 970 V40 arm-S games (6-ply / 0.005 from the first 6144, W31, cycle rule; V38 pool and
rollout seeds) that reach the final-768 chain cards; each replay must equal its V40 record. Board
snapshots: at the first final-768 move and every 5 moves after it while all four cards are on the
board. `analyze.py` classifies each snapshot (2x2 square in order, straight line, L/snake chain,
2x2 out of order, touching, apart) and reports shares and P(12288) per shape. Written before the run.
