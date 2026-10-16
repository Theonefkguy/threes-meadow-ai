# V39 endgame diagnostic (descriptive, no test)

Question: where do W31 games starting from the first 6144 fail under the cycle rule? Replays the
V38 W31 arm (2048 first-6144 states, rollout seeds 38400001+index, cycle-aware website search)
with milestone tracking (`diag.cpp`): a 3072; 3072+1536; 3072+1536+768 (final-768 chain cards,
plus whether 6144-3072-1536-768 lie on an adjacent path); "ready" (every card needed for 12288
already on the board). Every replayed game must equal its V38 record (success, moves, score).
Outputs (`analyze.py` -> `results.json`): P(12288) after each milestone, the last milestone of
each failed game and how long it lived after it, moves from "ready" and from the final-768 chain
to 12288, and P(12288) for chained vs unchained final-768 positions. Written before the replay.
