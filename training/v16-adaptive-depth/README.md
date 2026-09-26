# V16 adaptive-depth pilot

This paired pilot tests the user's compute-allocation hypothesis: use depth3 with at least6 empty cells, depth4 with4-5, and complete depth5 only with3 or fewer. Sixteen frozen first-768 snapshots, two matched rollout seeds, and two arms produce64 continuations to3072/death.

The gate was calibrated on96 sampled late-game decisions. The threshold policy skipped depth5 on25/96 decisions and disagreed with complete depth5 on5/96. A paired outcome pilot is still required because action agreement alone does not prove equal survival.

Every task atomically checkpoints full game and RNG state every10 moves. Run from the repository root with `python training/v16-adaptive-depth/run.py`.
