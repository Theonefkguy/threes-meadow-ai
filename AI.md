# Current release: V31 / 当前发布版

Default: V31 model (V4 stage 0; stages 1-3 = four-cell tables plus float16 five-cell residuals,
17 MB gzip download, 32 MB decoded, with a progress bar), five-ply probability-cutoff expectimax (0.01),
nonnegative leaf clamp and a 160 ms browser soft budget. Over 2048 fresh whole games the shipped
weights reach 6144 in 59.3% and 12288 in 5.6% (V23: 38.6% / 3.9% on the same openings). See
[README](README.md), [V31](RL-V31-STAGES.md) and training/web-release/v31-check. The sections
below are a chronological record: statements about earlier defaults describe old releases.
If V31 cannot load, the worker falls back to the V23 model.

当前默认采用 V31 模型（第 0 段沿用 V4，第 1–3 段为 4 格表加 16 位浮点 5 格残差，下载约 17 MB，解压后 32 MB）与五层剪枝搜索。网页只保留默认、第四版和经典搜索。下文为历史记录，早期默认配置以 README 为准。

# Automatic player

The AI button below the directional controls toggles play from the current board.
Manual arrows, a board touch, Escape, a new game, leaving the page, or hiding the
page stop it. A slide already committed finishes normally. It stops at game over
and never starts a fresh game by itself.

The module worker runs iterative-deepening expectimax (one to three player moves),
with a 160 ms soft time budget and a 24,000-node budget. Only complete search depths
are used; the one-move search always finishes to provide a legal fallback. Worker
failure or a watchdog stops automatic play without disabling manual play (10 seconds
for classic search, 30 seconds for learned mode including its first model load).
Stale replies are invalidated when stopping or restarting.

Search uses a flat rank board and cached row transitions. Equivalence tests check
its single-cell movement against the canonical game engine, which remains
unchanged. It averages over eligible entry lanes and the conditional weights of
visible bonus candidates. It observes nine starting cards and every subsequent
ordinary preview to maintain remaining bag counts; it does this during manual
play too, so starting AI mid-game has the full public history. Counts reset on a
new game, refill only when needed, and bonus previews do not consume the bag.
No shuffled deck or future order is sent to the worker. Search decrements the
counts in each hypothetical normal-draw branch. A caller without counts falls
back to long-run equal ordinary-card frequencies.

Bonus previews share the game's uniform marginal rank law and conditional
preview probabilities (see RULES.md). The selected evaluation favors ordered
merge chains and penalizes small cards between larger cards on both axes. It
uses a softer corner preference than the previous AI. `mode: classic`, `space`
and the optional four-move limit remain available to the benchmark harness.
The classic option uses the selected chain policy. The learned option uses a
separate afterstate search, adding immediate merge and spawn rewards exactly
once, then evaluating leaves with the three-stage N-tuple model. Model loads and
search happen in the worker; switching modes terminates old work. V2 model-load failure visibly falls back to V1; failure of both models
or a board outside the trained rank range visibly falls back to classic search.

Six policies were screened on six seeds, then the two highest geometric-mean
scorers were evaluated against the original on twelve new seeds under the
original full budget. See BENCHMARKS.md and benchmarks/*-results.json for scores,
per-game timing, limitations and reproduction commands. Search acceleration by
itself reproduced all six baseline pilot games exactly while taking less time.

The learned model is trained offline with temporal-difference learning; it is
not a language model or a guarantee of optimal play. Its static weights are
downloaded on first use. No API or server inference is used. See RL.md and
RL-RESULTS.md for historical V1 results, and RL-V2.md for the current training,
independent evaluation and limits. V2 is the default; V1 remains selectable.

Validation: node --test tests/*.test.mjs. Coverage includes legal search results,
visible-state immutability, multi-turn play, the start/stop loop, manual takeover,
backgrounding, stale replies, worker failure and terminal-state cleanup. DOM tests
use doubles and do not constitute mobile browser rendering validation.

V17 adds the cutoff search option (`rl-fast`, dist/fast-search.js): the same
public-information expectimax on the V4 model with a probability cutoff (decision
nodes whose root path probability is below the threshold use the one-move value).
V17 tested four moves with threshold 0.003. It keeps the 160 ms soft budget and complete-depth rule.
With threshold 0 it reproduces the complete search exactly. On 1024 paired
openings with a frozen model it reached 1536 in 98.24% vs 94.43% for complete
three-move search. See RL-V17-FAST-SEARCH.md.
V18 played the same searches to 6144 (1024 paired openings, frozen model whose
mid/late tables equal V4): complete three-move search reached 6144 in 13.09%,
the four-move cutoff search in 34.47% and a five-move cutoff search (0.01) in
43.36%. See RL-V18-LATE.md.

Since V18 the website default is "五层剪枝搜索" (`rl-fast`): five player moves,
cutoff 0.01, 160 ms soft budget, only complete depths used. Above 6144 it hands
over to the V4 search. V4 and the other versions remain selectable.
V19 compared deeper settings from 1024 real first-3072 positions (to 6144):
five moves/0.01 50.39%, six/0.01 50.88%, six/0.02 49.12%, five/0.005 53.81%;
no candidate beat the default after Holm correction, so the default is unchanged.
See RL-V19-DEPTH-SNAPSHOTS.md.

Since V20 the website and the C++ engine use the official-modern rules in RULES.md:
bonus window chosen uniformly, then the card uniformly within it, and 12288 ends
the game. All searches use the same bonus probabilities. V1-V19 results were
measured under the legacy rule (C++: THREES_BONUS_RULE=legacy).

Since V21 the default AI loads `ntuple-v21.bin`, a four-stage model: V4 stages 0-2
plus a stage for afterstates holding a 6144 or more, trained under the new rules.
From real first-6144 positions it made 12288 in 21/256 continuations vs 7/256 and
gained 66k more points on average. See RL-V21-LATE-TRAIN.md.
V22: the default AI clamps negative leaf estimates to 0 (`clampLeaf`), removing
certain-death moves that were chosen while a survivable move existed. The effect
on results is negligible; the retrained stage 2 remains not adopted. See RL-V22-CLAMP.md.
V23: the default AI loads `ntuple-v23.bin` (V4 stages 0-1, a retrained stage 2 from
on-policy starts with a strong-search share, V21 stage 3). From 2560 fresh first-3072
positions it made 6144 in 46.1% vs 41.1% (p=1.5e-4). See RL-V23-STAGE2.md.
V24 whole-game acceptance (official-modern rules, 1024 paired openings): the current
default AI reached 6144 in 39.1% and 12288 in 3.0% of games, versus 11.1% and 0% for
the website AI at the start of this work. See RL-V24-ACCEPTANCE.md.
V25 (not adopted): cutoff 0.005 once the board holds a 3072 left 6144 unchanged
(429 vs 426/1024) and gave 12288 in 43 vs 27 games (Holm p=0.14). See RL-V25-ADAPTIVE-CUTOFF.md.
V26 (not adopted): from 1024 fresh first-6144 positions, 12288 in 89 (default), 83 (five-ply
0.005) and 101 (six-ply 0.005) games; neither candidate passed Holm. See RL-V26-LATE-CUTOFF.md.
V27 (not adopted): stage 3 retrained with the V23 method (on-policy starts, strong sparring)
reached 12288 from 2048 fresh first-6144 positions in 107 vs 152 games for the current table
(McNemar p=0.0027); four times the training was worse still. See RL-V27-STAGE3.md.
V28 (exploratory): replaying V27 arm A with checkpoints showed the decline within 1e7 moves at
learning rate .02 (7.85% -> 5.5% reach 12288 over 3072 positions, ~4% thereafter); the learning
rate, not training length, is the main cause. See RL-V28-CURVE.md.
V29 (offline, not yet on the website): a zero-initialised five-cell residual (4 patterns) on
stage 2 raised first-3072 -> 6144 from 44.8% to 58.1% over 2048 fresh positions (McNemar
p=1.5e-18 vs the current model); the four-cell table alone, trained identically, did not
change. See RL-V29-FIVE.md.
V30 (offline, not yet on the website): 16-thread lock-free training for 1.2e10 moves; over 2048
fresh whole games the stage-2 residual raised reach-6144 from 39.7% to 52.7%, 12288 from 3.2% to
4.5% and mean score by ~70k (Holm-significant). See RL-V30-PARALLEL.md.
V31 (offline, not yet on the website): adding five-cell residuals to stages 1 and 3 on top of V30
raised, over 2048 fresh whole games, reach-3072 from 87.4% to 92.8%, 6144 from 55.8% to 58.9% and
12288 from 4.2% to 5.9% (P1-P3 Holm-significant); the stage-3 residual alone gave +1.0 pp (n.s.).
See RL-V31-STAGES.md.
V32 (not adopted): stage 3 fine-tuned only on strong-search episodes (3-ply or the website
search itself, 3 h each) did not raise first-6144 -> 12288 (9.4% vs 9.3%, p=0.96). The 12288 gains
of V30/V31 came from more games reaching 6144; P(12288 | 6144) stayed near 10%. See RL-V32-STRONG.md.
V33 (stopped at screening): refitting stage-3 residuals to realised scores, corner-focused five-cell
shapes beat the V29 shapes by only 1% of their gain over the four-cell table (pre-set threshold 5%);
eight five-cell or four six-cell patterns underfit with 3.6M records. See RL-V33-PATTERNS.md.
