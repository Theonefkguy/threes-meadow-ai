# Current release: V24 / 当前发布版

Default: V23 four-stage weights, five-ply probability-cutoff expectimax (0.01),
nonnegative leaf clamp and a 160 ms browser soft budget. See [README](README.md)
and [V24 acceptance](RL-V24-ACCEPTANCE.md). The sections below are a chronological
record: early statements about three-ply search and V2 defaults describe old releases.

当前默认采用 V23 四阶段模型与五层剪枝搜索。下文为历史记录，早期默认配置以 README 为准。

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
