# Threes Meadow AI

[Play in your browser](https://theonefkguy.github.io/threes-meadow-ai/) · **English** | [简体中文](README.zh-CN.md)

A high-performance Threes AI you can watch, interrupt and play alongside. Multi-stage temporal-difference learning of four- and five-cell n-tuple tables meets five- to six-ply expectimax with probability cutoffs. The model runs entirely on your device: no account, API key or inference server needed.

**67.9% reached 6144 · 9.9% reached 12288 · 4,096 games of the website AI under the official bonus cycle.**

## Play

Open the [live demo](https://theonefkguy.github.io/threes-meadow-ai/), use the English default or switch to 中文, and press **Play with AI**. Use arrow keys, swipe, or the direction buttons to take over. The default V31 model is a 17 MB gzip download (32 MB decoded) fetched on first use, with a progress bar. A full-precision option (30 MB download) uses the 5-cell tables at their original 32-bit precision instead of 16-bit; an earlier paired test (old bonus rule, five-ply) found no significant difference between the two precisions. The V4 search and a classic heuristic search remain selectable for comparison.

## Video

[![How a Threes AI Actually Thinks](https://i.ytimg.com/vi/Rov_8bpDScY/hqdefault.jpg)](https://www.youtube.com/watch?v=Rov_8bpDScY)

Watch **How a Threes AI Actually Thinks** on YouTube.

## V41: deeper search after 6144

Once a 6144 is on the board, the website AI now searches 6 moves ahead with finer pruning (cutoff 0.005) instead of 5 (cutoff 0.01), with up to 400 ms per move. In 4,096 paired whole games (shipped float16 weights, official bonus cycle, identical until the first 6144):

| Metric | 5-ply after 6144 | 6-ply after 6144 (website) |
|---|---:|---:|
| Reach 6144 | 67.9% | 67.9% |
| 6144+3072+1536+768 on the board | 24.2% | 30.7% |
| Reach 12288 | 6.5% | **9.9%** |
| 12288 after 6144 | 9.5% | **14.5%** |
| Mean final score | 624,319 | **659,680** |

The 12288 gain is +3.4 pp (exact McNemar p = 1.4e-9). Depth was the main part (6-ply alone: +3.4 pp from first-6144 positions; finer pruning alone: +1.1 pp, n.s.), but the two together worked best. A desktop browser needs 29 ms per move on average in this phase (P95 62 ms); slower devices fall back to the completed 5-ply result if 6 plies do not finish in time. JS and C++ choose identical moves on 320 positions at this setting. Earlier attempts to improve the stage-3 evaluation itself (V27–V38) did not help; a diagnostic (V39) showed that most losses happen while building the last 768, not in the final merges.

[V41 report (Chinese)](RL-V41-STAGE3-DEPTH.md) · [V40](RL-V40-LATE-DEPTH.md) · [V39 diagnostic](RL-V39-ENDGAME.md)

## V35: the official bonus cycle

The official app gives exactly one bonus in every block of 21 previews, not an independent 1/21 chance per preview. We confirmed this by letting the AI play two complete games of the iPhone app (148 complete blocks, every one with exactly one bonus; see [RULES.md](RULES.md)). The website engine now uses this rule, and the default AI tracks the cycle from the previews it has seen: no bonus once this block's bonus has appeared, otherwise 1/(21 − positions already ruled out).

4,096 whole games per arm on the same seeds, V31 model, five-ply website search:

| Metric | Old rule (V31 release, 2,048 games) | Cycle rule, AI unaware | Cycle rule, AI aware |
|---|---:|---:|---:|
| Reach 1536 | 98.6% | 99.4% | 99.3% |
| Reach 3072 | 92.9% | 95.8% | 96.0% |
| Reach 6144 | 59.3% | 65.2% | **66.5%** |
| Reach 12288 | 5.6% | 6.0% | **7.2%** |
| Mean final score | 561,463 | 604,132 | **622,923** |

Most of the gain comes from the rule itself (bonuses arrive evenly, never more than 41 previews apart). Knowing the cycle adds +18.8k mean score (Holm p = 0.041); the 6144 (+1.3 pp) and 12288 (+1.1 pp) gains are not significant on their own. The shipped float16 weights scored 65.1% / 6.7% in a separate 2,048-game check (not significantly different). JS and C++ choose identical moves on 320 positions with and without the cycle, and the cycle-aware search is not slower.

[V35 report (Chinese)](RL-V35-CYCLE.md) · [Release check](training/web-release/v35-check/)

## V31 release results

Old independent 1/21 bonus rule; otherwise same rules and engine as below. Native C++ evaluation of the exact weights shipped to the browser (float16 residuals), from standard openings until 12288 or death, on the same 2,048 opening seeds for both arms.

| Metric | Previous website AI (V23) | V31 (shipped) |
|---|---:|---:|
| Reach 3072 | 87.5% | 92.9% |
| Reach 6144 | 790/2048 (38.6%) | **1215/2048 (59.3%)** |
| Reach 12288 | 79/2048 (3.9%) | **115/2048 (5.6%)** |
| Mean final score | 438,894 | **561,463** |

95% Wilson intervals for V31: 6144 **57.2–61.4%**; 12288 **4.7–6.7%**. All three primary comparisons pass Holm correction (6144 p = 1e-44, 12288 p = 0.009, score p = 3e-33). The 12288 gain comes from more games reaching 6144: once a 6144 exists, both AIs reach 12288 about 10% of the time (V23 79/790, V31 115/1215), and stage-3 experiments so far (V27, V31, [V32](RL-V32-STRONG.md)) have not improved that rate. The gain comes from five-cell residual tables added to stages 1–3 (V29–V31): a four-cell table trained further did not improve, while the larger patterns did. JS and C++ choose identical moves on 320 sampled positions; desktop Chromium needs 5.4 ms per move on average (P95 9.0 ms).

[V31 report (Chinese)](RL-V31-STAGES.md) · [Release checks](training/web-release/v31-check/) · [V29 five-cell test](RL-V29-FIVE.md) · [V30 parallel training](RL-V30-PARALLEL.md)

## V24 results (previous release)

Standard nine-card openings under modern Threes core rules documented by community reverse engineering. Native C++ evaluation, from the opening until 12288 or death, on the same 1,024 opening seeds for each arm.

| Metric | Earlier project AI | V23 + five-ply search |
|---|---:|---:|
| Reach 1536 | 93.8% | 98.8% |
| Reach 3072 | 65.4% | 86.2% |
| Reach 6144 | 114/1024 (11.1%) | **400/1024 (39.1%)** |
| Reach 12288 | 0/1024 | **31/1024 (3.0%)** |
| Mean final score | 227,400 | **432,649** |

95% Wilson intervals: 6144 **36.1–42.1%**; 12288 **2.1–4.3%**. The three primary comparisons against the earlier project AI pass Holm correction. This is an internal baseline comparison, not a uniform benchmark against all public AIs.

[Evaluation and reproduction](docs/REPRODUCING.md) · [Original V24 report (Chinese)](RL-V24-ACCEPTANCE.md) · [Protocol](training/v24-acceptance/protocol.json) · [Raw NEW games](training/v24-acceptance/games-NEW.jsonl) · [Public comparisons](docs/COMPARISONS.md)

The browser uses a 160 ms soft search budget; V24's native NEW evaluation completes five plies on every move without that time limit. Slower devices may complete fewer plies. Results were measured in the included engine, not through the official app. Training experiments stopped by CPU time may not reproduce identical weights on different hardware; the evaluated weights and hashes are included.

## Run locally

Requires Python 3 for a static server; Node.js 22 is recommended for tests. No npm packages or build step are required.

```sh
git clone https://github.com/Theonefkguy/threes-meadow-ai.git
cd threes-meadow-ai
python3 -m http.server 8000 --bind 127.0.0.1 --directory dist
```

Open http://localhost:8000. Use HTTP rather than opening index.html as a file, so module workers and model loading work.

```sh
node --test tests/*.test.mjs
python3 scripts/verify-release.py
```

## Repository

- `dist/`: editable browser JavaScript, styles, game and model weights.
- `training/`: training/evaluation implementations, protocols and archived experiment data.
- `tests/`: engine, model, search, input and AI lifecycle tests.
- `docs/`: reproducibility, comparison scope and publishing instructions.
- `.github/workflows/pages.yml`: tests and checks before publishing only `dist/` to Pages.

The current AI (V31) uses the V4 stage-0 table and, for stages 1–3 (max card 1536, 3072, 6144+), four-cell tables plus five-cell residual tables trained by parallel TD learning (V30/V31). It observes the board, preview candidates and publicly reconstructable bag counts; it does not read the shuffled future deck. See [AI.md](AI.md) for history and [RULES.md](RULES.md) for rule details.

## License and attribution

[MIT](LICENSE), including this project's model weights. [Attributions](NOTICE.md). Independent research project; not affiliated with or endorsed by Threes! / Sirvo.
