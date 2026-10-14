# Threes Meadow AI

[Play in your browser](https://theonefkguy.github.io/threes-meadow-ai/) · **English** | [简体中文](README.zh-CN.md)

A high-performance Threes AI you can watch, interrupt and play alongside. Multi-stage temporal-difference learning of four- and five-cell n-tuple tables meets five-ply expectimax with probability cutoffs. The model runs entirely on your device: no account, API key or inference server needed.

**59.3% reached 6144 · 5.6% reached 12288 · 2,048 fresh games.**

## Play

Open the [live demo](https://theonefkguy.github.io/threes-meadow-ai/), use the English default or switch to 中文, and press **Play with AI**. Use arrow keys, swipe, or the direction buttons to take over. The default V31 model is about 32 MB and downloads on first use, with a progress bar. The V4 search and a classic heuristic search remain selectable for comparison.

## Video

[![How a Threes AI Actually Thinks](https://i.ytimg.com/vi/Rov_8bpDScY/hqdefault.jpg)](https://www.youtube.com/watch?v=Rov_8bpDScY)

Watch **How a Threes AI Actually Thinks** on YouTube.

## V31 release results

Same rules and engine as below. Native C++ evaluation of the exact weights shipped to the browser (float16 residuals), from standard openings until 12288 or death, on the same 2,048 opening seeds for both arms.

| Metric | Previous website AI (V23) | V31 (shipped) |
|---|---:|---:|
| Reach 3072 | 87.5% | 92.9% |
| Reach 6144 | 790/2048 (38.6%) | **1215/2048 (59.3%)** |
| Reach 12288 | 79/2048 (3.9%) | **115/2048 (5.6%)** |
| Mean final score | 438,894 | **561,463** |

95% Wilson intervals for V31: 6144 **57.2–61.4%**; 12288 **4.7–6.7%**. All three primary comparisons pass Holm correction (6144 p = 1e-44, 12288 p = 0.009, score p = 3e-33). The gain comes from five-cell residual tables added to stages 1–3 (V29–V31): a four-cell table trained further did not improve, while the larger patterns did. JS and C++ choose identical moves on 320 sampled positions; desktop Chromium needs 5.4 ms per move on average (P95 9.0 ms).

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
