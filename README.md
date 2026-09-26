# Threes Meadow AI · 合三

[Play in your browser](https://theonefkguy.github.io/threes-meadow-ai/) · **English** | [简体中文](README.zh-CN.md)

A high-performance Threes AI you can watch, interrupt and play alongside. Multi-stage temporal-difference learning meets five-ply expectimax with probability cutoffs. The model runs entirely on your device: no account, API key or inference server needed.

**39.1% reached 6144 · 3.0% reached 12288 · 1,024 fresh games.**

## Play

Open the [live demo](https://theonefkguy.github.io/threes-meadow-ai/), choose 中文 or English, and press **Play with AI**. Use arrow keys, swipe, or the direction buttons to take over. The default V23 model is about 8 MiB and loads on first use. Earlier strategies are available for exploration.

## V24 results

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

The current AI uses V4 stages 0–1, retrained V23 stage 2, and V21 stage 3. It observes the board, preview candidates and publicly reconstructable bag counts; it does not read the shuffled future deck. See [AI.md](AI.md) for history and [RULES.md](RULES.md) for rule details.

## License and attribution

[MIT](LICENSE), including this project's model weights. [Attributions](NOTICE.md). Independent research project; not affiliated with or endorsed by Threes! / Sirvo.
