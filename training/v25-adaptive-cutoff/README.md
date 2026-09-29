# V25 adaptive cutoff: archived evidence

Run on 2026-09-29 (07:09–07:50 +08:00) in a separate Codex session and archived here unchanged. See [RL-V25-ADAPTIVE-CUTOFF.md](../../RL-V25-ADAPTIVE-CUTOFF.md) for the write-up; `REPORT.md` is the original generated report.

Files are byte-identical to the original output folder, with these omissions:

- `model.bin` and `browser/model.bin`: identical to `dist/models/ntuple-v23.bin` (SHA-256 `cdef81fd…`), not duplicated.
- `game-macos-arm64`, `browser-reference`: compiled binaries. `run.py` checks the binary hash recorded in `protocol.json`; to rerun elsewhere, compile `source/training/v24-acceptance/game.cpp` and record the new hash in a new protocol rather than editing this one.
- `data/`: the 256 per-chunk checkpoint files, which `run.py` merged into `games-*.jsonl`.

`source/` holds the exact evaluator sources. Only `source/training/v24-acceptance/game.cpp` differs from the repository copy: one line switches the threshold to the candidate value once the real board's maximum tile reaches 3072.

Check: copy this folder elsewhere, run `python3 analyze.py`, and compare. It regenerates `results.json` and `REPORT.md` byte-identically (verified at archive time).
