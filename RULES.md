# Threes mechanics implemented in 合三

This is an independent implementation using public descriptions and third-party
reconstructions, not official game source. It targets the basic nine-card opening
and the later candidate-set preview. Original artwork, audio and character assets
are not used. Unlockable high-card starts are not implemented.

## Rule references

- https://github.com/nneonneo/threes-ai/tree/c554820ad848a73b7cd34e358c974ab2d85bd792
  - `threes.h`: four copies each of 1, 2, 3; HIGH_CARD_FREQ = 21.
  - `threes.cpp`: `initial_board`, `draw_deck`, `play_game`, `insert_tile`.
  - `README.md`: Entering tiles describes historical color, plus-sign, and newer
    candidate-set previews. It does not establish complete current-app fidelity.
- https://github.com/rilka/threesjs/blob/master/client/js/game.js
  - `generate_new_board` and `insert_new_tile`: stepwise moves and eligible lanes.
- https://github.com/nneonneo/threes-ai/issues/17
  - The author describes observational reconstruction; a comment's suggested
    repeated-direction spawn-lane behavior is unverified and not implemented.
- https://apps.apple.com/us/app/threes/id779157948
  - Official update history documents changes to previews and high-card starts.

## Evidence for the official-modern baseline (checked 2026-09-25)

Verified against primary sources by us:

- nneonneo/threes-ai C++ (`threes.cpp` `play_game`): 1/21 bonus trials at max >= 48,
  window chosen uniformly, card uniform within window, bonus does not use the deck.
  Its Python `threes.py` still uses 1/24 with a TODO saying bonus generation was not
  updated to the "pick-three" implementation; treated as a historical snapshot.
- Yeh et al. 2016: 12-card bag, 1/21 bonus, bonus 6..max/8 all equally likely,
  6144 cannot merge; no preview is described. Used here only as a legacy ruleset.
- Wikipedia (Threes): two 6144s reveal a final 12288 character and the game ends
  immediately, first documented in 2017.
- Movement: our row rule equals nneonneo's `find_fold` (first fold only, rest shift)
  on all 15^4 row states.

Additional primary community source verified on 2026-09-26:
[Kamikaze28 & ekisacik, Basic & Advanced Secrets of Threes! (2024)](https://steamcommunity.com/sharedfiles/filedetails/?id=3155431026).
The guide explicitly documents the balanced 12-card deck, nine-card standard opening,
1/21 bonus frequency, equal-probability candidate windows with uniform cards within
each window, and uniform placement among moved rows/columns. This independently
supports the modern core-rule implementation. Optional boosted starts are not included.
This is community reverse engineering, not certification by the official developer.

## Implemented decisions

- Draw nine starting cards without replacement from a shuffled 12-card deck.
  The first preview is the tenth draw, leaving two cards in the same deck.
- Ordinary cards exhaust the deck before another balanced deck is shuffled.
- At maximum tile >= 48, each new preview has a 1/21 chance to be a bonus.
  A bonus does not consume the ordinary deck. These are random trials, not a
  guaranteed interval of 21 moves.
- Eligible bonuses are 6, 12, 24, ... through maximum/8 (48 -> {6}, 96 -> {6,12},
  192 -> {6,12,24}, 384 -> {6,12,24,48}, ...).
- Bonus card (official-modern rule, default since V20): choose uniformly among
  the legal windows of up to three consecutive values, show that window, then
  draw the card uniformly within it. Middle values are therefore more likely
  than the extremes; at max=768 the marginals of 6/12/24/48/96 are [1,2,3,2,1]/9,
  at max=384 the marginals of 6/12/24/48 are [1,2,2,1]/6.
  This is also how nneonneo/threes-ai `play_game` draws bonuses (C++ version).
  It replaces the V1-V19 rule, which gave every legal value equal marginal
  probability (Yeh et al. 2016) and derived the window from the drawn value.
  The C++ engine keeps that rule as `THREES_BONUS_RULE=legacy` so earlier
  results can be reproduced; the website uses only the new rule.
- Combining two 6144s makes 12288, the final card. The game ends immediately:
  no card enters and no further preview is drawn. Searches treat that move as
  terminal (its merge points only).
- Preview exposes only candidates, including to screen readers. Dragging and
  canceling cannot resample either a card or an entry location.
- Traverse from the leading edge; each nonempty card can advance one cell.
  1+2 and 2+1 merge; equal cards >=3 merge. No same-turn cascade.
- One card enters the opposite edge, uniformly among lanes that actually moved.
  A blocked swipe consumes no card, turn or random number.
- Otherwise the game ends only when all four directions are blocked. Points for 3,6,12,...
  are 3,9,27,... respectively; 1 and 2 score zero.

## Input behavior

Dragging previews tile motion only. Returning near the starting point cancels.
Release beyond the threshold commits the direction and animates the slide before
inserting the new tile. Escape, pointer cancellation, lost capture and focus loss
cancel a preview. New game cancels any pending animation/terminal callback.
