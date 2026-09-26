// Independently implemented from the public Threes rule descriptions.
// References and reconstruction limits: ../RULES.md.
export const DIRECTIONS = ['left', 'right', 'up', 'down'];
const BASE_DECK = [1,1,1,1,2,2,2,2,3,3,3,3];
const randomIndex = (length, rng) => Math.floor(rng() * length);
function weightedIndex(weights, rng) {
  const roll = rng();
  let sum = 0;
  for (let i = 0; i < weights.length; i++) {
    sum += weights[i];
    if (roll < sum) return i;
  }
  return weights.length - 1;
}

export const cardProbabilities = preview => preview.probabilities || preview.candidates.map(() => 1 / preview.candidates.length);

export function shuffled(items, rng = Math.random) {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = randomIndex(i + 1, rng);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

export const canMerge = (a, b) => a > 0 && b > 0 &&
  ((a === 1 && b === 2) || (a === 2 && b === 1) || (a >= 3 && a === b));

// Pure projection: previews must never draw cards, consume randomness or alter state.
export function projectMove(board, direction) {
  if (!DIRECTIONS.includes(direction)) throw new RangeError('Invalid direction');
  const state = board.map(row => [...row]);
  const moves = [], merged = [], lanes = [];
  for (let lane = 0; lane < 4; lane++) {
    const positions = Array.from({ length: 4 }, (_, step) => {
      if (direction === 'left') return [lane, step];
      if (direction === 'right') return [lane, 3 - step];
      if (direction === 'up') return [step, lane];
      return [3 - step, lane];
    });
    let changed = false;
    for (let step = 1; step < 4; step++) {
      const [y,x] = positions[step], [ty,tx] = positions[step - 1];
      const value = state[y][x], target = state[ty][tx];
      if (!value || (target && !canMerge(value, target))) continue;
      state[ty][tx] = value + target;
      state[y][x] = 0;
      changed = true;
      moves.push({ from: y * 4 + x, to: ty * 4 + tx });
      if (target) merged.push(ty * 4 + tx);
    }
    if (changed) lanes.push(lane);
  }
  return { board: state, moves, merged, lanes, changed: lanes.length > 0 };
}

export function entryCell(direction, lane) {
  if (direction === 'left') return [lane, 3];
  if (direction === 'right') return [lane, 0];
  if (direction === 'up') return [3, lane];
  return [0, lane];
}

export const hasMoves = board => DIRECTIONS.some(dir => projectMove(board, dir).changed);
export const score = board => Math.round(board.flat().reduce((sum, n) =>
  sum + (n < 3 ? 0 : 3 ** (Math.log2(n / 3) + 1)), 0));

export function bonusGroups(highest) {
  const values = [];
  for (let n = 6; n <= highest / 8; n *= 2) values.push(n);
  if (!values.length) return [];
  if (values.length <= 3) return [values];
  return Array.from({ length: values.length - 2 }, (_, i) => values.slice(i, i + 3));
}

// Official-modern bonus rule (see RULES.md): every legal window of up to three
// consecutive values is equally likely, then the card is uniform within the
// shown window. Middle values are therefore more common than the extremes.
export function bonusPreviews(highest) {
  const groups = bonusGroups(highest);
  return groups.map(candidates => ({
    candidates, probability: 1 / groups.length,
    probabilities: candidates.map(() => 1 / candidates.length),
  }));
}

// Combining two 6144s makes 12288, the final card: the game ends at once.
export const FINAL_CARD = 12288;
export const isCleared = board => board.some(row => row.includes(FINAL_CARD));

export function createGame(rng = Math.random) {
  const game = { board: [], deck: [], next: null, turns: 0, over: false };
  const drawNormal = () => {
    if (!game.deck.length) game.deck = shuffled(BASE_DECK, rng);
    return game.deck.pop();
  };
  const drawUpcoming = () => {
    const previews = bonusPreviews(Math.max(...game.board.flat()));
    if (previews.length && rng() < 1 / 21) {
      const { candidates, probabilities } = previews[weightedIndex(previews.map(p => p.probability), rng)];
      return { bonus: true, candidates, probabilities };
    }
    return { bonus: false, candidates: [drawNormal()] };
  };
  game.reset = () => {
    game.deck = shuffled(BASE_DECK, rng);
    const opening = Array.from({ length: 9 }, drawNormal);
    const cells = shuffled([...opening, ...Array(7).fill(0)], rng);
    game.board = Array.from({ length: 4 }, (_, y) => cells.slice(y * 4, y * 4 + 4));
    game.turns = 0;
    game.over = false;
    game.cleared = false;
    game.next = drawUpcoming(); // Continues the three cards left in the opening deck.
    return game;
  };
  game.move = direction => {
    const projected = projectMove(game.board, direction);
    if (game.over || !projected.changed) return { changed: false };
    if (isCleared(projected.board)) {
      // Final card: no new card enters and no further preview is drawn.
      game.board = projected.board; game.turns++; game.over = true; game.cleared = true; game.next = null;
      return { ...projected, cleared: true };
    }
    const lane = projected.lanes[randomIndex(projected.lanes.length, rng)];
    const [y, x] = entryCell(direction, lane);
    const candidates = game.next.candidates;
    const value = candidates.length === 1 ? candidates[0] : candidates[weightedIndex(cardProbabilities(game.next), rng)];
    if (projected.board[y][x] !== 0) throw new Error('Occupied entry cell');
    projected.board[y][x] = value;
    game.board = projected.board;
    game.turns++;
    game.over = !hasMoves(game.board);
    // No phantom draw after the terminal move.
    game.next = game.over ? null : drawUpcoming();
    return { ...projected, spawned: y * 4 + x, value, lane };
  };
  return game.reset();
}
