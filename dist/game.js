import { t, locale, setLanguage, applyLanguage } from './i18n.js';
import { createGame, projectMove, score } from './engine.js?v=20';
import { initialCounts, observePreview } from './card-memory.js?v=20';

const boardEl = document.querySelector('#board');
const nextEl = document.querySelector('#next-tile');
const nextHint = document.querySelector('#next-hint');
const scoreEl = document.querySelector('#score');
const bestEl = document.querySelector('#best');
const overEl = document.querySelector('#game-over');
const liveEl = document.querySelector('#move-status');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const game = createGame();
let publicCounts = initialCounts(game.board, game.next);
const aiButton = document.querySelector('#ai-toggle');
const aiStatus = document.querySelector('#ai-status');
const aiPolicy = document.querySelector('#ai-policy');
let aiRunning = false, aiWorker = null, aiTimer = null, aiWatchdog = null, aiRequest = 0;
const directionNames = { left:'左', right:'右', up:'上', down:'下' };

function stopAI(message = t('已停止，可以手动继续。')) {
  aiRunning = false;
  aiRequest++;
  clearTimeout(aiTimer); clearTimeout(aiWatchdog);
  aiTimer = aiWatchdog = null;
  aiWorker?.terminate(); aiWorker = null;
  aiButton.textContent = t('AI 自动玩');
  aiButton.setAttribute('aria-pressed', 'false');
  aiStatus.textContent = message;
}

function scheduleAI() {
  if (!aiRunning) return;
  if (game.over) { stopAI(t('本局结束。点击新一局后，可以再次开启 AI。')); return; }
  clearTimeout(aiTimer);
  aiTimer = setTimeout(() => {
    aiTimer = null;
    if (!aiRunning) return;
    if (busy) { scheduleAI(); return; }
    aiStatus.textContent = aiPolicy.value.startsWith('rl')?t('强化学习 AI 正在思考…'):t('AI 正在思考…');
    const id = ++aiRequest;
    aiWatchdog = setTimeout(() => stopAI(t('加载或思考超时，已停止。可以重试或手动继续。')), aiPolicy.value.startsWith('rl')?30000:10000);
    // Send visible state only, never game.deck or the random generator.
    aiWorker.postMessage({ id, board:game.board, next:game.next, remaining:publicCounts, policy:aiPolicy.value });
  }, 230);
}

aiButton.addEventListener('click', () => {
  if (aiRunning) { stopAI(); return; }
  if (game.over) { aiStatus.textContent = t('本局已结束，请先开始新一局。'); return; }
  cancelPreview();
  try {
    aiWorker = new Worker('./ai-worker.js?v=20', { type:'module' });
    aiWorker.onmessage = ({ data }) => {
      if (!aiRunning || data.id !== aiRequest) return;
      clearTimeout(aiWatchdog); aiWatchdog = null;
      if (data.error || !directionNames[data.direction] || !projectMove(game.board,data.direction).changed) {
        stopAI(t('AI 暂时无法继续，可以手动操作或重新开启。')); return;
      }
      if(data.fallback) {
        aiPolicy.value=['rl-fast','rl-v8','rl-v7','rl-v6','rl-v5','rl-v4','rl-v3','rl-v2','rl'].includes(data.policy)?data.policy:'classic';
        aiStatus.textContent=data.fallback==='fast-load'?t('五层剪枝搜索暂时无法加载，已切换第四版。'):data.fallback==='v8-load'?t('蒙特卡洛策略暂时无法加载，已切换第四版。'):data.fallback==='v7-load'?t('第七版暂时无法加载，已切换第四版。'):data.fallback==='v6-load'?t('第六版暂时无法加载，已切换第四版。'):data.fallback==='v5-load'?t('第五版暂时无法加载，已切换第四版。'):data.fallback==='v4-load'?t('第四版暂时无法加载，已切换第三版。'):data.fallback==='v3-load'?t('第三版暂时无法加载，已切换第二版。'):data.fallback==='v2-load'?t('第二版暂时无法加载，已切换第一版。'):data.fallback==='load'?t('模型暂时无法加载，已切换经典搜索。'):t('已超过模型训练范围，切换经典搜索。');
      } else {
        const phase=data.policy?.startsWith('rl')?t(' · {stage}策略', {stage:[t('前期'),t('中期'),t('后期'),t('冲刺 12288')][data.stage]}):'';
        aiStatus.textContent=t('AI 向{direction}移动{phase} · 已完成 {turns} 步',{direction:t(directionNames[data.direction]),phase,turns:game.turns+1});
      }
      commitMove(data.direction);
    };
    const activeWorker = aiWorker;
    aiWorker.onerror = () => {
      if (aiWorker === activeWorker) stopAI(t('AI 加载失败，可以重试或手动继续。'));
    };
    aiRunning = true;
    aiButton.textContent = t('停止 AI');
    aiButton.setAttribute('aria-pressed', 'true');
    aiStatus.textContent = aiPolicy.value.startsWith('rl')?t('正在加载强化学习模型…'):t('AI 已接手，正在观察棋盘…');
    scheduleAI();
  } catch { stopAI(t('当前浏览器无法启动 AI，可以继续手动玩。')); }
});
aiPolicy.addEventListener('change',()=>{
  if(aiRunning)stopAI(t('策略已切换，点击 AI 自动玩继续。'));
  else aiStatus.textContent=t('策略已切换，点击 AI 自动玩从当前棋盘开始。');
});
let best = 0;
try { best = Number(localStorage.getItem('sumGardenBest')) || 0; } catch { /* Storage is optional. */ }
let cells = [], tiles = [];
let pointer = null, busy = false, animationTimer = null;
const format = n => new Intl.NumberFormat(locale()).format(n);
const tileClass = n => n === 1 ? 'tile-1' : n === 2 ? 'tile-2' : 'tile-white';

function renderNext() {
  nextEl.replaceChildren();
  if (!game.next) {
    nextHint.textContent = t('本局结束');
    nextEl.setAttribute('aria-label', t('本局结束'));
    return;
  }
  const { bonus, candidates } = game.next;
  nextHint.textContent = bonus ? (candidates.length > 1 ? t('其中一张') : t('奖励牌')) : t('下一张');
  nextEl.classList.toggle('bonus-preview', bonus);
  nextEl.setAttribute('aria-label', t('下一张{kind} {cards}',{kind:candidates.length>1?t('可能是'):t('是'),cards:candidates.join(locale()==='en'?', ':'、')}));
  candidates.forEach(n => {
    const tile = document.createElement('span');
    tile.className = `preview-card ${tileClass(n)}`;
    tile.textContent = String(n);
    nextEl.append(tile);
  });
}

function render(effects = {}) {
  aiButton.disabled = game.over;
  const currentScore = score(game.board);
  if (currentScore > best) {
    best = currentScore;
    try { localStorage.setItem('sumGardenBest', String(best)); } catch { /* Keep playing. */ }
  }
  scoreEl.textContent = format(currentScore);
  bestEl.textContent = format(best);
  renderNext();
  boardEl.replaceChildren();
  cells = []; tiles = [];
  game.board.forEach((row, y) => {
    const rowEl = document.createElement('div');
    rowEl.className = 'board-row';
    rowEl.setAttribute('role', 'row');
    row.forEach((value, x) => {
      const index = y * 4 + x;
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.setAttribute('role', 'gridcell');
      cell.setAttribute('aria-label', t('第{row}行，第{col}列：{value}',{row:y+1,col:x+1,value:value||t('空')}));
      cells[index] = cell;
      if (value) {
        const tile = document.createElement('div');
        tile.className = `tile ${tileClass(value)}${value >= 1000 ? ' tile-large' : ''}`;
        if (effects.spawned === index) {
          tile.classList.add('arrive');
          const offsets = { left:[45,0], right:[-45,0], up:[0,45], down:[0,-45] };
          const [dx,dy] = offsets[effects.direction] || [0,0];
          tile.style.setProperty('--enter-x', `${dx}%`);
          tile.style.setProperty('--enter-y', `${dy}%`);
        }
        if (effects.merged?.includes(index)) tile.classList.add('merge');
        tile.textContent = String(value);
        cell.append(tile);
        tiles[index] = tile;
      }
      rowEl.append(cell);
    });
    boardEl.append(rowEl);
  });
  overEl.hidden = !game.over;
  if (game.over) {
    document.querySelector('#final-score').textContent = format(currentScore);
    const kicker = document.querySelector('#over-kicker'), title = document.querySelector('#over-title');
    if (kicker) kicker.textContent = game.cleared ? t('合出 12288') : t('无路可走');
    if (title) title.textContent = game.cleared ? t('通关！两张 6144 合成了最终的 12288') : t('这一局走到了尽头');
    document.querySelector('#play-again').focus({ preventScroll: true });
  }
}

function clearTransforms() {
  cells.forEach(cell => cell.classList.remove('moving-cell', 'merge-target'));
  tiles.forEach(tile => { tile.style.transform = ''; });
}

function shiftTiles(projection, fraction) {
  clearTransforms();
  const first = cells[0].getBoundingClientRect();
  const pitchX = cells[1].getBoundingClientRect().left - first.left;
  const pitchY = cells[4].getBoundingClientRect().top - first.top;
  projection.moves.forEach(({ from, to }) => {
    const dx = (to % 4 - from % 4) * pitchX * fraction;
    const dy = (Math.floor(to / 4) - Math.floor(from / 4)) * pitchY * fraction;
    tiles[from].style.transform = `translate(${dx}px, ${dy}px)`;
    cells[from].classList.add('moving-cell');
  });
  projection.merged.forEach(index => cells[index].classList.add('merge-target'));
}

function releasePointer() {
  const id = pointer?.id;
  pointer = null;
  if (id !== undefined && boardEl.hasPointerCapture?.(id)) boardEl.releasePointerCapture(id);
}

function cancelPreview() {
  releasePointer();
  // A late capture-loss event must not undo an already committed animation.
  if (busy) return;
  boardEl.classList.remove('previewing');
  clearTransforms();
}

function move(direction) {
  if (aiRunning) stopAI(t('已切回手动操作。'));
  if (busy || game.over) return;
  cancelPreview();
  commitMove(direction);
}

function commitMove(direction) {
  const projection = projectMove(game.board, direction);
  if (!projection.changed) {
    boardEl.classList.remove('previewing');
    clearTransforms();
    liveEl.textContent = t('这个方向无法移动。');
    return;
  }
  busy = true;
  boardEl.classList.remove('previewing');
  boardEl.classList.add('animating');
  void boardEl.offsetWidth;
  shiftTiles(projection, 1);
  animationTimer = setTimeout(() => {
    animationTimer = null;
    const result = game.move(direction);
    publicCounts = observePreview(publicCounts, game.next);
    boardEl.classList.remove('animating');
    busy = false;
    render({ ...result, direction });
    liveEl.textContent = game.over ? t('{cleared}本局结束，得分 {score}。',{cleared:game.cleared?t('通关，'):'',score:format(score(game.board))}) : t('第 {turns} 步，补入 {value}。',{turns:game.turns,value:result.value});
    scheduleAI();
  }, reducedMotion.matches ? 0 : 145);
}

function startGame() {
  stopAI(t('新的一局，可以点击 AI 自动玩。'));
  clearTimeout(animationTimer);
  animationTimer = null;
  releasePointer();
  busy = false;
  boardEl.classList.remove('animating', 'previewing');
  game.reset();
  publicCounts = initialCounts(game.board, game.next);
  render();
  liveEl.textContent = t('新的一局。开局九张牌已发好。');
  boardEl.focus({ preventScroll: true });
}

const keyDirections = { ArrowLeft:'left', ArrowRight:'right', ArrowUp:'up', ArrowDown:'down' };
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') { if (aiRunning) stopAI(); cancelPreview(); return; }
  const direction = keyDirections[event.key];
  if (!direction || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.target.closest?.('input,textarea,select,[contenteditable="true"],summary,a')) return;
  event.preventDefault();
  if (!event.repeat) move(direction);
}, { passive: false });
document.querySelectorAll('[data-dir]').forEach(button => {
  button.addEventListener('click', () => move(button.dataset.dir));
});

function pointerDirection(event) {
  const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
  const distance = Math.max(Math.abs(dx), Math.abs(dy));
  const direction = Math.abs(dx) > Math.abs(dy)
    ? (dx > 0 ? 'right' : 'left') : (dy > 0 ? 'down' : 'up');
  return { direction, distance };
}

boardEl.addEventListener('pointerdown', event => {
  if (aiRunning && event.isPrimary !== false && event.button === 0) stopAI(t('已切回手动操作。'));
  if (busy || game.over || pointer || event.isPrimary === false || event.button !== 0) return;
  pointer = { id:event.pointerId, x:event.clientX, y:event.clientY };
  boardEl.focus({ preventScroll: true });
  boardEl.setPointerCapture?.(event.pointerId);
  boardEl.classList.add('previewing');
  clearTransforms();
});
boardEl.addEventListener('pointermove', event => {
  if (!pointer || pointer.id !== event.pointerId) return;
  const { direction, distance } = pointerDirection(event);
  const cellSize = cells[0].getBoundingClientRect().width;
  shiftTiles(projectMove(game.board, direction), Math.min(.78, distance / cellSize));
});
// Listen at the window so releasing outside the board still ends the gesture,
// including hosts which lose pointer capture during an interrupted touch.
window.addEventListener('pointerup', event => {
  if (!pointer || pointer.id !== event.pointerId) return;
  const { direction, distance } = pointerDirection(event);
  const threshold = Math.max(24, Math.min(40, cells[0].getBoundingClientRect().width * .35));
  if (distance < threshold) { cancelPreview(); return; }
  releasePointer();
  commitMove(direction);
}, { capture: true });
window.addEventListener('pointercancel', event => {
  if (pointer?.id === event.pointerId) cancelPreview();
}, { capture: true });
boardEl.addEventListener('lostpointercapture', event => {
  if (pointer?.id === event.pointerId) cancelPreview();
});
window.addEventListener('blur', cancelPreview);
window.addEventListener('pagehide', cancelPreview);
window.addEventListener('pagehide', () => { if (aiRunning) stopAI(t('已离开页面，AI 已停止。')); });
window.addEventListener('resize', cancelPreview);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { cancelPreview(); if (aiRunning) stopAI(t('页面已切到后台，AI 已停止。')); }
});
document.querySelector('#new-game').addEventListener('click', startGame);
document.querySelector('#play-again').addEventListener('click', startGame);
applyLanguage();
document.querySelector('#language')?.addEventListener('change', event => {
  setLanguage(event.target.value);
  cancelPreview();
  applyLanguage();
  stopAI(t('语言已切换，可以继续游戏。'));
  liveEl.textContent = '';
  if (!busy) render();
});
render();
