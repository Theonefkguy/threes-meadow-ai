// Analysis mode: shows the public hidden state (the 12-card bag, the bonus cycle) and the AI's view of
// the four directions. Uses only information a player could infer; never reads the shuffled deck.
import { t } from './i18n.js?v=33';

const DIRS = ['left', 'right', 'up', 'down'];
const NAMES = { left:'左', right:'右', up:'上', down:'下' };
const ARROWS = { left:'←', right:'→', up:'↑', down:'↓' };
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const pct = (x, digits = 0) => `${(100 * x).toFixed(digits)}%`;

export function createAnalysis({ getState, send }) {
  const toggle = document.querySelector('#analysis-toggle'), panel = document.querySelector('#analysis-panel');
  if (!toggle || !panel) return null;
  const bagEl = document.querySelector('#an-bag'), bagNote = document.querySelector('#an-bag-note');
  const cycleEl = document.querySelector('#an-cycle'), cycleNote = document.querySelector('#an-cycle-note');
  const recEl = document.querySelector('#an-rec'), recNote = document.querySelector('#an-rec-note');
  const arrows = document.querySelector('#rec-arrows'), app = document.querySelector('.app');
  const bagLeft = document.querySelector('#an-bag-left'), cyclePos = document.querySelector('#an-cycle-pos');
  let on = false, request = 0, timer = null, lastKey = '', sentKey = '', lastRec = null;
  const keyOf = s => JSON.stringify([s.board, s.next, s.remaining, s.cycle]);

  function renderBag(s) {
    bagEl.replaceChildren();
    const next = s.next, rem = s.remaining;
    if (next?.bonus) {
      const row = el('div', 'an-row');
      for (const v of next.candidates) row.append(el('span', 'mc mc-bonus', String(v)));
      row.append(el('span', 'an-pct', t('奖励牌')));
      bagEl.append(row);
    }
    const left = rem[0] + rem[1] + rem[2];
    [1, 2, 3].forEach((n, i) => {
      const row = el('div', 'an-row'), isNext = next && !next.bonus && next.candidates[0] === n;
      const solid = rem[i] + (isNext ? 1 : 0);
      for (let k = 0; k < 4; k++) {
        const c = el('span', `mc mc-${n}${k < solid ? '' : ' mc-used'}${isNext && k === 0 ? ' mc-next' : ''}`, String(n));
        row.append(c);
      }
      row.append(el('span', 'an-pct', pct(left ? rem[i] / left : 1 / 3)));
      bagEl.append(row);
    });
    bagLeft.textContent = t('袋里还剩 {n} 张', { n: left });
    bagNote.textContent = left ? t('百分比：预告之后那张普通牌是 1、2、3 的概率。') : t('本袋已发完，预告之后那张普通牌来自新的一袋。');
  }

  function renderCycle(s) {
    cycleEl.replaceChildren();
    const c = s.cycle, possible = Math.max(...s.board.flat()) >= 48;
    for (let i = 0; i < 21; i++) cycleEl.append(el('span', `cy${i < c.pos ? ' cy-past' : ''}${i === c.pos ? ' cy-now' : ''}`));
    const chance = !possible ? 0 : c.used ? 0 : 1 / (21 - c.excl);
    cyclePos.textContent = t('第 {pos} / 21 张', { pos: c.pos + 1 });
    cycleNote.textContent = !possible ? t('最大数字到 48 后才会出现奖励牌。')
      : c.used ? t('本轮的奖励牌已经出现，本轮不会再有。')
      : t('本轮奖励牌还没出现。预告之后那张是奖励牌的概率：{p}', { p: pct(chance, 1) });
  }

  function showRec(values, direction, fromAI = false, key = '') {
    lastRec = { values, direction, fromAI, key };
    recEl.replaceChildren();
    if (!values || !direction) { recNote.textContent = t('当前策略不提供各方向估值。'); setArrows(null); return; }
    const rows = DIRS.map((d, i) => ({ d, v: values[i] })).filter(r => r.v !== null && r.v !== undefined).sort((a, b) => b.v - a.v);
    const best = rows[0].v, scale = Math.max(1, Math.abs(best));
    // Bars: the best move is full; the others shrink with their gap, on a scale of the largest gap
    // (at least 0.5%), so small but real differences stay visible without being exaggerated.
    const span = Math.max(0.005, ...rows.map(r => (best - r.v) / scale));
    for (const r of rows) {
      const diff = (best - r.v) / scale, row = el('div', `an-bar${r.d === direction ? ' an-best' : ''}`);
      const name = t(NAMES[r.d]), dir = el('span', 'an-dir', ARROWS[r.d]);
      dir.setAttribute('aria-label', name); dir.title = name; row.append(dir);
      const track = el('span', 'an-track'), fill = el('span', 'an-fill');
      fill.style.width = `${Math.max(8, 100 - 85 * diff / span)}%`;
      track.append(fill); row.append(track);
      row.append(el('span', 'an-diff', r.d === direction ? t('推荐') : `−${pct(diff, 2)}`));
      recEl.append(row);
    }
    const margin = rows.length > 1 ? (best - rows[1].v) / scale : 1;
    const verdict = rows.length < 2 ? t('只有一个方向可以走。')
      : margin < 0.0025 ? t('险胜：前几个方向几乎一样好，选哪个差别很小。')
      : margin < 0.01 ? t('推荐方向略好一些。') : t('推荐方向明显更好。');
    recNote.textContent = fromAI ? t('AI 刚才这一步：{verdict}', { verdict }) : verdict;
    setArrows(fromAI ? null : rows.map(r => ({ d: r.d, shade: 1 - (best - r.v) / scale / span })), direction);
  }

  // One arrow per legal direction on the board edge: the best is solid and ringed, the others fade
  // and shrink with their gap to the best (same scale as the bars); illegal directions get none.
  function setArrows(list, direction) {
    if (!arrows) return;
    arrows.hidden = !on || !list;
    if (!list) return;
    for (const a of arrows.children) {
      const r = list.find(x => x.d === a.dataset.dir);
      a.hidden = !r;
      if (!r) continue;
      const s = Math.max(0, Math.min(1, r.shade));
      a.style.setProperty('--shade', String(0.18 + 0.82 * s));
      a.style.setProperty('--size', String(0.7 + 0.3 * s));
      a.classList.toggle('rec-best', a.dataset.dir === direction);
    }
  }

  function requestRec() {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const s = getState();
      if (!on || s.over || s.aiRunning) return;
      const key = keyOf(s);
      if (key === lastKey) return;
      lastKey = sentKey = key;
      recNote.textContent = t('AI 正在计算…');
      send({ id: ++request, analysis: true, board: s.board, next: s.next, remaining: s.remaining, cycle: s.cycle, policy: s.policy === 'rl-full' ? 'rl-full' : 'rl-fast' });
    }, 120);
  }

  function update() {
    if (!on) return;
    const s = getState();
    renderBag(s); renderCycle(s);
    // Re-draw the last suggestion (for example after a language change) if it still belongs here.
    if (lastRec && (lastRec.fromAI || lastRec.key === keyOf(s))) showRec(lastRec.values, lastRec.direction, lastRec.fromAI, lastRec.key);
    if (s.over) { recEl.replaceChildren(); recNote.textContent = t('本局已结束。'); setArrows(null); return; }
    if (!s.aiRunning) requestRec();
  }

  toggle.addEventListener('click', () => {
    on = !on; toggle.setAttribute('aria-pressed', String(on)); panel.hidden = !on; app?.classList.toggle('analysis-on', on);
    if (!on) { setArrows(null); return; }
    lastKey = ''; update();
  });

  return {
    update,
    get on() { return on; },
    // Replies to analysis requests (data.analysis); stale ones are ignored.
    onReply(data) {
      if (!on || data.id !== request) return;
      if (data.type === 'progress') { if (data.total) recNote.textContent = t('正在下载模型 {p}', { p: pct(data.loaded / data.total) }); return; }
      if (data.error) { recNote.textContent = t('AI 暂时无法计算。'); return; }
      showRec(data.values, data.direction, false, sentKey);
    },
    // The AI's own replies while it plays carry the same values.
    onAIReply(data) { if (on) showRec(data.values, data.direction, true); },
    reset() { lastKey = ''; }
  };
}
