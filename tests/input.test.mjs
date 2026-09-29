// DOM test double: exercises production event handlers and timer races, not visual rendering.
import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import fs from 'node:fs';
import { createGame, projectMove, score } from '../dist/engine.js';
import { t, locale, setLanguage } from '../dist/i18n.js';
import { initialCounts, observePreview } from '../dist/card-memory.js';

function setup() {
  setLanguage("zh-CN");
  let board, engine;
  class Element {
    constructor() {
      this.children = []; this.events = {}; this.attrs = {}; this.hidden = true;
      this.dataset = {}; this.captured = null; this.textContent = ''; this.className = '';
      const names = new Set();
      this.classList = {
        add: (...n) => n.forEach(s => names.add(s)),
        remove: (...n) => n.forEach(s => names.delete(s)),
        toggle: (n, on) => on ? names.add(n) : names.delete(n),
        contains: n => names.has(n)
      };
      this.style = { transform:'', setProperty: (k,v) => { this.style[k] = v; } };
    }
    setAttribute(k,v) { this.attrs[k] = v; }
    append(node) { this.children.push(node); }
    replaceChildren() { this.children = []; }
    focus() {}
    closest() { return null; }
    addEventListener(n, fn) { (this.events[n] ||= []).push(fn); }
    emit(n, e = {}) { for (const fn of this.events[n] || []) fn({ target:this, ...e }); }
    setPointerCapture(id) { this.captured = id; }
    hasPointerCapture(id) { return id === this.captured; }
    releasePointerCapture() { const pointerId = this.captured; this.captured = null; this.emit('lostpointercapture', { pointerId }); }
    getBoundingClientRect() {
      const i = board.children.flatMap(row => row.children).indexOf(this);
      return { left:(i % 4) * 100, top:Math.floor(i / 4) * 100, width:90, height:90 };
    }
  }
  const ids = Object.fromEntries(['board','next-tile','next-hint','score','best','game-over','move-status','final-score','play-again','new-game','ai-toggle','ai-status','ai-policy','over-kicker','over-title'].map(id => ['#'+id,new Element()]));
  ids['#ai-policy'].value='classic';
  board = ids['#board'];
  const document = new Element();
  document.hidden = false;
  document.querySelector = s => ids[s];
  document.querySelectorAll = () => [];
  document.createElement = () => new Element();
  const window = new Element();
  window.matchMedia = () => ({ matches:false });
  let serial = 0;
  const timers = new Map();
  const delays = new Map();
  const workers = [];
  class Worker {
    constructor() { this.messages = []; this.terminated = false; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    reply(direction) { this.onmessage({ data:{ id:this.messages.at(-1).id, direction } }); }
  }
  const context = vm.createContext({ document, window, Intl, console, Worker,
    localStorage: { getItem:() => { throw new Error('denied'); }, setItem:() => { throw new Error('denied'); } },
    t, locale, setLanguage, applyLanguage: () => {},
    createGame: () => (engine = createGame(() => .5)), projectMove, score, initialCounts, observePreview,
    setTimeout: (fn, delay) => { timers.set(++serial,fn); delays.set(serial,delay); return serial; },
    clearTimeout: id => timers.delete(id)
  });
  const source = fs.readFileSync(new URL('../dist/game.js', import.meta.url), 'utf8').replace(/^import[^;]+;/gm, '');
  vm.runInContext(source, context);
  const flush = () => { const tasks = [...timers].filter(([id]) => delays.get(id) < 10000); tasks.forEach(([id]) => timers.delete(id)); tasks.forEach(([,fn]) => fn()); };
  const snapshot = () => JSON.stringify(engine);
  const setBoard = row => {
    engine.board = [row, [0,0,0,0], [0,0,0,0], [0,0,0,0]];
    engine.over = false;
    vm.runInContext('render()', context);
  };
  const pointer = (name, x, id = 1) => {
    const target = name === 'pointerup' || name === 'pointercancel' ? window : board;
    target.emit(name, { pointerId:id, clientX:x, clientY:100, button:0, isPrimary:true });
  };
  const cards = () => board.children.flatMap(row => row.children).flatMap(cell => cell.children);
  return { board, engine, ids, document, window, timers, context, flush, snapshot, setBoard, pointer, cards, workers };
}

test('drag cancel, committed swipe, hidden candidates, blocked swipe and restart race', () => {
  const { board, engine, ids, document, timers, context, flush, snapshot, setBoard, pointer } = setup();
  setBoard([0,3,0,0]);
  const before = snapshot();
  pointer('pointerdown',100); pointer('pointermove',50);
  assert.equal(snapshot(), before, 'preview must not mutate game');
  assert(board.children[0].children[1].children[0].style.transform.includes('-'));
  pointer('pointermove',98); pointer('pointerup',98); flush();
  assert.equal(snapshot(), before, 'pulling back cancels');
  pointer('pointerdown',100); pointer('pointermove',50); pointer('pointercancel',50); flush();
  assert.equal(snapshot(), before, 'OS cancellation cancels');
  pointer('pointerdown',100); pointer('pointerup',50); flush();
  assert.equal(engine.turns, 1);
  assert.equal(engine.board[0][0], 3);
  assert(engine.board[0][3] > 0);

  engine.next = { bonus:true, candidates:[12,24,48] };
  vm.runInContext('render()', context);
  assert.deepEqual(ids['#next-tile'].children.map(n => Number(n.textContent)), [12,24,48]);
  assert.equal(ids['#next-tile'].attrs['aria-label'], '下一张可能是 12、24、48');

  setBoard([1,1,1,1]);
  const blocked = snapshot();
  document.emit('keydown', { key:'ArrowLeft', preventDefault(){} }); flush();
  assert.equal(snapshot(), blocked);
  setBoard([0,3,0,0]);
  document.emit('keydown', { key:'ArrowLeft', preventDefault(){} });
  assert.equal(timers.size, 1);
  ids['#new-game'].emit('click');
  const restarted = snapshot(); flush();
  assert.equal(snapshot(), restarted, 'old animation cannot commit into new game');
  assert.equal(engine.turns, 0);
  assert.equal(ids['#game-over'].hidden, true);
});

test('interrupted gestures always clear preview without drawing a card', () => {
  for (const interruption of ['pointercancel', 'lostpointercapture', 'blur', 'pagehide', 'resize', 'visibilitychange']) {
    const { board, document, window, snapshot, setBoard, pointer, cards, flush } = setup();
    setBoard([0,3,0,0]);
    const before = snapshot();
    pointer('pointerdown',100); pointer('pointermove',50);
    assert(board.classList.contains('previewing'));
    assert(cards().some(card => card.style.transform !== ''));
    if (interruption === 'pointercancel') pointer(interruption,50);
    else if (interruption === 'lostpointercapture') board.emit(interruption, { pointerId:1 });
    else if (interruption === 'visibilitychange') { document.hidden = true; document.emit(interruption); }
    else window.emit(interruption);
    flush();
    assert.equal(snapshot(), before, interruption);
    assert(!board.classList.contains('previewing'), interruption);
    assert(cards().every(card => card.style.transform === ''), interruption);
    assert.equal(board.captured, null, interruption);
  }
});

test('release outside the board commits exactly once', () => {
  const { board, engine, window, setBoard, pointer, flush } = setup();
  setBoard([0,3,0,0]);
  pointer('pointerdown',100); pointer('pointermove',50);
  const release = { pointerId:1, clientX:-50, clientY:100 };
  window.emit('pointerup', release);
  window.emit('pointerup', release);
  board.emit('lostpointercapture', { pointerId:1 });
  flush();
  assert.equal(engine.turns, 1);
  assert.equal(engine.board[0][0], 3);
  assert(!board.classList.contains('previewing'));
  assert(!board.classList.contains('animating'));
});

test('late capture loss from an older touch cannot cancel the new touch', () => {
  const { board, engine, setBoard, pointer, flush } = setup();
  setBoard([0,3,0,0]);
  pointer('pointerdown',100,1); pointer('pointercancel',100,1);
  pointer('pointerdown',100,2); pointer('pointermove',50,2);
  board.emit('lostpointercapture', { pointerId:1 });
  assert(board.classList.contains('previewing'));
  assert.equal(board.captured, 2);
  pointer('pointerup',50,2); flush();
  assert.equal(engine.turns, 1);
});

test('focus loss during a committed slide preserves the pending move', () => {
  const { board, engine, window, setBoard, pointer, cards, flush } = setup();
  setBoard([0,3,0,0]);
  pointer('pointerdown',100); pointer('pointerup',50);
  const transforms = cards().map(card => card.style.transform);
  window.emit('blur');
  assert(board.classList.contains('animating'));
  assert.deepEqual(cards().map(card => card.style.transform), transforms);
  flush();
  assert.equal(engine.turns, 1);
  assert(cards().every(card => card.style.transform === ''));
});

test('AI starts from the current game, moves repeatedly, and stops without accepting stale replies', () => {
  const { engine, ids, setBoard, flush, workers, timers } = setup();
  setBoard([0,3,0,0]);
  const original = JSON.stringify(engine.board);
  ids['#ai-toggle'].emit('click'); flush();
  const worker = workers[0];
  assert.equal(JSON.stringify(engine.board), original);
  assert.equal(ids['#ai-toggle'].attrs['aria-pressed'], 'true');
  assert.deepEqual(Object.keys(worker.messages[0]).sort(), ['board','id','next','policy','remaining']);
  worker.reply('left'); flush();
  assert.equal(engine.turns,1);
  flush();
  assert.equal(worker.messages.length,2, 'automatically requests the next move');
  ids['#ai-toggle'].emit('click');
  worker.reply('right'); flush();
  assert.equal(engine.turns,1, 'stopped worker cannot move the board');
  assert(worker.terminated);
  assert.equal(timers.size,0);
  assert.equal(ids['#ai-toggle'].attrs['aria-pressed'], 'false');
});

test('manual input, new game, backgrounding and worker failure stop AI', () => {
  for (const action of ['manual','restart','hidden','failure']) {
    const { engine, ids, document, setBoard, flush, workers } = setup();
    setBoard([0,3,0,0]);
    ids['#ai-toggle'].emit('click'); flush();
    const worker = workers[0];
    if (action === 'manual') document.emit('keydown',{key:'ArrowLeft',preventDefault(){}});
    if (action === 'restart') ids['#new-game'].emit('click');
    if (action === 'hidden') { document.hidden = true; document.emit('visibilitychange'); }
    if (action === 'failure') worker.onerror();
    worker.reply('left'); flush();
    assert(worker.terminated,action);
    assert.equal(ids['#ai-toggle'].attrs['aria-pressed'],'false',action);
    assert.equal(engine.turns, action === 'manual' ? 1 : 0, action);
  }
});

test('AI automatically stops at game over', () => {
  const { engine, ids, context, flush, workers } = setup();
  engine.board = [[3,3,24,96],[12,48,192,6],[24,96,6,12],[48,192,12,24]];
  engine.next = { bonus:false,candidates:[1] };
  engine.over = false;
  vm.runInContext('render()',context);
  ids['#ai-toggle'].emit('click'); flush();
  workers[0].reply('left'); flush();
  assert(engine.over);
  assert(workers[0].terminated);
  assert(ids['#ai-toggle'].disabled);
  assert.equal(ids['#ai-toggle'].attrs['aria-pressed'],'false');
});

test('manual play is counted before AI takeover and restart resets the visible memory', () => {
 const {engine,ids,document,flush,workers}=setup();
 for(let i=0;i<20&&!engine.over;i++){
  const dir=['left','right','up','down'].find(d=>projectMove(engine.board,d).changed);
  document.emit('keydown',{key:{left:'ArrowLeft',right:'ArrowRight',up:'ArrowUp',down:'ArrowDown'}[dir],preventDefault(){}});flush();
 }
 if(!engine.over){
  ids['#ai-toggle'].emit('click');flush();
  assert.deepEqual(Array.from(workers.at(-1).messages[0].remaining),[1,2,3].map(n=>engine.deck.filter(v=>v===n).length));
 }
 ids['#new-game'].emit('click');ids['#ai-toggle'].emit('click');flush();
 assert.deepEqual(Array.from(workers.at(-1).messages[0].remaining),[1,2,3].map(n=>engine.deck.filter(v=>v===n).length));
});

test('switching learned policy stops old work, and failed model load visibly falls back',()=>{
 const {ids,engine,flush,workers,setBoard}=setup();setBoard([0,3,0,0]);
 ids['#ai-policy'].value='rl-v4';ids['#ai-policy'].emit('change');ids['#ai-toggle'].emit('click');flush();
 const worker=workers[0];assert.equal(worker.messages[0].policy,'rl-v4');
 ids['#ai-policy'].value='classic';ids['#ai-policy'].emit('change');worker.reply('left');flush();
 assert(worker.terminated);assert.equal(engine.turns,0);
 ids['#ai-policy'].value='rl-v4';ids['#ai-toggle'].emit('click');flush();const nextWorker=workers[1];
 nextWorker.onmessage({data:{id:nextWorker.messages[0].id,direction:'left',policy:'classic',fallback:'load'}});
 assert.equal(ids['#ai-policy'].value,'classic');assert.match(ids['#ai-status'].textContent,/切换经典/);flush();assert.equal(engine.turns,1);
});

test('default takeover sends only visible information and load fallback keeps subsequent turns on V4',()=>{
 const {ids,workers,engine,flush,setBoard}=setup();setBoard([0,3,0,0]);
 ids['#ai-policy'].value='rl-fast';ids['#ai-toggle'].emit('click');flush();
 const worker=workers[0],message=worker.messages[0];
 assert.equal(message.policy,'rl-fast');assert(!('deck' in message));
 worker.onmessage({data:{id:message.id,direction:'left',policy:'rl-v4',fallback:'fast-load'}});
 assert.equal(ids['#ai-policy'].value,'rl-v4');assert.match(ids['#ai-status'].textContent,/切换第四版/);
 flush();assert.equal(engine.turns,1);
 flush();assert.equal(worker.messages.length,2);assert.equal(worker.messages.at(-1).policy,'rl-v4');
});
