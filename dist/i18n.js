// Text-only localization; gameplay and model inputs are language-independent.
export const english = {
  "合三": "Threes Meadow",
  "让数字找到彼此": "Make room for the next match",
  "新一局": "New game",
  "得分": "Score",
  "最佳": "Best",
  "再来一局": "Play again",
  "无路可走": "No moves left",
  "这一局走到了尽头": "Game over",
  "AI 策略": "AI strategy",
  "AI 自动玩": "Play with AI",
  "停止 AI": "Stop AI",
  "五层剪枝搜索（默认）": "Five-ply search (default)",
  "强化学习 · 第四版": "RL · V4",
  "经典搜索": "Classic search",
  "从当前棋盘接手，点击停止或手动操作即可接管。": "Starts from your board. Stop the AI or make a move to take over.",
  "默认使用 V23 模型与五层剪枝搜索。模型约 8 MB，下载后在你的设备上运行，无需账号。": "The default uses the V23 model and five-ply search. The 8 MB model runs on your device after download. No account needed.",
  "每次移动一格。1 与 2 相合，3 以上相同数字相合。拖动可预览，拉回可取消。": "Slide one cell at a time. Merge 1 with 2, or equal cards of 3 and above. Drag to preview; pull back to cancel.",
  "规则与来牌": "Rules and incoming cards",
  "使用滑动、方向键或屏幕按钮操作。一次有效移动后，一张新牌从反方向的边缘进入，只会进入刚刚移动过的行或列。无法移动时不会补牌。": "Use swipes, arrow keys or the on-screen buttons. After a valid move, a card enters from the opposite edge, uniformly among the rows or columns that moved. Blocked moves do not draw cards.",
  "普通牌是一副 12 张的牌组：1、2、3 各四张。开局从中抽九张，之后继续抽剩余牌；抽完再洗牌。因此下一张普通牌的概率取决于牌组中还剩什么。": "The basic deck contains twelve cards: four each of 1, 2 and 3. Nine cards are dealt at the start. Play continues through the remaining deck before reshuffling, so draw probabilities depend on what remains.",
  "最大数字达到 48 后，每次生成来牌提示有 1/21（约 4.76%）的机会出现奖励牌。奖励牌不消耗普通牌组，数值从 6 开始逐级翻倍，最高为棋盘最大数字的八分之一。": "Once the highest card reaches 48, each new preview has a 1/21 chance of being a bonus. Bonuses do not consume the basic deck. Values double from 6 up to one eighth of the highest card.",
  "奖励牌先等概率选择一个合法候选窗口，再在窗口内等概率抽牌。预告最多显示三个连续候选值；中间数值因此可能比两端更常见。": "A bonus window is chosen uniformly, then a card is drawn uniformly within it. The preview shows up to three consecutive values, making middle values more likely than the extremes when there are multiple windows.",
  "1、2 不计分；3、6、12、24… 分别计 3、9、27、81… 分。无路可走时结束；两张 6144 合出 12288 时立即通关。": "Cards 1 and 2 score zero; 3, 6, 12, 24… score 3, 9, 27, 81… points. The game ends when no move remains, or immediately when two 6144 cards merge into 12288.",
  "规则参考：": "Rules reference:",
  "社区逆向指南": "Community reverse-engineering guide",
  "独立研究项目，与 Threes! / Sirvo 无隶属关系。": "Independent research project, not affiliated with Threes! or Sirvo.",
  "源码与评测": "Source & benchmarks",
  "合三游戏": "Threes Meadow game",
  "游戏状态": "Game status",
  "4乘4游戏棋盘": "4 by 4 game board",
  "移动方向": "Move direction",
  "向上移动": "Move up",
  "向左移动": "Move left",
  "向下移动": "Move down",
  "向右移动": "Move right",
  "已停止，可以手动继续。": "Stopped. You can continue manually.",
  "本局结束。点击新一局后，可以再次开启 AI。": "Game over. Start a new game to play with AI again.",
  "强化学习 AI 正在思考…": "The learned AI is thinking…",
  "AI 正在思考…": "AI is thinking…",
  "加载或思考超时，已停止。可以重试或手动继续。": "Loading or thinking timed out. Retry or continue manually.",
  "本局已结束，请先开始新一局。": "This game has ended. Start a new game first.",
  "AI 暂时无法继续，可以手动操作或重新开启。": "AI cannot continue. Play manually or restart the AI.",
  "五层剪枝搜索暂时无法加载，已切换第四版。": "Five-ply search could not load. Switched to V4.",
  "模型暂时无法加载，已切换经典搜索。": "Model could not load. Switched to classic search.",
  "已超过模型训练范围，切换经典搜索。": "Board exceeds the trained range. Switched to classic search.",
  "前期": "early game",
  "中期": "midgame",
  "后期": "late game",
  "冲刺 12288": "12288 endgame",
  "左": "left",
  "右": "right",
  "上": "up",
  "下": "down",
  "AI 加载失败，可以重试或手动继续。": "AI failed to load. Retry or continue manually.",
  "正在加载强化学习模型…": "Loading the learned model…",
  "AI 已接手，正在观察棋盘…": "AI has taken over and is reading the board…",
  "当前浏览器无法启动 AI，可以继续手动玩。": "AI could not start in this browser. Manual play is still available.",
  "策略已切换，点击 AI 自动玩继续。": "Strategy changed. Press Play with AI to continue.",
  "策略已切换，点击 AI 自动玩从当前棋盘开始。": "Strategy changed. Press Play with AI to start from this board.",
  "本局结束": "Game over",
  "其中一张": "One of these",
  "奖励牌": "Bonus card",
  "下一张": "Next card",
  "空": "empty",
  "合出 12288": "12288 reached",
  "通关！两张 6144 合成了最终的 12288": "You won! Two 6144 cards made the final 12288.",
  "已切回手动操作。": "Manual control restored.",
  "这个方向无法移动。": "You cannot move in that direction.",
  "新的一局，可以点击 AI 自动玩。": "New game. Press Play with AI to start the AI.",
  "新的一局。开局九张牌已发好。": "New game. Nine opening cards have been dealt.",
  "已离开页面，AI 已停止。": "AI stopped because you left the page.",
  "页面已切到后台，AI 已停止。": "AI stopped because the page is in the background.",
  "语言已切换，可以继续游戏。": "Language changed. Continue playing.",
  "下一张{kind} {cards}": "Next card {kind} {cards}",
  "可能是": "may be",
  "是": "is",
  "第{row}行，第{col}列：{value}": "Row {row}, column {col}: {value}",
  " · {stage}策略": " · {stage}",
  "AI 向{direction}移动{phase} · 已完成 {turns} 步": "AI moved {direction}{phase} · {turns} moves",
  "{cleared}本局结束，得分 {score}。": "{cleared}Game over. Score: {score}.",
  "通关，": "You won! ",
  "第 {turns} 步，补入 {value}。": "Move {turns}: added {value}."
};
let language = 'en';
try {
  const saved = localStorage.getItem('threesLanguage');
  language = saved === 'en' || saved === 'zh-CN' ? saved : 'en';
} catch { /* Browser storage can be unavailable. */ }
export const locale = () => language;
export function setLanguage(value) {
  language = value === 'en' ? 'en' : 'zh-CN';
  try { localStorage.setItem('threesLanguage', language); } catch {}
}
export function t(key, values = {}) {
  const text = language === 'en' ? (english[key] ?? key) : key;
  return text.replace(/\{(\w+)\}/g, (_, name) => String(values[name] ?? ''));
}
export function applyLanguage() {
  document.documentElement.lang = language;
  document.title = t('合三');
  document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll('[data-i18n-aria]').forEach(el => { el.setAttribute('aria-label', t(el.dataset.i18nAria)); });
  document.querySelector('#language').value = language;
  document.querySelector('meta[name="description"]').content = language === 'en'
    ? 'Play Threes Meadow with a five-ply AI. Free, on-device, no account required.'
    : '合三：体验五层搜索 AI，免费试玩，本机运行，无需账号。';
}
