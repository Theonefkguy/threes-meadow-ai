import { bonusPreviews } from './engine.js?v=21';

const dirs = ['left','right','up','down'];
const lines = dirs.map((d) => Array.from({length:4},(_,i) => Array.from({length:4},(_,j) =>
  d==='left'?i*4+j:d==='right'?i*4+3-j:d==='up'?j*4+i:(3-j)*4+i)));
const toRank = n => n < 3 ? n : Math.log2(n / 3) + 3;
const fromRank = r => r < 3 ? r : 3 * 2 ** (r-3);
const height = r => r < 3 ? (r ? 1 : 0) : r-1;
const mergeable = (a,b) => a && b && (a+b===3 || (a>=3 && a===b));
const rowCache = new Map(), previewCache = new Map();
function move(board, dir) {
  const result=board.slice(), entries=[];
  for (const line of lines[dir]) {
    const row=line.map(i=>board[i]);
    const key=row.join(',');
    let moved=rowCache.get(key);
    if (moved===undefined) {
      let changed=false;
      for(let j=1;j<4;j++) if(row[j] && (!row[j-1] || mergeable(row[j],row[j-1]))) {
        row[j-1]=!row[j-1]?row[j]:row[j]+row[j-1]===3?3:row[j]+1;
        row[j]=0;changed=true;
      }
      moved=changed?row:null;
      if(rowCache.size>100000) rowCache.clear();
      rowCache.set(key,moved);
    }
    if(moved) {line.forEach((idx,j)=>result[idx]=moved[j]);entries.push(line[3]);}
  }
  return entries.length?{board:result,entries}:null;
}

export { initialCounts, observePreview } from './card-memory.js?v=21';

// Reuse the verified, pure rank-board movement in the learned search.
export const projectRanks = move;

function evaluation(b, mode) {
  let empty=0, highest=0, rough=0, pairs=0, disorder=0, traps=0, sum=0;
  for(let i=0;i<16;i++) {
    const a=b[i];highest=Math.max(highest,a);empty+=a===0;sum+=height(a);
    for(const j of [i%4<3?i+1:-1,i<12?i+4:-1]) if(j>=0) {
      const c=b[j];
      if(mergeable(a,c)) pairs+=height(a);
      if(a&&c) rough+=Math.abs(height(a)-height(c));
    }
    if(mode!=='classic' && a) {
      // Penalize local valleys: a small card trapped between larger cards.
      if(i%4>0 && i%4<3 && b[i-1]>a && b[i+1]>a) traps+=Math.min(height(b[i-1]),height(b[i+1]))-height(a);
      if(i>=4 && i<12 && b[i-4]>a && b[i+4]>a) traps+=Math.min(height(b[i-4]),height(b[i+4]))-height(a);
    }
  }
  for(let axis of [0,2]) for(const line of lines[axis]) {
    let up=0,down=0;
    for(let j=1;j<4;j++) {const diff=height(b[line[j]])-height(b[line[j-1]]);up+=Math.max(0,diff);down+=Math.max(0,-diff);}
    disorder+=Math.min(up,down);
  }
  const corner=[b[0],b[3],b[12],b[15]].includes(highest)?height(highest):0;
  if(!empty && !dirs.some((_,d)=>move(b,d))) return -100000;
  if(mode==='space') return empty*340 + pairs*40 + sum*3 + corner*25 - rough*5 - disorder*16 - traps*45;
  if(mode==='chain') return empty*260 + pairs*32 + sum*3 + corner*45 - rough*7 - disorder*32 - traps*30;
  return empty*260 + pairs*28 + sum*3 + corner*65 - rough*7 - disorder*22;
}

export function chooseMove(board,next,{maxDepth=3,budgetMs=160,maxNodes=24000,mode='classic',remaining=null,remember=false}={}) {
  if(!next?.candidates?.length) return {direction:null,depth:0,nodes:0};
  const b=board.flat().map(toRank), legal=dirs.map((direction,d)=>({direction,d,m:move(b,d)})).filter(x=>x.m);
  if(!legal.length) return {direction:null,depth:0,nodes:0};
  const convert=p=>({cards:p.candidates.map(toRank),weights:p.probabilities||p.candidates.map(()=>1/p.candidates.length)});
  const root=convert(next), cache=new Map(), deadline=performance.now()+budgetMs, timeout=Symbol();
  let nodes=0,completed=0,best=legal[0].direction;
  function tick() {nodes++;if(completed && (nodes>maxNodes || (nodes%64===0 && performance.now()>=deadline))) throw timeout;}
  function future(b,counts,depth) {
    let high=0;for(const r of b)high=Math.max(high,r);
    let bonuses=previewCache.get(high);
    if(!bonuses) {bonuses=bonusPreviews(fromRank(high)).map(p=>({...convert(p),probability:p.probability}));previewCache.set(high,bonuses);}
    const chance=bonuses.length?1/21:0;
    let total=0, deck=counts;
    if(deck && deck.every(n=>n===0)) deck=[4,4,4];
    const size=deck?deck.reduce((a,b)=>a+b):3;
    for(let n=1;n<=3;n++) {
      const p=deck?deck[n-1]/size:1/3;if(!p)continue;
      const remaining=deck?deck.map((v,i)=>v-(i===n-1)):null;
      total+=(1-chance)*p*decision(b,{cards:[n],weights:[1]},remaining,depth);
    }
    for(const preview of bonuses) total+=chance*preview.probability*decision(b,preview,counts,depth);
    return total;
  }
  function decision(b,preview,counts,depth) {
    tick();const key=depth+'|'+b.join(',')+'|'+preview.cards.join(',')+'|'+preview.weights.join(',')+'|'+counts;
    if(cache.has(key))return cache.get(key);
    let value=-100000;
    for(let d=0;d<4;d++){const projected=move(b,d);if(projected)value=Math.max(value,after(projected,preview,counts,depth));}
    cache.set(key,value);return value;
  }
  function after(projected,preview,counts,depth) {
    let total=0;
    for(const pos of projected.entries) for(let i=0;i<preview.cards.length;i++) {
      tick();const state=projected.board.slice();state[pos]=preview.cards[i];
      total+=preview.weights[i]*(depth===1?evaluation(state,mode):future(state,counts,depth-1));
    }
    return total/projected.entries.length;
  }
  for(let depth=1;depth<=maxDepth;depth++) {
    try {
      let value=-Infinity,choice=best;
      for(const item of legal){const v=after(item.m,root,remember?remaining:null,depth);if(v>value){value=v;choice=item.direction;}}
      best=choice;completed=depth;
    }catch(e){if(e!==timeout)throw e;break;}
  }
  return {direction:best,depth:completed,nodes};
}

// Exposed only for equivalence tests; gameplay still uses the canonical engine.
export function projectFast(board,direction) {
  const result=move(board.flat().map(toRank),dirs.indexOf(direction));
  return result?{board:Array.from({length:4},(_,i)=>result.board.slice(i*4,i*4+4).map(fromRank)),entries:result.entries}:null;
}
