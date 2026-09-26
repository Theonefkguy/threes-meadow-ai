import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { chooseFastMove } from '../dist/fast-search.js';
import { chooseLearnedMove } from '../dist/rl-search.js';
import { decodeModel } from '../dist/rl-model.js';
import { projectMove, createGame } from '../dist/engine.js';
import { initialCounts, observePreview } from '../dist/card-memory.js';

const buf=readFileSync(new URL('../dist/models/ntuple-v4.bin',import.meta.url));
const model=decodeModel(buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength));
const slow={value:(b,s)=>model.value(b,s)}; // same model without the fast weights path
const unlimited={budgetMs:1e9,maxNodes:1e12};
const rngFor=seed=>{let s=seed;return ()=>((s=(Math.imul(s,1664525)+1013904223)>>>0)/4294967296);};

// Play a few moves with the existing search to reach varied mid-game states.
function states(count){
  const out=[];
  for(let seed=1;out.length<count;seed++){
    const g=createGame(rngFor(seed));g.reset();let counts=initialCounts(g.board,g.next);
    for(let t=0;t<60&&!g.over;t++){
      if(t%15===7)out.push({board:g.board.map(r=>r.slice()),next:structuredClone(g.next),remaining:counts.slice()});
      const {direction}=chooseLearnedMove(g.board,g.next,model,{remaining:counts,maxDepth:2,...unlimited});
      g.move(direction);if(!g.over)counts=observePreview(counts,g.next);
    }
  }
  return out.slice(0,count);
}

test('threshold 0 reproduces the existing complete search direction',()=>{
  for(const s of states(12)){
    const a=chooseLearnedMove(s.board,s.next,model,{remaining:s.remaining,maxDepth:3,...unlimited});
    const b=chooseFastMove(s.board,s.next,model,{remaining:s.remaining,maxDepth:3,threshold:0,...unlimited});
    assert.equal(a.depth,3);assert.equal(b.depth,3);assert.equal(b.direction,a.direction);
  }
});

test('fast leaf path equals the generic model.value path, with and without cutoff',()=>{
  for(const s of states(6))for(const threshold of [0,0.003]){
    const x={},y={};
    const a=chooseFastMove(s.board,s.next,model,{remaining:s.remaining,maxDepth:3,threshold,...unlimited,stats:x});
    const b=chooseFastMove(s.board,s.next,slow,{remaining:s.remaining,maxDepth:3,threshold,...unlimited,stats:y});
    assert.equal(a.direction,b.direction);assert.deepEqual(x.values,y.values);assert.equal(x.cutoffs,y.cutoffs);
  }
});

test('cutoff reduces work and still returns a legal move at depth 4',()=>{
  let fewer=0;
  for(const s of states(4)){
    const exact=chooseFastMove(s.board,s.next,model,{remaining:s.remaining,maxDepth:4,threshold:0,...unlimited});
    const cut=chooseFastMove(s.board,s.next,model,{remaining:s.remaining,maxDepth:4,threshold:0.02,...unlimited});
    assert.equal(cut.depth,4);assert(cut.nodes<=exact.nodes);fewer+=cut.nodes<exact.nodes;assert(projectMove(s.board,cut.direction).changed);
  }
  assert(fewer>=2);
});

test('budget fallback, immutability, terminal and range handling',()=>{
  const g=createGame(()=>.5);g.reset();const next={candidates:[6,12,24],probabilities:[.5,.25,.25]},before=JSON.stringify([g.board,next]);
  const r=chooseFastMove(g.board,next,model,{maxNodes:1,budgetMs:0});
  assert.equal(r.depth,1);assert(projectMove(g.board,r.direction).changed);assert.equal(JSON.stringify([g.board,next]),before);
  const terminal=[[3,6,12,24],[6,12,24,48],[12,24,48,96],[24,48,96,192]];
  assert.equal(chooseFastMove(terminal,{candidates:[1]},model).direction,null);
  const huge=[[12288,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]];
  assert.equal(chooseFastMove(huge,{candidates:[1]},model).unsupported,true);
  // Unknown counts fall back to equal thirds rather than failing.
  assert(projectMove(g.board,chooseFastMove(g.board,next,model,{remaining:null,maxDepth:2}).direction).changed);
});

test('a move that makes 12288 is terminal: merge points only, no spawn or future value', () => {
  const board=[[6144,6144,3,1],[2,6,12,24],[48,96,192,384],[768,1536,3072,1]];
  for (const maxDepth of [1,3,5]) {
    const stats={};
    chooseFastMove(board,{candidates:[2]},model,{remaining:[3,3,3],maxDepth,threshold:0.01,...unlimited,stats});
    assert.equal(stats.values[0], 3**13 - 2*3**12);
  }
});

test('four-stage V21 model: stages 0-2 equal V4, stage 3 is used only from 6144 on', async () => {
  const { stageForRanks } = await import('../dist/rl-model.js');
  const b21=readFileSync(new URL('../dist/models/ntuple-v21.bin',import.meta.url));
  const v21=decodeModel(b21.buffer.slice(b21.byteOffset,b21.byteOffset+b21.byteLength));
  assert.equal(v21.stages,4);assert.equal(model.stages,3);
  const S=524288;
  assert.deepEqual(v21.weights.subarray(0,3*S),model.weights.subarray(0,3*S));
  assert.notDeepEqual(v21.weights.subarray(3*S),model.weights.subarray(2*S,3*S));
  assert.equal(stageForRanks([14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],4),3);
  assert.equal(stageForRanks([14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],3),2);
  assert.equal(stageForRanks([13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],4),2);
  // Below 6144 the two models give identical searches; with a 6144 on board they differ.
  const low=[[3072,1536,768,384],[6,12,24,48],[3,2,1,0],[0,0,0,0]], high=[[6144,1536,768,384],[6,12,24,48],[3,2,1,0],[0,0,0,0]];
  for (const [board,same] of [[low,true],[high,false]]) {
    const x={},y={};
    chooseFastMove(board,{candidates:[1]},model,{remaining:[3,3,3],maxDepth:2,threshold:0,...unlimited,stats:x});
    chooseFastMove(board,{candidates:[1]},v21,{remaining:[3,3,3],maxDepth:2,threshold:0,...unlimited,stats:y});
    if (same) assert.deepEqual(x.values,y.values); else assert.notDeepEqual(x.values,y.values);
  }
  // Malformed stage counts are rejected.
  const bad=new ArrayBuffer(16+5*S*4),v=new DataView(bad);v.setUint32(0,0x3144544e,true);v.setUint32(4,5,true);v.setUint32(8,8,true);v.setUint32(12,65536,true);
  assert.throws(()=>decodeModel(bad));
});

test('clampLeaf: a survivable move never ranks below certain death because of a negative leaf', () => {
  // Case from the V21 death audit: up dies for certain after the 3 enters, down survives.
  const board=[[768,2,6,3],[24,96,1536,3072],[768,48,12,3],[1,96,12,2]];
  const negative={value:()=>-1000};
  const run=clampLeaf=>{const stats={};const r=chooseFastMove(board,{candidates:[3]},negative,{remaining:[2,2,2],maxDepth:2,threshold:0,...unlimited,stats,clampLeaf});return {r,stats};};
  const off=run(false), on=run(true);
  assert.equal(off.r.direction,'up');   // unclamped: -1000 leaves make death look better
  assert.equal(on.r.direction,'down');
  assert.equal(on.stats.values[2],30);  // death: 27 merge + 3 card, no future
  assert(on.stats.values[3]>=on.stats.values[2]);
  // With non-negative leaves the clamp changes nothing.
  for (const s of states(4)) {
    const a={},b={};
    chooseFastMove(s.board,s.next,model,{remaining:s.remaining,maxDepth:3,threshold:0.01,...unlimited,stats:a});
    const pos={value:(x,st)=>Math.abs(model.value(x,st))};
    chooseFastMove(s.board,s.next,pos,{remaining:s.remaining,maxDepth:3,threshold:0.01,...unlimited,stats:a});
    chooseFastMove(s.board,s.next,pos,{remaining:s.remaining,maxDepth:3,threshold:0.01,...unlimited,stats:b,clampLeaf:true});
    assert.deepEqual(a.values,b.values);
  }
});

test('V23 website model: V4 stages 0-1, retrained stage 2, V21 stage 3', () => {
  const load=name=>{const b=readFileSync(new URL(`../dist/models/${name}`,import.meta.url));return decodeModel(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
  const v23=load('ntuple-v23.bin'),v21=load('ntuple-v21.bin'),S=524288;
  assert.equal(v23.stages,4);
  assert.deepEqual(v23.weights.subarray(0,2*S),model.weights.subarray(0,2*S));
  assert.notDeepEqual(v23.weights.subarray(2*S,3*S),model.weights.subarray(2*S,3*S));
  assert.deepEqual(v23.weights.subarray(3*S),v21.weights.subarray(3*S));
});
