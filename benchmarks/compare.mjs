import fs from 'node:fs';
import { createGame, score } from '../dist/engine.js';
import { chooseMove as baseline } from './baseline.js';
import { chooseMove, initialCounts, observePreview } from '../dist/ai-search.js';
const stage=process.argv[2]||'pilot';
const presets={baseline:{},fast:{mode:'classic'},memory:{mode:'classic',remember:true},space:{mode:'space',remember:true},chain:{mode:'chain',remember:true},deep:{mode:'space',remember:true,maxDepth:4}};
const names=process.argv[3]?.split(',')||Object.keys(presets);
const startSeed=stage==='pilot'?101:1001, count=Number(process.argv[4]||(stage==='pilot'?6:12));
const maxNodes=stage==='pilot'?6000:24000;
const output=`benchmarks/${stage}-results.json`;
const results=!process.argv.includes('--fresh') && fs.existsSync(output)?JSON.parse(fs.readFileSync(output)):[];
for(let offset=0;offset<count;offset++)for(const name of names){
 if(results.some(r=>r.name===name && r.seed===startSeed+offset))continue;
 let state=startSeed+offset;
 const rng=()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/4294967296);
 const game=createGame(rng);let remaining=initialCounts(game.board,game.next),depth=0,nodes=0,think=0,maxMs=0;
 const options={budgetMs:160,maxNodes,...presets[name]};
 while(!game.over && game.turns<2500){
  const start=performance.now();
  const choice=(name==='baseline'?baseline:chooseMove)(game.board,game.next,{...options,remaining});
  const ms=performance.now()-start;think+=ms;maxMs=Math.max(maxMs,ms);depth+=choice.depth;nodes+=choice.nodes;
  if(!game.move(choice.direction).changed)throw Error('invalid move');
  remaining=observePreview(remaining,game.next);
  if(remaining.some(n=>n<0||n>4))throw Error('invalid visible counts');
 }
 const result={stage,name,seed:startSeed+offset,score:score(game.board),highest:Math.max(...game.board.flat()),turns:game.turns,terminal:game.over,meanDepth:depth/game.turns,meanNodes:nodes/game.turns,meanMs:think/game.turns,maxMs};
 results.push(result);console.log(JSON.stringify(result));
 fs.writeFileSync(`benchmarks/${stage}-results.json`,JSON.stringify(results,null,2));
}
