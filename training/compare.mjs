import fs from 'node:fs';
import { createGame, score } from '../dist/engine.js';
import { initialCounts, observePreview } from '../dist/card-memory.js';
import { chooseMove } from '../dist/ai.js';
import { chooseLearnedMove } from '../dist/rl-search.js';
import { decodeModel } from '../dist/rl-model.js';
const modelPath=process.argv[2]||'dist/models/ntuple-v1.bin';
const count=Number(process.argv[3]||12),start=Number(process.argv[4]||810001);
const output=process.argv[5]||'training/evaluation-v1.json';
const names=(process.argv[6]||'classic,single,staged').split(',');
const bytes=fs.readFileSync(modelPath),model=decodeModel(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
const results=fs.existsSync(output)?JSON.parse(fs.readFileSync(output)):[];
for(let i=0;i<count;i++)for(const name of names){
 const seed=start+i;if(results.some(r=>r.seed===seed&&r.name===name))continue;
 let s=seed;const rng=()=>((s=(Math.imul(s,1664525)+1013904223)>>>0)/4294967296),game=createGame(rng);
 let remaining=initialCounts(game.board,game.next),totalMs=0,maxMs=0,nodes=0,depth=0;
 while(!game.over&&game.turns<6000){
  const options={remaining,budgetMs:160,maxNodes:24000,maxDepth:3,singleStage:name==='single'},start=performance.now();
  const result=name==='classic'?chooseMove(game.board,game.next,options):chooseLearnedMove(game.board,game.next,model,options);
  const ms=performance.now()-start;totalMs+=ms;maxMs=Math.max(maxMs,ms);nodes+=result.nodes;depth+=result.depth;
  if(!game.move(result.direction).changed)throw new Error('AI returned illegal move');remaining=observePreview(remaining,game.next);
 }
 const r={name,seed,score:score(game.board),highest:Math.max(...game.board.flat()),turns:game.turns,terminal:game.over,meanMs:totalMs/game.turns,maxMs,meanNodes:nodes/game.turns,meanDepth:depth/game.turns};
 results.push(r);fs.writeFileSync(output,JSON.stringify(results,null,2)+'\n');console.log(JSON.stringify(r));
}
