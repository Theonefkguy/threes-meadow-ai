import fs from 'node:fs';
import {decodeModel} from '../../dist/rl-model.js';
import {decodeResidual} from '../../dist/v4-model.js';
import {chooseLearnedMove} from '../../dist/rl-search.js';
const buffer=p=>{const b=fs.readFileSync(p);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);},base=decodeModel(buffer('dist/models/ntuple-v4.bin')),config=JSON.parse(fs.readFileSync('dist/models/v4-policy.json'));
const models={v3:decodeModel(buffer('dist/models/ntuple-v3.bin')),v4:config.kind==='residual-five'?decodeResidual(buffer('dist/models/residual-v4.bin'),base):base};
const all=fs.readFileSync('training/v4/parity-selected.jsonl','utf8').trim().split('\n').map(JSON.parse),rows=[];
for(const stage of [0,1,2]){const g=all.filter(r=>{const h=Math.max(...r.ranks);return stage===0?h<12:stage===1?h===12:h>=13;});for(let i=0;i<g.length;i+=Math.max(1,Math.floor(g.length/30)))rows.push(g[i]);}
const val=r=>r<3?r:3*2**(r-3),out={v3:{ms:[],depths:[0,0,0,0],changed:0},v4:{ms:[],depths:[0,0,0,0],changed:0}};
for(const [i,r] of rows.entries()){
 const board=Array.from({length:4},(_,j)=>r.ranks.slice(j*4,j*4+4).map(val)),hint={candidates:r.cards.map(val),probabilities:r.weights};
 for(const n of i%2?['v4','v3']:['v3','v4']){const ideal=chooseLearnedMove(board,hint,models[n],{remaining:r.counts,budgetMs:Infinity});const start=performance.now(),a=chooseLearnedMove(board,hint,models[n],{remaining:r.counts,budgetMs:160});if(!a.direction)throw Error('illegal result');out[n].ms.push(performance.now()-start);out[n].depths[a.depth]++;out[n].changed+=a.direction!==ideal.direction;}
}
for(const n of ['v3','v4']){const a=out[n],ms=a.ms.sort((x,y)=>x-y);out[n]={positions:ms.length,meanMs:ms.reduce((x,y)=>x+y,0)/ms.length,p95Ms:ms[Math.floor(.95*(ms.length-1))],maxMs:ms.at(-1),depths:a.depths,changedFromUnlimited:a.changed};}
out.environment='Actual browser JS search in Node on this machine, alternating order; same leaf cache optimization for both models. Not a phone-specific success estimate.';out.compatible=out.v4.changedFromUnlimited/out.v4.positions<=out.v3.changedFromUnlimited/out.v3.positions+.05;
fs.writeFileSync('training/v4/timing.json',JSON.stringify(out,null,2)+'\n');console.log(out);
