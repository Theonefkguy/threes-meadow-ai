import {Worker} from 'node:worker_threads';
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createGame,projectMove} from '../../dist/engine.js';
import {initialCounts} from '../../dist/card-memory.js';
const entry=new URL('../../dist/ai-worker.js?v=11',import.meta.url).href;
function start(failed){return new Worker(`const {parentPort}=require('node:worker_threads');globalThis.self={postMessage:d=>parentPort.postMessage(d)};globalThis.fetch=async url=>{const s=String(url),failed=${JSON.stringify(failed)};if(failed.includes('all')||failed.some(v=>s.includes(v)))throw Error('offline');const b=await require('node:fs/promises').readFile(url);return{ok:true,json:async()=>JSON.parse(b.toString()),arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)}};import(${JSON.stringify(entry)}).then(()=>{parentPort.on('message',d=>self.onmessage({data:d}));parentPort.postMessage({ready:true})});`,{eval:true});}
function wait(w){return new Promise((resolve,reject)=>{const t=setTimeout(()=>reject(Error('worker timeout')),30000);w.once('message',m=>{clearTimeout(t);resolve(m)});w.once('error',reject)});}
const g=createGame(()=>.5),rows=[],policies=['rl-v5','rl-v4','rl-v3','rl-v2','rl','classic'];
for(const failed of [[],['v5'],['v5','v4'],['v5','v4','v3'],['all']]){
 const w=start(failed);await wait(w);
 for(const policy of policies){const p=wait(w);w.postMessage({id:policy,policy,board:g.board,next:g.next,remaining:initialCounts(g.board,g.next)});const r=await p;assert.equal(r.id,policy);assert(projectMove(g.board,r.direction).changed);
  const expected=failed.includes('all')?'classic':policies.slice(policies.indexOf(policy)).find(n=>!failed.includes(n.replace('rl-','')));assert.equal(r.policy,expected);rows.push({failed,requested:policy,actual:r.policy});
 }
 await w.terminate();
}
fs.writeFileSync('training/v5/worker-results.json',JSON.stringify({passed:rows.length,cases:rows},null,2)+'\n');console.log('PASS '+rows.length+' actual worker load/fallback cases');
