// Runs the actual browser worker entry in a Node worker, with only a local
// model-file fetch bridge. UI lifecycle is covered by input.test.mjs.
import { Worker } from 'node:worker_threads';
import assert from 'node:assert/strict';
import { createGame,projectMove } from '../dist/engine.js';
import { initialCounts } from '../dist/card-memory.js';
const entry=new URL('../dist/ai-worker.js?v=8',import.meta.url).href;
function start(fail=false){
 const worker=new Worker(`const {parentPort}=require('node:worker_threads');
 globalThis.self={postMessage:data=>parentPort.postMessage(data)};
 globalThis.fetch=async url=>{if(${JSON.stringify(fail)}==='all'||(${JSON.stringify(fail)}==='v2'&&!String(url).includes('ntuple-v1.bin')))throw new Error('offline'); const fs=require('node:fs/promises');const b=await fs.readFile(url);return {ok:true,json:async()=>JSON.parse(b.toString()),arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};};
 import(${JSON.stringify(entry)}).then(()=>{parentPort.on('message',data=>self.onmessage({data}));parentPort.postMessage({ready:true});});`,{eval:true});
 return worker;
}
function wait(worker){return new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(new Error('worker timed out')),30000);worker.once('message',message=>{clearTimeout(timeout);resolve(message);});worker.once('error',reject);});}
const game=createGame(()=>.5);
for(const fail of [false,'v2','all']){
 const worker=start(fail);await wait(worker);
 for(const policy of ['rl-v2','rl','classic']){
  const response=wait(worker);worker.postMessage({id:policy,board:game.board,next:game.next,remaining:initialCounts(game.board,game.next),policy});
  const r=await response;assert.equal(r.id,policy);assert(projectMove(game.board,r.direction).changed);assert.equal(r.policy,policy==='classic'?'classic':fail==='all'?'classic':fail==='v2'?'rl':policy);
  if(fail==='all'&&policy.startsWith('rl'))assert.equal(r.fallback,'load');
  if(fail==='v2'&&policy==='rl-v2')assert.equal(r.fallback,'v2-load');
 }
 await worker.terminate();
}
console.log('Actual worker: learned inference, classic mode, and failed-load fallback passed.');
