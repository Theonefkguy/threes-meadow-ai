import {Worker} from 'node:worker_threads';
import assert from 'node:assert/strict';
import {createGame,projectMove} from '../../dist/engine.js';
import {initialCounts} from '../../dist/card-memory.js';
const entry=new URL('../../dist/ai-worker.js?v=9',import.meta.url).href;
function start(fail){return new Worker(`const {parentPort}=require('node:worker_threads');globalThis.self={postMessage:d=>parentPort.postMessage(d)};globalThis.fetch=async url=>{const s=String(url);if(${JSON.stringify(fail)}==='all'||(${JSON.stringify(fail)}==='v3'&&s.includes('v3')))throw Error('offline');const b=await require('node:fs/promises').readFile(url);return{ok:true,json:async()=>JSON.parse(b.toString()),arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)}};import(${JSON.stringify(entry)}).then(()=>{parentPort.on('message',d=>self.onmessage({data:d}));parentPort.postMessage({ready:true})});`,{eval:true});}
function wait(w){return new Promise((resolve,reject)=>{const t=setTimeout(()=>reject(Error('worker timeout')),30000);w.once('message',m=>{clearTimeout(t);resolve(m)});w.once('error',reject)});}
const g=createGame(()=>.5),results=[];
for(const fail of [false,'v3','all']){const w=start(fail);await wait(w);for(const policy of ['rl-v3','rl-v2','rl','classic']){const p=wait(w);w.postMessage({id:policy,policy,board:g.board,next:g.next,remaining:initialCounts(g.board,g.next)});const r=await p;assert.equal(r.id,policy);assert(projectMove(g.board,r.direction).changed);const expected=policy==='classic'?'classic':fail==='all'?'classic':fail==='v3'&&policy==='rl-v3'?'rl-v2':policy;assert.equal(r.policy,expected);if(fail==='v3'&&policy==='rl-v3')assert.equal(r.fallback,'v3-load');results.push({fail,requested:policy,actual:r.policy});}await w.terminate();}
console.log(JSON.stringify({passed:results.length,cases:results}));
