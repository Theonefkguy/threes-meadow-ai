import fs from 'node:fs';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import { decodeModel,phaseForRanks } from '../dist/rl-model.js';
const path=process.argv[2]||'dist/models/ntuple-v1.bin';
const data=fs.readFileSync(path),model=decodeModel(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength));
let seed=92347;const rand=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
const cases=Array.from({length:60},()=>({board:Array.from({length:16},()=>Math.floor(rand()*16)),stage:Math.floor(rand()*3)}));
const result=execFileSync('training/bin/probe',{input:cases.map(c=>`value ${path} ${c.stage} ${c.board.join(' ')}`).join('\n'),encoding:'utf8'}).trim().split('\n').map(Number);
cases.forEach((c,i)=>{
 assert.ok(Math.abs(model.value(c.board,c.stage)-result[i])<1e-5,'browser/C++ value mismatch');
 const rotated=Array.from({length:16},(_,i)=>c.board[(3-i%4)*4+(i>>2)]);
 assert.ok(Math.abs(model.value(c.board,c.stage)-model.value(rotated,c.stage))<1e-5,'rotation changed value');
});
assert.deepEqual([11,12,13].map(r=>phaseForRanks([r])),[0,1,2]);
console.log('Export verified: 60 C++/JS predictions, rotation symmetry and all stage thresholds.');
