import fs from 'node:fs';
import assert from 'node:assert/strict';
import {decodeModel} from '../../dist/rl-model.js';
import {decodeResidual} from '../../dist/v4-model.js';
const read=p=>{const b=fs.readFileSync(p);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const base=decodeModel(read('/tmp/threes-v4-model-check.ntd')),model=decodeResidual(read('/tmp/threes-v4-model-check.five'),base);
const b=[13,0,0,0,12,0,0,0,1,2,3,4,0,0,0,0];
assert(Math.abs(model.value(b,2)-base.value(b,2)-10)<1e-5);assert.equal(model.value(b,0),base.value(b,0));assert.equal(model.value(b),model.value(b,2));
for(let sym=0;sym<8;sym++){const t=Array(16).fill(0);for(let i=0;i<16;i++){let x=i%4,y=i>>2;if(sym>=4)x=3-x;for(let k=0;k<sym%4;k++)[y,x]=[x,3-y];t[y*4+x]=b[i];}assert(Math.abs(model.value(t,2)-model.value(b,2))<1e-6);}
assert.throws(()=>decodeResidual(new ArrayBuffer(8),base));const bad=read('/tmp/threes-v4-model-check.five');new DataView(bad).setFloat32(16,NaN,true);assert.throws(()=>decodeResidual(bad,base));
console.log('Browser residual decoder: native fixture, stage isolation, D4 symmetry, invalid/truncated model rejection pass.');
