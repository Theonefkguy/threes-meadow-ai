import assert from 'node:assert/strict';
import {cornerPotential,withCornerPotential} from '../../dist/v5-model.js';
const corner=Array(16).fill(0);corner[0]=13;corner[5]=12;
const middle=corner.slice();[middle[0],middle[5]]=[middle[5],middle[0]];
const base={value:()=>100};
for(const c of [0,2048,8192,32768]){
 const m=withCornerPotential(base,c);
 assert.equal(m.value(corner),100+c);assert.equal(m.value(middle),100);assert.equal(m.value(corner,1),100);
 for(let sym=0;sym<8;sym++){
  const b=Array(16).fill(0);
  for(let i=0;i<16;i++){let y=i>>2,x=i&3;if(sym>=4)x=3-x;for(let k=0;k<sym%4;k++)[y,x]=[x,3-y];b[y*4+x]=corner[i];}
  assert.equal(m.value(b),100+c);
 }
}
for(const c of [-1,NaN,Infinity,'8192'])assert.throws(()=>withCornerPotential(base,c));
const early=corner.slice();early[0]=12;assert.equal(cornerPotential(early),0);
console.log('PASS corrected value, zero coefficient identity, early-stage isolation, D4 and invalid config');
