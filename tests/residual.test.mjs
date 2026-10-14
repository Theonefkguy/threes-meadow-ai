import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { chooseFastMove } from '../dist/fast-search.js';
import { decodeModel, decodeResidual, halfToNumber, RESIDUAL_MAGIC, PATTERNS5, TABLE5, FEATURES5, gunzipIfNeeded, readBody } from '../dist/rl-model.js';
import { gunzipSync, gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { createGame } from '../dist/engine.js';

const N=PATTERNS5.length*TABLE5;
// Synthetic web residual: deterministic float16 values (|v| <= 2^10) for one stage.
function residual(stage,{scaleExp=1,seed=7,magic=RESIDUAL_MAGIC,dtype=1}={}){
  const buf=new ArrayBuffer(32+N*2),v=new DataView(buf),h=new Uint16Array(buf,32,N);
  [magic,PATTERNS5.length,TABLE5,stage,dtype].forEach((x,i)=>v.setUint32(i*4,x,true));v.setInt32(20,scaleExp,true);
  let s=seed;for(let i=0;i<N;i++){s=(Math.imul(s,1664525)+1013904223)>>>0;h[i]=(s>>>16)&0x63ff;} // exponent <= 24: finite, small
  return buf;
}
const base=readFileSync(new URL('../dist/models/ntuple-v23.bin',import.meta.url));
const baseBuf=()=>base.buffer.slice(base.byteOffset,base.byteOffset+base.byteLength);

test('float16 decoding is exact for normals, subnormals, zero and specials',()=>{
  assert.equal(halfToNumber(0x3c00),1);assert.equal(halfToNumber(0xc000),-2);assert.equal(halfToNumber(0x7bff),65504);
  assert.equal(halfToNumber(0x0001),2**-24);assert.equal(Object.is(halfToNumber(0x8000),-0),true);
  assert.equal(halfToNumber(0x7c00),Infinity);assert.ok(Number.isNaN(halfToNumber(0x7e00)));
});

test('residual files are validated and scaled by a power of two',()=>{
  const r=decodeResidual(residual(2,{scaleExp:3}));assert.equal(r.stage,2);assert.equal(r.weights.length,N);
  const raw=new Uint16Array(residual(2,{scaleExp:3}),32,8);
  for(let i=0;i<8;i++)assert.equal(r.weights[i],halfToNumber(raw[i])*8);
  assert.throws(()=>decodeResidual(residual(2,{magic:1})));assert.throws(()=>decodeResidual(residual(2,{dtype:2})));
  assert.throws(()=>decodeResidual(residual(0)));assert.throws(()=>decodeResidual(new ArrayBuffer(64)));
  const inf=residual(1);new Uint16Array(inf,32,1)[0]=0x7c00;assert.throws(()=>decodeResidual(inf));
  assert.throws(()=>decodeModel(baseBuf(),[residual(2),residual(2,{seed:9})]));
});

test('residual value adds the five-cell sum to the four-cell value of its stage only',()=>{
  const plain=decodeModel(baseBuf()),model=decodeModel(baseBuf(),[residual(2)]);
  const b=[13,2,3,4,5,6,7,8,1,0,2,3,4,5,6,7];
  let r=0;for(let f=0;f<FEATURES5.length;f++){const p=FEATURES5[f];r+=model.residuals[2][(f>>3)*TABLE5+(b[p[0]]|b[p[1]]<<4|b[p[2]]<<8|b[p[3]]<<12|b[p[4]]<<16)];}
  assert.equal(model.value(b),plain.value(b)+r);assert.notEqual(r,0);
  const low=b.map(x=>Math.min(x,12));assert.equal(model.value(low),plain.value(low));
});

test('fast leaf path with residuals equals the generic model.value path',()=>{
  const model=decodeModel(baseBuf(),[residual(1,{seed:3}),residual(2),residual(3,{seed:5})]);
  const slow={stages:model.stages,value:(b,s)=>model.value(b,s)};
  const rng=(seed=>()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296))(11);
  for(const high of [1536,3072,6144]){
    const g=createGame(rng);g.board[0][0]=high;
    for(const threshold of [0,0.01]){
      const x={},y={},o={maxDepth:3,threshold,clampLeaf:true,budgetMs:1e9,maxNodes:1e12};
      const a=chooseFastMove(g.board,g.next,model,{...o,stats:x}),c=chooseFastMove(g.board,g.next,slow,{...o,stats:y});
      assert.equal(a.direction,c.direction);assert.deepEqual(x.values,y.values);
    }
  }
});

test('the shipped V31 gzip files decode to a model with residuals for stages 1-3 and match their declared size',async()=>{
  const {MODEL_FILES}=await import('../dist/rl-model.js');const f=MODEL_FILES.v31;
  const names=[f.base,...f.residuals];assert(names.every(n=>n.endsWith('.gz')));
  const gz=names.map(n=>readFileSync(new URL(`../dist/models/${n}`,import.meta.url)));
  assert.equal(gz.reduce((s,b)=>s+b.byteLength,0),f.bytes);
  const files=await Promise.all(gz.map(b=>gunzipIfNeeded(new Uint8Array(b.buffer,b.byteOffset,b.byteLength))));
  files.forEach((buf,i)=>assert.deepEqual(new Uint8Array(buf),new Uint8Array(gunzipSync(gz[i]))));
  const v31=decodeModel(files[0],files.slice(1)),v23=decodeModel(baseBuf()),S=524288;
  assert.deepEqual(v31.residuals.map(r=>r!==null),[false,true,true,true]);
  assert.deepEqual(v31.weights.subarray(0,S),v23.weights.subarray(0,S));          // stage 0 unchanged since V4
  for(let s=1;s<4;s++)assert.notDeepEqual(v31.weights.subarray(s*S,(s+1)*S),v23.weights.subarray(s*S,(s+1)*S));
  // SHA-256 of the decoded files equals the evaluated V31 release (training/web-release/v31-check/README.md).
  const hash=b=>createHash('sha256').update(new Uint8Array(b)).digest('hex');
  assert.equal(hash(files[0]),'b94a7da2671fe28ad3390a5fcd2fbda59a599f50437b23d5e3cf7097e45f9617');
  assert.deepEqual(files.slice(1).map(b=>hash(b).slice(0,16)),['ef8673c9a38d6452','3d77ea6bb2643b3b','cc4ab3e129eaf3a6']);
});

test('gzip decoding passes plain bytes through and reads streamed bodies with byte counts',async()=>{
  const plain=new Uint8Array([1,2,3,4]);assert.deepEqual(new Uint8Array(await gunzipIfNeeded(plain)),plain);
  const packed=gzipSync(Buffer.from('threes'));assert.equal(Buffer.from(await gunzipIfNeeded(new Uint8Array(packed))).toString(),'threes');
  let counted=0;const body=await readBody(new Response(packed),n=>counted+=n);
  assert.equal(counted,packed.byteLength);assert.deepEqual(Buffer.from(body),packed);
});
