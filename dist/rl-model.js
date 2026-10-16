// Compact N-tuple afterstate value model; trained offline with actual game rules.
export const PATTERNS = [[0,1,2,3],[4,5,6,7],[0,1,4,5],[1,2,5,6],
  [5,6,9,10],[0,1,2,4],[0,1,5,6],[0,1,4,8]];
export const TABLE = 65536;
const symmetries = patterns => patterns.flatMap(pattern => Array.from({length:8},(_,sym)=>pattern.map(pos=>{
  let y=pos>>2,x=pos&3;
  if(sym>=4)x=3-x;
  for(let k=0;k<sym%4;k++)[y,x]=[x,3-y];
  return y*4+x;
})));
export const FEATURES = symmetries(PATTERNS);
// V29+: an optional five-cell residual per stage (training/v29-five/model5.h): four five-cell
// patterns x 8 symmetries, 16^5 entries each, added to that stage's four-cell value.
export const PATTERNS5 = [[0,1,2,3,4],[4,5,6,7,8],[0,1,2,4,5],[4,5,6,8,9]];
export const TABLE5 = 1<<20;
export const FEATURES5 = symmetries(PATTERNS5);
export const phaseForRanks = b => Math.max(...b)>=13?2:Math.max(...b)>=12?1:0;
// Four-stage models (V21+) add stage 3 for afterstates holding a 6144 or larger.
export const stageForRanks = (b,stages=3) => {const h=Math.max(...b);return stages===4&&h>=14?3:h>=13?2:h>=12?1:0;};

// Exact IEEE 754 binary16 -> number.
export function halfToNumber(h) {
  const sign=h&0x8000?-1:1,e=h>>10&31,m=h&1023;
  if(e===0)return sign*m*2**-24;
  if(e===31)return m?NaN:sign*Infinity;
  return sign*(1024+m)*2**(e-25);
}

// Web residual file: 32-byte header {0x3548444e, patterns, table, stage, dtype=1 (float16),
// scaleExp (int32), 0, 0} then patterns*table little-endian float16 values; each weight is
// half * 2^scaleExp (scripts/quantize-residual.py). Decoded to float32, which holds these
// values exactly, so native evaluation of the same quantized weights matches bit for bit.
export const RESIDUAL_MAGIC = 0x3548444e;
// Full-precision residual (the native training format, model5s.h): 16-byte header {0x3554444e,
// patterns, table, stage} then patterns*table little-endian float32 values, used as they are.
export const NATIVE_RESIDUAL_MAGIC = 0x3554444e;
export function decodeResidual(buffer) {
  const view=new DataView(buffer),n=PATTERNS5.length*TABLE5;
  if(buffer.byteLength>=16&&view.getUint32(0,true)===NATIVE_RESIDUAL_MAGIC){
    if(buffer.byteLength!==16+n*4||view.getUint32(4,true)!==PATTERNS5.length||view.getUint32(8,true)!==TABLE5)throw new Error('残差格式不匹配');
    const stage=view.getUint32(12,true);if(stage<1||stage>3)throw new Error('残差阶段无效');
    const weights=new Float32Array(buffer.slice(16));
    for(const v of weights)if(!Number.isFinite(v))throw new Error('残差包含无效数值');
    return {stage,weights};
  }
  if(buffer.byteLength!==32+n*2 || view.getUint32(0,true)!==RESIDUAL_MAGIC || view.getUint32(4,true)!==PATTERNS5.length ||
    view.getUint32(8,true)!==TABLE5 || view.getUint32(16,true)!==1) throw new Error('残差格式不匹配');
  const stage=view.getUint32(12,true),scale=2**view.getInt32(20,true);
  if(stage<1||stage>3)throw new Error('残差阶段无效');
  const half=new Uint16Array(buffer,32,n),table=new Float32Array(65536);
  for(let h=0;h<65536;h++)table[h]=halfToNumber(h)*scale;
  const weights=new Float32Array(n);
  for(let i=0;i<n;i++){const v=table[half[i]];if(!Number.isFinite(v))throw new Error('残差包含无效数值');weights[i]=v;}
  return {stage,weights};
}

export function decodeModel(buffer,residualBuffers=[]) {
  const view=new DataView(buffer);
  const stages=buffer.byteLength>=8?view.getUint32(4,true):0;
  if((stages!==3&&stages!==4) || buffer.byteLength!==16+stages*PATTERNS.length*TABLE*4 || view.getUint32(0,true)!==0x3144544e ||
    view.getUint32(8,true)!==PATTERNS.length || view.getUint32(12,true)!==TABLE) {
    throw new Error('模型格式不匹配');
  }
  const weights=new Float32Array(buffer,16);
  for(const v of weights)if(!Number.isFinite(v))throw new Error('模型包含无效数值');
  const residuals=Array(stages).fill(null);
  for(const r of residualBuffers.map(decodeResidual)){
    if(r.stage>=stages||residuals[r.stage])throw new Error('残差阶段无效');
    residuals[r.stage]=r.weights;
  }
  return {
    weights, stages, residuals,
    // Same arithmetic as the native ModelS: four-cell sum, then (if present) the residual
    // summed separately and added.
    value(b,stage=stageForRanks(b,stages)) {
      let v=0;const offset=stage*PATTERNS.length*TABLE;
      for(let f=0;f<FEATURES.length;f++) {
        const p=FEATURES[f];
        const key=Math.min(15,b[p[0]]) | Math.min(15,b[p[1]])<<4 |
          Math.min(15,b[p[2]])<<8 | Math.min(15,b[p[3]])<<12;
        v+=weights[offset+Math.floor(f/8)*TABLE+key];
      }
      const R=residuals[stage];
      if(R){
        let r=0;
        for(let f=0;f<FEATURES5.length;f++){
          const p=FEATURES5[f];
          r+=R[(f>>3)*TABLE5+(Math.min(15,b[p[0]])|Math.min(15,b[p[1]])<<4|Math.min(15,b[p[2]])<<8|Math.min(15,b[p[3]])<<12|Math.min(15,b[p[4]])<<16)];
        }
        v+=r;
      }
      return v;
    }
  };
}

// Model files per version: the four-cell base and any residual files (one per stage), with
// the exact bytes to download so progress works even without a Content-Length header.
// Files ending in .gz are gzip-compressed copies (scripts/compress-models.py) that decode to
// the identical model bytes.
export const MODEL_FILES = {
  v4:{base:'ntuple-v4.bin',residuals:[]},
  v23:{base:'ntuple-v23.bin',residuals:[]},
  // V31: four-cell tables plus float16 five-cell residuals for stages 1-3 (RL-V31-STAGES.md),
  // 32 MB decoded, 17.4 MB downloaded.
  v31:{base:'ntuple-v31.bin.gz',residuals:['ntuple-v31-s1.f16.gz','ntuple-v31-s2.f16.gz','ntuple-v31-s3.f16.gz'],
    bytes:5040193+2993234+4262781+5955585,cache:'threes-models-v31-b94a7da2'},
  // V31 full precision: the same four-cell tables plus the float32 residuals exactly as trained and
  // evaluated natively (S1-final, V30 P-final, S3-final), 64 MB decoded, 30.4 MB downloaded.
  v31f:{base:'ntuple-v31.bin.gz',residuals:['ntuple-v31-s1.res.gz','ntuple-v31-s2.res.gz','ntuple-v31-s3.res.gz'],
    bytes:5040193+5720853+8153302+11480934,cache:'threes-models-v31f-a5c1d053'}
};
// Gzip bytes (magic 1f 8b) are decompressed with the browser's DecompressionStream; anything
// else is returned unchanged (e.g. when a server already removed a gzip Content-Encoding).
export async function gunzipIfNeeded(bytes) {
  if(bytes.byteLength<2||bytes[0]!==0x1f||bytes[1]!==0x8b)return bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);
  if(typeof DecompressionStream!=='function')throw new Error('浏览器不支持解压模型');
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
// Reads a response body, reporting each received chunk's size through onBytes(n).
export async function readBody(response,onBytes=()=>{}) {
  if(!response.body){const buf=new Uint8Array(await response.arrayBuffer());onBytes(buf.byteLength);return buf;}
  const reader=response.body.getReader(),chunks=[];let size=0;
  for(;;){const {done,value}=await reader.read();if(done)break;chunks.push(value);size+=value.byteLength;onBytes(value.byteLength);}
  const out=new Uint8Array(size);let at=0;for(const c of chunks){out.set(c,at);at+=c.byteLength;}
  return out;
}
// Downloaded files of versions with a `cache` name are kept in Cache Storage under that name, so
// a page reload or a site redeploy (which changes ETags) does not download them again. The name
// changes whenever the model files change; other threes-models-* caches are then deleted. Any
// Cache Storage failure (private mode, quota) silently falls back to the network.
const cacheStore=name=>typeof caches==='object'&&name?caches.open(name).catch(()=>null):Promise.resolve(null);
// Caches of every current version are kept (standard and full precision can both stay downloaded).
const CURRENT_CACHES=new Set(Object.values(MODEL_FILES).map(f=>f.cache).filter(Boolean));
async function dropOtherCaches(){try{for(const k of await caches.keys())if(k.startsWith('threes-models-')&&!CURRENT_CACHES.has(k))await caches.delete(k);}catch{/* Optional. */}}
const pending=new Map();
// onProgress({loaded,total}) is called as bytes arrive (for every caller while a load is shared).
export function loadModel(version,onProgress) {
  const files=MODEL_FILES[version];
  if(!files)return Promise.reject(new Error('未知模型版本'));
  let entry=pending.get(version);
  if(!entry){
    const listeners=new Set(),total=files.bytes||0;let loaded=0;
    const report=()=>{for(const f of listeners)try{f({loaded,total});}catch{/* Progress is optional. */}};
    const store=cacheStore(files.cache);
    const get=async name=>{
      const url=new URL(`./models/${name}`,import.meta.url).href,cache=await store;
      let response=cache&&await cache.match(url).catch(()=>undefined);const cached=!!response;
      if(!response){response=await fetch(url);if(!response.ok)throw new Error('模型加载失败');}
      const bytes=await readBody(response,n=>{loaded+=n;if(total)report();});
      if(cache&&!cached)await cache.put(url,new Response(bytes,{headers:{'content-type':'application/octet-stream'}})).catch(()=>{});
      return name.endsWith('.gz')?gunzipIfNeeded(bytes):bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);
    };
    const request=Promise.all([get(files.base),...files.residuals.map(get)])
      .then(([base,...residuals])=>{const model=decodeModel(base,residuals);if(files.cache)dropOtherCaches(files.cache);return model;})
      .catch(async error=>{
        pending.delete(version);
        // A damaged cached copy must not block later attempts.
        if(files.cache&&typeof caches==='object')await caches.delete(files.cache).catch(()=>{});
        throw error;
      });
    entry={request,listeners};pending.set(version,entry);
  }
  if(onProgress){entry.listeners.add(onProgress);entry.request.finally(()=>entry.listeners.delete(onProgress)).catch(()=>{});}
  return entry.request;
}
