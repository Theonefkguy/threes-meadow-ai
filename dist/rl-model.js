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
export function decodeResidual(buffer) {
  const view=new DataView(buffer),n=PATTERNS5.length*TABLE5;
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

// Model files per version: the four-cell base and any residual files (one per stage).
export const MODEL_FILES = {v4:{base:'ntuple-v4.bin',residuals:[]},v23:{base:'ntuple-v23.bin',residuals:[]}};
const pending=new Map();
export function loadModel(version) {
  const files=MODEL_FILES[version];
  if(!files)return Promise.reject(new Error('未知模型版本'));
  if(pending.has(version))return pending.get(version);
  const get=name=>fetch(new URL(`./models/${name}`,import.meta.url)).then(response=>{
    if(!response.ok)throw new Error('模型加载失败');
    return response.arrayBuffer();
  });
  const request=Promise.all([get(files.base),...files.residuals.map(get)])
    .then(([base,...residuals])=>decodeModel(base,residuals))
    .catch(error=>{pending.delete(version);throw error;});
  pending.set(version,request);return request;
}
