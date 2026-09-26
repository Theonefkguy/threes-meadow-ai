// Compact N-tuple afterstate value model; trained offline with actual game rules.
export const PATTERNS = [[0,1,2,3],[4,5,6,7],[0,1,4,5],[1,2,5,6],
  [5,6,9,10],[0,1,2,4],[0,1,5,6],[0,1,4,8]];
export const TABLE = 65536;
export const FEATURES = PATTERNS.flatMap(pattern => Array.from({length:8},(_,sym)=>pattern.map(pos=>{
  let y=pos>>2,x=pos&3;
  if(sym>=4)x=3-x;
  for(let k=0;k<sym%4;k++)[y,x]=[x,3-y];
  return y*4+x;
})));
export const phaseForRanks = b => Math.max(...b)>=13?2:Math.max(...b)>=12?1:0;
// Four-stage models (V21+) add stage 3 for afterstates holding a 6144 or larger.
export const stageForRanks = (b,stages=3) => {const h=Math.max(...b);return stages===4&&h>=14?3:h>=13?2:h>=12?1:0;};

export function decodeModel(buffer) {
  const view=new DataView(buffer);
  const stages=buffer.byteLength>=8?view.getUint32(4,true):0;
  if((stages!==3&&stages!==4) || buffer.byteLength!==16+stages*PATTERNS.length*TABLE*4 || view.getUint32(0,true)!==0x3144544e ||
    view.getUint32(8,true)!==PATTERNS.length || view.getUint32(12,true)!==TABLE) {
    throw new Error('模型格式不匹配');
  }
  const weights=new Float32Array(buffer,16);
  for(const v of weights)if(!Number.isFinite(v))throw new Error('模型包含无效数值');
  return {
    weights, stages,
    value(b,stage=stageForRanks(b,stages)) {
      let v=0;const offset=stage*PATTERNS.length*TABLE;
      for(let f=0;f<FEATURES.length;f++) {
        const p=FEATURES[f];
        const key=Math.min(15,b[p[0]]) | Math.min(15,b[p[1]])<<4 |
          Math.min(15,b[p[2]])<<8 | Math.min(15,b[p[3]])<<12;
        v+=weights[offset+Math.floor(f/8)*TABLE+key];
      }
      return v;
    }
  };
}

const pending=new Map();
export function loadModel(version='v1') {
  if(!['v1','v2','v3','v4','v5','v6','v7','v8','v21','v23'].includes(version))return Promise.reject(new Error('未知模型版本'));
  if(pending.has(version))return pending.get(version);
  const request=fetch(new URL(`./models/ntuple-${version}.bin`,import.meta.url)).then(async response=>{
    if(!response.ok)throw new Error('模型加载失败');
    return decodeModel(await response.arrayBuffer());
  }).catch(error=>{pending.delete(version);throw error;});
  pending.set(version,request);return request;
}
