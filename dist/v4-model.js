const TABLE=1048576;
const PATTERNS=[[0,1,2,3,4],[0,1,2,4,5]];
const FEATURES=PATTERNS.flatMap(pattern=>Array.from({length:8},(_,sym)=>pattern.map(pos=>{
 let y=pos>>2,x=pos&3;if(sym>=4)x=3-x;for(let k=0;k<sym%4;k++)[y,x]=[x,3-y];return y*4+x;
})));
export function decodeResidual(buffer,base){
 const view=new DataView(buffer);
 if(buffer.byteLength!==16+2*TABLE*4||view.getUint32(0,true)!==0x3544544e||view.getUint32(4,true)!==2||view.getUint32(8,true)!==TABLE||view.getUint32(12,true)!==5)throw Error('五格模型格式不匹配');
 const weights=new Float32Array(buffer,16);for(const x of weights)if(!Number.isFinite(x))throw Error('模型包含无效数值');
 return {value(b,stage=Math.max(...b)>=13?2:Math.max(...b)>=12?1:0){let value=base.value(b,stage);if(stage===2)for(let f=0;f<FEATURES.length;f++){
  const p=FEATURES[f],key=Math.min(15,b[p[0]])|Math.min(15,b[p[1]])<<4|Math.min(15,b[p[2]])<<8|Math.min(15,b[p[3]])<<12|Math.min(15,b[p[4]])<<16;
  value+=weights[Math.floor(f/8)*TABLE+key];
 }return value;}};
}
