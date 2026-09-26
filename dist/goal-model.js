import { FEATURES, TABLE, PATTERNS, phaseForRanks } from './rl-model.js?v=20';
const BASE=PATTERNS.length*TABLE, CONTEXT=131072;
const sigmoid=z=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));
export function contextIndices(b,preview,counts) {
  let empty=0,ones=0,twos=0;
  for(const r of b){empty+=r===0;ones+=r===1;twos+=r===2;}
  const hint=(preview.cards.length-1)*16+preview.cards[0];
  return [(empty*48+hint)*125+counts[0]+5*counts[1]+25*counts[2],
    ((ones*17+twos)*48+hint)*9+counts[0]-counts[1]+4];
}
export function decodeGoalModel(buffer) {
  if(buffer.byteLength<16)throw new Error('目标模型不完整');
  const view=new DataView(buffer),extra=view.getUint32(12,true);
  if(view.getUint32(0,true)!==0x324c4f47||view.getUint32(4,true)!==2||view.getUint32(8,true)!==BASE||
    (extra!==0&&extra!==2*CONTEXT)||buffer.byteLength!==16+2*(BASE+extra)*4)throw new Error('目标模型格式不匹配');
  const weights=new Float32Array(buffer,16);
  for(const w of weights)if(!Number.isFinite(w))throw new Error('目标模型包含无效数值');
  return {contextual:!!extra,value(b,preview,counts){
    const stage=phaseForRanks(b);if(stage===2)return 1;
    const offset=stage*(BASE+extra);let z=0;
    for(let f=0;f<FEATURES.length;f++){
      const p=FEATURES[f],key=b[p[0]]|b[p[1]]<<4|b[p[2]]<<8|b[p[3]]<<12;
      z+=weights[offset+Math.floor(f/8)*TABLE+key];
    }
    if(extra){const ids=contextIndices(b,preview,counts);z+=weights[offset+BASE+ids[0]]+weights[offset+BASE+CONTEXT+ids[1]];}
    return sigmoid(z);
  }};
}
