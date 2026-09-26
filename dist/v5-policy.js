import {loadModel} from './rl-model.js?v=20';
import {withCornerPotential} from './v5-model.js?v=20';
let pending;
export function loadV5Policy(){return pending ||= (async()=>{
  const response=await fetch(new URL('./models/v5-policy.json',import.meta.url));
  if(!response.ok)throw new Error('策略加载失败');
  const config=await response.json();
  if(config.kind!=='corner-potential')throw new Error('策略格式错误');
  const base=await loadModel('v5');
  return {score:withCornerPotential(base,config.coefficient)};
})().catch(error=>{pending=null;throw error;});}
