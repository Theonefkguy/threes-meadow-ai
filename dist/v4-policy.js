import {loadModel} from './rl-model.js?v=20';
import {decodeResidual} from './v4-model.js?v=20';
let pending;
export function loadV4Policy(){return pending ||= (async()=>{
 const reply=await fetch(new URL('./models/v4-policy.json',import.meta.url));if(!reply.ok)throw Error('策略加载失败');const config=await reply.json();if(!['score','residual-five'].includes(config.kind))throw Error('策略格式错误');
 const base=await loadModel('v4');if(config.kind==='score')return {score:base};
 const response=await fetch(new URL('./models/residual-v4.bin',import.meta.url));if(!response.ok)throw Error('五格模型加载失败');return {score:decodeResidual(await response.arrayBuffer(),base)};
})().catch(error=>{pending=null;throw error;});}
