import {loadModel} from './rl-model.js?v=20';
import {decodeGoalModel} from './goal-model.js?v=20';
let pending;
export function loadV2Policy(){
  return pending ||= (async()=>{
    const response=await fetch(new URL('./models/v2-policy.json',import.meta.url));
    if(!response.ok)throw new Error('策略加载失败');
    const config=await response.json();
    if(!['score','goal'].includes(config.kind)||!['v1','v2'].includes(config.scoreVersion))throw new Error('策略格式错误');
    const score=await loadModel(config.scoreVersion);let goal=null;
    if(config.kind==='goal'){
      const response=await fetch(new URL('./models/goal-v2.bin',import.meta.url));
      if(!response.ok)throw new Error('目标模型加载失败');
      goal=decodeGoalModel(await response.arrayBuffer());
    }
    return {kind:config.kind,score,goal};
  })().catch(error=>{pending=null;throw error;});
}
