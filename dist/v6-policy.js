import {loadModel} from './rl-model.js?v=20';
import {chooseLearnedMove} from './rl-search.js?v=20';
import {chooseEarlyGoalMove} from './v6-search.js?v=20';
let pending;
export function loadV6Policy(){return pending ||= (async()=>{
  const response=await fetch(new URL('./models/v6-policy.json',import.meta.url));
  if(!response.ok)throw new Error('策略加载失败');
  const config=await response.json();
  if(!['score','reach1536'].includes(config.kind))throw new Error('策略格式错误');
  const score=await loadModel('v6');
  return {score,kind:config.kind};
})().catch(error=>{pending=null;throw error;});}
export function chooseV6Move(board,next,model,options){
  return (model.kind==='reach1536'?chooseEarlyGoalMove:chooseLearnedMove)(board,next,model.score,options);
}
