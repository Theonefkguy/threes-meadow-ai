import {loadV8Policy,chooseMCTSMove} from './v8-policy.js?v=20';
import {loadV7Policy} from './v7-policy.js?v=20';
import {loadV6Policy,chooseV6Move} from './v6-policy.js?v=20';
import {loadV5Policy} from './v5-policy.js?v=20';
import {loadV4Policy} from './v4-policy.js?v=20';
import {loadV3Policy} from './v3-policy.js?v=20';
import { chooseMove } from './ai.js?v=20';
import { loadModel } from './rl-model.js?v=20';
import { chooseLearnedMove } from './rl-search.js?v=20';
import { loadV2Policy } from './v2-policy.js?v=20';
import { chooseGoalMove } from './goal-search.js?v=20';
import { chooseFastMove } from './fast-search.js?v=20';
self.onmessage = async ({ data }) => {
  try {
    const options={remaining:data.remaining};
    if(data.policy==='rl-fast'||data.policy==='rl'||data.policy==='rl-v2'||data.policy==='rl-v3'||data.policy==='rl-v4'||data.policy==='rl-v5'||data.policy==='rl-v6'||data.policy==='rl-v7'||data.policy==='rl-v8') {
      let result,policy=data.policy,fallback;
      if(policy==='rl-fast'){
        // Default (V18/V21): five-move search, probability cutoff 0.01, 160 ms soft budget,
        // on the V23 four-stage model (V4 stages 0-1, V23 stage 2, V21 stage 3 for 6144+);
        // V22: negative leaf estimates are clamped to 0 (true future score is never negative).
        let model;
        try{model=await loadModel('v23');}catch{policy='rl-v4';fallback='fast-load';}
        if(model){result=chooseFastMove(data.board,data.next,model,{...options,maxDepth:5,threshold:0.01,clampLeaf:true});if(result.unsupported){policy='rl-v4';result=undefined;}}
      }
      if(policy==='rl-v8'){
        let model;
        try{model=await loadV8Policy();}catch{policy='rl-v4';fallback='v8-load';}
        if(model)result=chooseMCTSMove(data.board,data.next,model.score,options);
      }
      if(policy==='rl-v7'){
        let model;
        try{model=await loadV7Policy();}catch{policy='rl-v4';fallback='v7-load';}
        if(model)result=chooseLearnedMove(data.board,data.next,model.score,options);
      }
      if(policy==='rl-v6'){
        let model;
        try{model=await loadV6Policy();}catch{policy='rl-v4';fallback='v6-load';}
        if(model)result=chooseV6Move(data.board,data.next,model,options);
      }
      if(policy==='rl-v5'){
        let model;
        try{model=await loadV5Policy();}catch{policy='rl-v4';fallback='v5-load';}
        if(model)result=chooseLearnedMove(data.board,data.next,model.score,options);
      }
      if(policy==='rl-v4'){
        let model;
        try{model=await loadV4Policy();}catch{policy='rl-v3';fallback='v4-load';}
        if(model)result=chooseLearnedMove(data.board,data.next,model.score,options);
      }
      if(policy==='rl-v3'){
        let model;
        try{model=await loadV3Policy();}catch{policy='rl-v2';fallback='v3-load';}
        if(model)result=chooseLearnedMove(data.board,data.next,model.score,options);
      }
      if(policy==='rl-v2'){
        let model;
        try{model=await loadV2Policy();}catch{policy='rl';fallback='v2-load';}
        if(model)result=model.kind==='goal'?chooseGoalMove(data.board,data.next,model.goal,model.score,options):chooseLearnedMove(data.board,data.next,model.score,options);
      }
      if(policy==='rl'){
        let model;
        try{model=await loadModel();}
        catch{self.postMessage({id:data.id,...chooseMove(data.board,data.next,options),policy:'classic',fallback:'load'});return;}
        result=chooseLearnedMove(data.board,data.next,model,options);
      }
      if(!result.unsupported){self.postMessage({id:data.id,...result,policy,...(fallback?{fallback}:{})});return;}
      self.postMessage({id:data.id,...chooseMove(data.board,data.next,options),policy:'classic',fallback:'range'});
    } else self.postMessage({id:data.id,...chooseMove(data.board,data.next,options),policy:'classic'});
  } catch {
    self.postMessage({ id:data.id, error:true });
  }
};
