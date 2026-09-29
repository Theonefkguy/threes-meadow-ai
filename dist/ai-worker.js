import {loadV4Policy} from './v4-policy.js?v=21';
import { chooseMove } from './ai.js?v=21';
import { loadModel } from './rl-model.js?v=21';
import { chooseLearnedMove } from './rl-search.js?v=21';
import { chooseFastMove } from './fast-search.js?v=21';
self.onmessage = async ({ data }) => {
  try {
    const options={remaining:data.remaining};
    if(data.policy==='rl-fast'||data.policy==='rl-v4') {
      let result,policy=data.policy,fallback;
      if(policy==='rl-fast'){
        // Default (V18/V21): five-move search, probability cutoff 0.01, 160 ms soft budget,
        // on the V23 four-stage model (V4 stages 0-1, V23 stage 2, V21 stage 3 for 6144+);
        // V22: negative leaf estimates are clamped to 0 (true future score is never negative).
        let model;
        try{model=await loadModel('v23');}catch{policy='rl-v4';fallback='fast-load';}
        if(model){result=chooseFastMove(data.board,data.next,model,{...options,maxDepth:5,threshold:0.01,clampLeaf:true});if(result.unsupported){policy='rl-v4';result=undefined;}}
      }
      if(policy==='rl-v4'){
        let model;
        try{model=await loadV4Policy();}
        catch{self.postMessage({id:data.id,...chooseMove(data.board,data.next,options),policy:'classic',fallback:'load'});return;}
        result=chooseLearnedMove(data.board,data.next,model.score,options);
      }
      if(!result.unsupported){self.postMessage({id:data.id,...result,policy,...(fallback?{fallback}:{})});return;}
      self.postMessage({id:data.id,...chooseMove(data.board,data.next,options),policy:'classic',fallback:'range'});
    } else self.postMessage({id:data.id,...chooseMove(data.board,data.next,options),policy:'classic'});
  } catch {
    self.postMessage({ id:data.id, error:true });
  }
};
