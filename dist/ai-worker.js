import {loadV4Policy} from './v4-policy.js?v=23';
import { chooseMove } from './ai.js?v=23';
import { loadModel } from './rl-model.js?v=23';
import { chooseLearnedMove } from './rl-search.js?v=23';
import { chooseFastMove } from './fast-search.js?v=23';
// A failed V31 download is not retried on every move of this worker (each AI start makes a new worker).
let v31Failed=false;
self.onmessage = async ({ data }) => {
  try {
    const options={remaining:data.remaining};
    if(data.policy==='rl-fast'||data.policy==='rl-v4') {
      let result,policy=data.policy,fallback;
      if(policy==='rl-fast'){
        // Default (V31 release): five-move search, probability cutoff 0.01, clamped leaves,
        // 160 ms soft budget, on the V31 model: four-cell tables plus float16 five-cell residuals
        // for stages 1-3 (17.4 MB gzip download, 32 MB decoded, with progress reports). If it cannot load,
        // fall back to the V23 four-cell model, then to V4.
        let model;
        const progress=({loaded,total})=>self.postMessage({id:data.id,type:'progress',loaded,total});
        try{if(v31Failed)throw new Error('V31 unavailable');model=await loadModel('v31',progress);}
        catch{
          v31Failed=true;
          try{model=await loadModel('v23');fallback='v31-load';}catch{policy='rl-v4';fallback='fast-load';}
        }
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
