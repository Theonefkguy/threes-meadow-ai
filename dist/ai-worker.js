import {loadV4Policy} from './v4-policy.js?v=32';
import { chooseMove } from './ai.js?v=32';
import { loadModel } from './rl-model.js?v=32';
import { chooseLearnedMove } from './rl-search.js?v=32';
import { chooseFastMove, webSearchOptions } from './fast-search.js?v=32';
// A failed V31 download is not retried on every move; it is retried once when the AI is started
// again (the page marks the first request of each start as fresh).
let v31Failed=false,fullFailed=false;
self.onmessage = async ({ data }) => {
  try {
    if(data.fresh)v31Failed=fullFailed=false;
    const options={remaining:data.remaining};
    if(data.policy==='rl-fast'||data.policy==='rl-full'||data.policy==='rl-v4') {
      let result,policy=data.policy,fallback;
      if(policy==='rl-full'){
        // Full precision: the same search on the float32 residuals (30.4 MB download). If it cannot
        // load, continue with the standard model below.
        try{if(fullFailed)throw new Error('full unavailable');
          const model=await loadModel('v31f',({loaded,total})=>self.postMessage({id:data.id,analysis:data.analysis,type:'progress',loaded,total}));
          const stats={};result=chooseFastMove(data.board,data.next,model,{...options,...webSearchOptions(data.board),clampLeaf:true,cycle:data.cycle,stats});
          if(result.unsupported){policy='rl-v4';result=undefined;}else result.values=stats.values;
        }catch{fullFailed=true;policy='rl-fast';fallback='full-load';}
      }
      if(policy==='rl-fast'){
        // Default (V31 release): five-move search, probability cutoff 0.01, clamped leaves,
        // 160 ms soft budget (since V43: six moves, cutoff 0.005, 400 ms on every move), on the V31 model: four-cell tables plus float16 five-cell residuals
        // for stages 1-3 (17.4 MB gzip download, 32 MB decoded, with progress reports). If it cannot load,
        // fall back to the V23 four-cell model, then to V4. Since V35 the search also uses the public
        // bonus-cycle state (data.cycle); the V4 and classic fallbacks keep the constant 1/21.
        let model;
        const progress=({loaded,total})=>self.postMessage({id:data.id,analysis:data.analysis,type:'progress',loaded,total});
        try{if(v31Failed)throw new Error('V31 unavailable');model=await loadModel('v31',progress);}
        catch{
          v31Failed=true;
          try{model=await loadModel('v23');fallback='v31-load';}catch{policy='rl-v4';fallback='fast-load';}
        }
        if(model){const stats={};result=chooseFastMove(data.board,data.next,model,{...options,...webSearchOptions(data.board),clampLeaf:true,cycle:data.cycle,stats});if(result.unsupported){policy='rl-v4';result=undefined;}else result.values=stats.values;}
      }
      if(policy==='rl-v4'){
        let model;
        try{model=await loadV4Policy();}
        catch{self.postMessage({id:data.id,analysis:data.analysis,...chooseMove(data.board,data.next,options),policy:'classic',fallback:'load'});return;}
        result=chooseLearnedMove(data.board,data.next,model.score,options);
      }
      if(!result.unsupported){self.postMessage({id:data.id,analysis:data.analysis,...result,policy,...(fallback?{fallback}:{})});return;}
      self.postMessage({id:data.id,analysis:data.analysis,...chooseMove(data.board,data.next,options),policy:'classic',fallback:'range'});
    } else self.postMessage({id:data.id,analysis:data.analysis,...chooseMove(data.board,data.next,options),policy:'classic'});
  } catch {
    self.postMessage({ id:data.id, analysis:data.analysis, error:true });
  }
};
