import {loadModel} from './rl-model.js?v=20';
export {chooseMCTSMove} from './mcts-search.js?v=20';
let pending;
export function loadV8Policy(){return pending ||= loadModel('v8').then(score=>({score})).catch(error=>{pending=null;throw error;});}
