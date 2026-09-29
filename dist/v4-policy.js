import {loadModel} from './rl-model.js?v=21';
let pending;
export function loadV4Policy(){return pending ||= loadModel('v4').then(score=>({score})).catch(error=>{pending=null;throw error;});}
