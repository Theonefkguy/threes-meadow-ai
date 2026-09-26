import {loadModel} from './rl-model.js?v=20';
let pending;
export function loadV7Policy(){return pending ||= loadModel('v7').then(score=>({score})).catch(error=>{pending=null;throw error;});}
