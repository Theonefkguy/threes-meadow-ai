import {loadModel} from './rl-model.js?v=20';
export async function loadV3Policy(){return {score:await loadModel('v3')};}
