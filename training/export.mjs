import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { decodeModel } from '../dist/rl-model.js';
const run=process.argv[2]||'training/run-v1-refined';
const input=path.join(run,'checkpoint.ntd');
const bytes=fs.readFileSync(input);
decodeModel(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
const config=JSON.parse(fs.readFileSync(path.join(run,'config.json')));
const pretraining=JSON.parse(fs.readFileSync('training/pretraining.json'));
fs.mkdirSync('dist/models',{recursive:true});
fs.copyFileSync(input,'dist/models/ntuple-v1.bin');
const metadata={name:'合三 MS-TD v1',format:'NTD1 / Float32 LE',bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex'),
  features:'8 four-cell tuples × 8 symmetries × 3 stage tables',thresholds:[1536,3072],
  reward:'merge score increment plus subsequent spawn score; terminal continuation = 0',
  learnedInput:'afterstate board only; expectimax separately uses public preview and inferred remaining composition',
  source:'independently trained in this project; no third-party weights',pretraining,training:config,
  totalEpisodesUsed:pretraining.episodes+config.episodes.reduce((a,b)=>a+b,0),totalStepsUsed:pretraining.steps+config.totalSteps};
fs.writeFileSync('dist/models/ntuple-v1.json',JSON.stringify(metadata,null,2)+'\n');
console.log(JSON.stringify(metadata));
