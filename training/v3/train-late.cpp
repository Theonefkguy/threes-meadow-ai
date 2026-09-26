#include "common.h"
using namespace v2;
int main(int argc,char**argv){try{
 init();std::string out=argv[1];int episodes=std::stoi(argv[2]);uint32_t seed=std::stoul(argv[3]);Model model;model.load(argv[4]);Pool pool=loadPool(argv[5]);std::filesystem::create_directories(out);std::ofstream log(out+"/progress.jsonl");RNG rng{seed},sample{seed^0x74b95213};auto start=std::chrono::steady_clock::now();uint64_t steps=0;int won=0,truncated=0;
 // All starts are real rank-13 trajectories. Replay is stratified by empty count
 // and whether a 1536 coexists. A small recent pool broadens on-policy coverage.
 std::array<std::vector<int>,10> buckets;for(int i=0;i<(int)pool.states.size();i++){auto&b=pool.states[i].game.b;int empty=std::count(b.begin(),b.end(),0);bool second=std::find(b.begin(),b.end(),12)!=b.end();buckets[std::min(4,empty)+5*second].push_back(i);}std::vector<int> active;for(int k=0;k<10;k++)if(!buckets[k].empty())active.push_back(k);Pool recent;
 for(int ep=1;ep<=episodes;ep++){
  Snapshot s;if(!recent.states.empty()&&sample.next()<.2)s=recent.sample(sample);else{auto&bucket=buckets[active[sample.index(active.size())]];s=pool.states[bucket[sample.index(bucket.size())]];}
  Game g=s.game;Counts counts=s.counts;std::vector<ScoreRecord>t;double last=0;int moves=0;
  while(!g.over&&moves<6000){auto a=scorePolicy(g,model);if(!a.size)throw std::runtime_error("illegal move");if(!t.empty())t.back().reward=last+a.reward;t.push_back({a.b,model.value(a.b,2),0,false});last=g.advance(a,rng);counts=observe(counts,g.next);steps++;moves++;if(moves%40==0&&high(g.b)==13)recent.add(g,counts,sample);}
  if(!g.over){truncated++;continue;}t.back().reward=last;won+=high(g.b)>=14;double fraction=double(ep)/episodes,alpha=fraction<.6?.02:fraction<.85?.008:.003;
  for(size_t i=0;i<t.size();i++)model.update(t[i].board,2,lambdaTarget(t,i),alpha);
  if(ep%10000==0||ep==episodes){log<<"{\"episode\":"<<ep<<",\"batchWins\":"<<won<<",\"steps\":"<<steps<<",\"truncated\":"<<truncated<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}"<<std::endl;std::cout<<ep<<" wins="<<won<<" steps="<<steps<<std::endl;won=0;model.save(out+"/checkpoint.ntd");}
 }
 savePool(recent,out+"/recent-pool.txt");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
