#include "search.h"
using namespace v2;
int main(int argc,char**argv){try{
 init();std::string out=argv[1],mode=argv[2];double budget=std::stod(argv[3]);uint32_t seed=std::stoul(argv[4]);Learner model;model.load(argv[5]);if(mode=="b")model.enableFive();if(mode=="c")model.enableTC();Pool pool=loadPool(argv[6]),recent;std::filesystem::create_directories(out);std::ofstream log(out+"/progress.jsonl");RNG rng{seed},sample{seed^0x4b612975};Search search(model);
 std::array<std::vector<int>,10>buckets;for(int i=0;i<int(pool.states.size());i++){auto&b=pool.states[i].game.b;int empty=std::count(b.begin(),b.end(),0);bool second=std::find(b.begin(),b.end(),12)!=b.end();buckets[std::min(4,empty)+5*second].push_back(i);}std::vector<int>active;for(int k=0;k<10;k++)if(!buckets[k].empty())active.push_back(k);
 auto wall=std::chrono::steady_clock::now();auto start=std::clock();uint64_t steps=0,deepSteps=0;int ep=0,wins=0,truncated=0,totalWins=0,deepEpisodes=0;double cpu=0;
 do{ep++;Snapshot s;if(!recent.states.empty()&&sample.next()<.2)s=recent.sample(sample);else{auto&bucket=buckets[active[sample.index(active.size())]];s=pool.states[bucket[sample.index(bucket.size())]];}Game g=s.game;Counts c=s.counts;for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
  bool deep=mode!="control"&&ep%20==0;deepEpisodes+=deep;std::vector<ScoreRecord>t;double last=0;int local=0;
  while(!g.over&&local<6000){Projection a;if(deep){auto answer=search.choose(g.b,g.next,c);a=move(g.b,answer.direction);deepSteps++;}else a=policyTwo(g,model);if(!a.size)throw std::runtime_error("illegal action");if(!t.empty())t.back().reward=last+a.reward;t.push_back({a.b,model.value(a.b,2),0,false});last=g.advance(a,rng);c=observe(c,g.next);local++;steps++;if(local%40==0&&high(g.b)==13)recent.add(g,c,sample);}
  cpu=double(std::clock()-start)/CLOCKS_PER_SEC;if(!g.over)truncated++;else{t.back().reward=last;int hit=high(g.b)>=14;wins+=hit;totalWins+=hit;double f=cpu/budget,alpha=f<.6?.02:f<.85?.008:.003;if(mode=="c")alpha*=2.5;for(size_t i=0;i<t.size();i++)model.update(t[i].board,lambdaTarget(t,i),alpha);}
  cpu=double(std::clock()-start)/CLOCKS_PER_SEC;
  if(ep%2500==0||cpu>=budget){log<<"{\"episodes\":"<<ep<<",\"batchWins\":"<<wins<<",\"totalWins\":"<<totalWins<<",\"steps\":"<<steps<<",\"deepSteps\":"<<deepSteps<<",\"deepEpisodes\":"<<deepEpisodes<<",\"cpuSeconds\":"<<cpu<<",\"wallSeconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count()<<",\"truncated\":"<<truncated<<"}"<<std::endl;std::cout<<mode<<" "<<ep<<" cpu="<<cpu<<std::endl;wins=0;model.save(out+"/checkpoint");}
 }while(cpu<budget);
 savePool(recent,out+"/recent-pool.txt");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
