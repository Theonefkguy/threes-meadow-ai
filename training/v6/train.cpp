#include "model.h"
#include "pool.h"
using namespace v2;
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("OUT ARM CPU_SECONDS SEED BASE REGULAR HARD TEACHER");
 init();std::string out=argv[1],arm=argv[2];if(arm!="control"&&arm!="hard"&&arm!="probability")throw std::runtime_error("invalid arm");bool probability=arm=="probability";double budget=std::stod(argv[3]);uint32_t seed=std::stoul(argv[4]);Learner model;model.load(argv[5]);if(probability)model.initializeProbability();Pool pool=loadEarlyPool(arm=="control"?argv[6]:argv[7]);
 struct Example{Board b;int y;};std::vector<Example> teacher;if(probability){std::ifstream f(argv[8]);int y;while(f>>y){Example e;e.y=y;for(auto&r:e.b){int v;f>>v;r=v;}if(!f||high(e.b)>=12||(y!=0&&y!=1))throw std::runtime_error("invalid teacher");teacher.push_back(e);}if(teacher.empty())throw std::runtime_error("empty teacher");}
 std::filesystem::create_directories(out);std::ofstream log(out+"/progress.jsonl");RNG rng{seed},sample{seed^0x4b612975};auto wall=std::chrono::steady_clock::now();auto start=std::clock();auto elapsed=[&](){return double(std::clock()-start)/CLOCKS_PER_SEC;};uint64_t warmUpdates=0,steps=0;int ep=0,wins=0,truncated=0,resumed=0;double cpu=0,warmCPU=0;
 if(probability){while(elapsed()<budget*.2){auto&e=teacher[sample.index(teacher.size())];model.update(e.b,e.y,.1);warmUpdates++;}warmCPU=elapsed();}
 do{ep++;double f=elapsed()/budget,resume=f<.5?.8:f<.85?.5:.2;Game g;Counts c;if(sample.next()<resume){auto s=pool.sample(sample);g=s.game;c=s.counts;resumed++;for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);}else{g.reset(rng);c=opening(g);}
  std::vector<ScoreRecord> t;double last=0;int local=0;bool success=false;
  while(!g.over&&local<6000){auto a=policyTwo(g,model);if(!a.size)throw std::runtime_error("illegal training action");if(!t.empty())t.back().reward=probability?0:last+a.reward;
   if(high(a.b)>=12){t.push_back({a.b,probability?1:model.value(a.b,stage(a.b)),0,true});success=true;steps++;break;}
   t.push_back({a.b,probability?model.prob(a.b):model.value(a.b,0),0,false});last=g.advance(a,rng);c=observe(c,g.next);local++;steps++;
  }
  cpu=elapsed();if(!success&&!g.over)truncated++;else{wins+=success;if(!success&&!t.empty())t.back().reward=probability?0:last;double f=cpu/budget,alpha=probability?(f<.5?.1:f<.85?.04:.015):(f<.5?.02:f<.85?.008:.003);for(size_t i=0;i<t.size();i++)if(!t[i].boundary)model.update(t[i].board,lambdaTarget(t,i),alpha);}
  cpu=elapsed();if(ep%2500==0||cpu>=budget){log<<"{\"episodes\":"<<ep<<",\"success1536\":"<<wins<<",\"resumed\":"<<resumed<<",\"steps\":"<<steps<<",\"cpuSeconds\":"<<cpu<<",\"wallSeconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count()<<",\"truncated\":"<<truncated<<",\"warmUpdates\":"<<warmUpdates<<",\"warmCPU\":"<<warmCPU<<"}"<<std::endl;std::cout<<out<<" episodes="<<ep<<" cpu="<<cpu<<std::endl;model.save(out+"/checkpoint.ntd");}
 }while(cpu<budget);
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
