#include "../v7/model.h"
#include "../v7/pool.h"
using namespace v2;
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("OUT RATIO CPU SEED BASE POOL");init();std::string out=argv[1];double ratio=std::stod(argv[2]),budget=std::stod(argv[3]);uint32_t seed=std::stoul(argv[4]);Learner m;m.load(argv[5]);Pool pool=loadEarlyPool(argv[6]);std::filesystem::create_directories(out);std::ofstream log(out+"/progress.jsonl");RNG rng{seed},sample{seed^0x4b612975};auto wall=std::chrono::steady_clock::now();auto start=std::clock();auto elapsed=[&](){return double(std::clock()-start)/CLOCKS_PER_SEC;};uint64_t steps=0,anchorUpdates=0,teacherUpdates=0;int ep=0,wins=0,resumed=0,truncated=0;double cpu=0;
 do{ep++;Game g;Counts c;if(sample.next()<ratio){auto s=pool.sample(sample);g=s.game;c=s.counts;resumed++;for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);}else{g.reset(rng);c=opening(g);}
  std::vector<ScoreRecord> t;double last=0;int local=0;bool success=false;
  while(!g.over&&local<6000){auto a=policyTwo(g,m);if(!a.size)throw std::runtime_error("illegal action");if(!t.empty())t.back().reward=last+a.reward;if(high(a.b)>=12){t.push_back({a.b,m.value(a.b,stage(a.b)),0,true});success=true;steps++;break;}t.push_back({a.b,m.value(a.b,0),0,false});last=g.advance(a,rng);c=observe(c,g.next);local++;steps++;}
  cpu=elapsed();double fraction=cpu/budget,alpha=fraction<.5?.008:fraction<.85?.003:.001;
  if(!success&&!g.over)truncated++;else{wins+=success;if(!success&&!t.empty())t.back().reward=last;for(size_t i=0;i<t.size();i++)if(!t[i].boundary)m.update(t[i].board,lambdaTarget(t,i),alpha);}
  cpu=elapsed();if(ep%2500==0||cpu>=budget){log<<"{\"episodes\":"<<ep<<",\"success1536\":"<<wins<<",\"resumed\":"<<resumed<<",\"steps\":"<<steps<<",\"cpuSeconds\":"<<cpu<<",\"wallSeconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count()<<",\"truncated\":"<<truncated<<",\"anchorUpdates\":"<<anchorUpdates<<",\"teacherUpdates\":"<<teacherUpdates<<"}"<<std::endl;m.save(out+"/checkpoint.ntd");}
 }while(cpu<budget);
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
