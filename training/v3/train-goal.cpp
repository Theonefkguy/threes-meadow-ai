#include "search.h"
using namespace v2;
int main(int argc,char**argv){try{
 init();std::string out=argv[1];int episodes=std::stoi(argv[2]);uint32_t seed=std::stoul(argv[3]);Model score;score.load(argv[4]);GoalModel goal;goal.seed(score);Pool pool=loadPool(argv[5]);std::filesystem::create_directories(out);std::ofstream log(out+"/progress.jsonl");RNG rng{seed},sampling{seed^0x392bcc17};Search policy(score,&goal);uint64_t steps=0;int won=0,truncated=0;auto start=std::chrono::steady_clock::now();
 for(int ep=1;ep<=episodes;ep++){
  Snapshot s=pool.sample(sampling);Game g=s.game;Counts counts=s.counts;
  // Hidden order is used only by the environment and resampled conditional on counts.
  for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
  std::vector<ScoreRecord> t;int local=0;
  while(!g.over&&high(g.b)<14&&local<6000){
   auto a=policy.choose(g.b,g.next,counts,2);int direction=a.direction;
   if(sampling.next()<.02){std::vector<int>legalDirs;for(int d=0;d<4;d++)if(move(g.b,d).size)legalDirs.push_back(d);direction=legalDirs[sampling.index(legalDirs.size())];}
   auto m=move(g.b,direction);if(!m.size)throw std::runtime_error("illegal policy");bool hit=high(m.b)>=14;t.push_back({m.b,goal.value(m.b,g.next,counts),0,hit});g.advance(m,rng);counts=observe(counts,g.next);steps++;local++;
  }
  bool hit=high(g.b)>=14;if(!hit&&!g.over){truncated++;continue;}won+=hit;
  double f=double(ep)/episodes,alpha=f<.6?.08:f<.85?.03:.01;
  for(size_t i=0;i<t.size();i++)if(!t[i].boundary)goal.update(t[i].board,lambdaTarget(t,i),alpha);
  if(ep%5000==0||ep==episodes){log<<"{\"episode\":"<<ep<<",\"batchWins\":"<<won<<",\"steps\":"<<steps<<",\"truncated\":"<<truncated<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}"<<std::endl;std::cout<<ep<<" wins="<<won<<std::endl;won=0;goal.save(out+"/checkpoint.goal");}
 }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
