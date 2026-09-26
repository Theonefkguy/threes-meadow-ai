#include "common.h"
using namespace v2;
int main(int argc,char**argv){try{
 init();std::string out=argv[1];int games=std::stoi(argv[2]);uint32_t seed=std::stoul(argv[3]);
 std::filesystem::create_directories(out);Model model;model.load(argv[4]);Pool pool=loadPool(argv[5]),hard;
 RNG rng{seed},sampling{seed^0x734be910};std::ofstream log(out+"/progress.jsonl");
 auto start=std::chrono::steady_clock::now();uint64_t steps=0;int reached[2]={0,0},batch[2]={0,0},truncated=0;
 for(int episode=1;episode<=games;episode++){
  bool middle=episode%2==0;Game game;Counts counts;
  if(middle){auto s=(!hard.states.empty()&&sampling.next()<.2?hard:pool).sample(sampling);game=s.game;counts=s.counts;}
  else{game.reset(rng);counts=opening(game);}
  std::vector<ScoreRecord> trajectory;std::deque<Snapshot> recent;double lastSpawn=0;int local=0;
  while(!game.over&&local<6000){
   recent.push_back({game,counts});if(recent.size()>12)recent.pop_front();
   auto action=scorePolicy(game,model);if(!action.size)throw std::runtime_error("illegal policy move");
   if(!trajectory.empty())trajectory.back().reward=lastSpawn+action.reward;
   bool hit=high(action.b)>=13;
   trajectory.push_back({action.b,model.value(action.b,stage(action.b)),0,hit});
   int previous=high(game.b);lastSpawn=game.advance(action,rng);counts=observe(counts,game.next);steps++;local++;
   if(previous<12&&high(game.b)==12)pool.add(game,counts,sampling);
   if(hit)break;
  }
  bool success=high(game.b)>=13;
  if(!success&&game.over){trajectory.back().reward=lastSpawn;if(recent.size()>8)hard.add(recent[recent.size()-8].game,recent[recent.size()-8].counts,sampling);}
  if(!success&&!game.over){truncated++;continue;}
  double fraction=double(episode)/games,alpha=fraction<.6?.02:fraction<.85?.008:.003;
  // All bootstrap values are from the pre-update trajectory, not partially
  // updated weights. Stage-2 score weights stay frozen at the V1 checkpoint.
  for(size_t i=0;i<trajectory.size();i++)if(!trajectory[i].boundary)model.update(trajectory[i].board,stage(trajectory[i].board),lambdaTarget(trajectory,i),alpha);
  batch[middle]++;reached[middle]+=success;
  if(episode%10000==0||episode==games){
   double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
   std::ostringstream s;s<<"{\"episode\":"<<episode<<",\"normal\":"<<batch[0]<<",\"normalSuccess\":"<<reached[0]<<",\"middle\":"<<batch[1]<<",\"middleSuccess\":"<<reached[1]<<",\"steps\":"<<steps<<",\"seconds\":"<<seconds<<",\"truncated\":"<<truncated<<",\"hardPool\":"<<hard.states.size()<<"}";
   log<<s.str()<<std::endl;std::cout<<s.str()<<std::endl;std::fill(reached,reached+2,0);std::fill(batch,batch+2,0);
   model.save(out+"/checkpoint.ntd");if(episode%100000==0)model.save(out+"/score-"+std::to_string(episode)+".ntd");
  }
 }
 savePool(pool,out+"/curriculum.txt");savePool(hard,out+"/hard.txt");
 std::ofstream config(out+"/config.json");config<<"{\"method\":\"TD(lambda=.5), truncated five-step return, score objective\",\"episodes\":"<<games<<",\"seed\":"<<seed<<",\"steps\":"<<steps<<",\"normalEpisodes\":"<<(games+1)/2<<",\"middleEpisodes\":"<<games/2<<",\"truncated\":"<<truncated<<"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
