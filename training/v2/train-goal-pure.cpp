#include "goal-pure.h"
using namespace v2;
struct Record{ScoreRecord td;Preview preview;Counts counts;};
int main(int argc,char**argv){try{
 if(argc!=8)throw std::runtime_error("out episodes seed initial-score pool context warm-episodes");
 init();std::string out=argv[1];int games=std::stoi(argv[2]),warm=std::stoi(argv[7]);uint32_t seed=std::stoul(argv[3]);
 std::filesystem::create_directories(out);Model teacher;teacher.load(argv[4]);GoalModel model(std::stoi(argv[6]));model.seed(teacher);Pool pool=loadPool(argv[5]),hard;
 RNG rng{seed},sampling{seed^0x734be910};std::ofstream log(out+"/progress.jsonl");auto start=std::chrono::steady_clock::now();uint64_t steps=0;int reached[2]{},batch[2]{},truncated=0;
 for(int episode=1;episode<=games;episode++){
  bool middle=episode%2==0;Game game;Counts counts;if(middle){auto s=(!hard.states.empty()&&sampling.next()<.2?hard:pool).sample(sampling);game=s.game;counts=s.counts;}else{game.reset(rng);counts=opening(game);}
  std::vector<Record> records;std::vector<ScoreRecord> trajectory;std::deque<Snapshot> recent;int local=0;
  while(!game.over&&local<6000){
   recent.push_back({game,counts});if(recent.size()>12)recent.pop_front();auto action=episode<=warm?scorePolicy(game,teacher):goalPolicy(game,counts,model,teacher);if(!action.size)throw std::runtime_error("illegal goal action");
   bool hit=high(action.b)>=13;ScoreRecord record{action.b,model.value(action.b,game.next,counts),0,hit};records.push_back({record,game.next,counts});trajectory.push_back(record);
   int previous=high(game.b);game.advance(action,rng);counts=observe(counts,game.next);steps++;local++;if(previous<12&&high(game.b)==12)pool.add(game,counts,sampling);if(hit)break;
  }
  bool success=high(game.b)>=13;if(!success&&game.over&&recent.size()>8)hard.add(recent[recent.size()-8].game,recent[recent.size()-8].counts,sampling);
  if(!success&&!game.over){truncated++;continue;}double fraction=double(episode)/games,alpha=fraction<.6?.2:fraction<.85?.08:.03;
  for(size_t i=0;i<records.size();i++)if(!records[i].td.boundary)model.update(records[i].td.board,records[i].preview,records[i].counts,episode<=warm?double(success):lambdaTarget(trajectory,i),alpha);
  batch[middle]++;reached[middle]+=success;
  if(episode%10000==0||episode==games){double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::ostringstream s;s<<"{\"episode\":"<<episode<<",\"normal\":"<<batch[0]<<",\"normalSuccess\":"<<reached[0]<<",\"middle\":"<<batch[1]<<",\"middleSuccess\":"<<reached[1]<<",\"steps\":"<<steps<<",\"seconds\":"<<seconds<<",\"truncated\":"<<truncated<<",\"hardPool\":"<<hard.states.size()<<"}";log<<s.str()<<std::endl;std::cout<<s.str()<<std::endl;std::fill(reached,reached+2,0);std::fill(batch,batch+2,0);model.save(out+"/checkpoint.goal");if(episode%100000==0)model.save(out+"/goal-"+std::to_string(episode)+".goal");}
 }
 savePool(pool,out+"/curriculum.txt");savePool(hard,out+"/hard.txt");std::ofstream config(out+"/config.json");config<<"{\"method\":\"3072 probability, TD(lambda=.5), five-step return\",\"episodes\":"<<games<<",\"warmTeacherMonteCarlo\":"<<warm<<",\"context\":"<<(model.contextual?"true":"false")<<",\"seed\":"<<seed<<",\"steps\":"<<steps<<",\"normalEpisodes\":"<<(games+1)/2<<",\"middleEpisodes\":"<<games/2<<",\"truncated\":"<<truncated<<"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
