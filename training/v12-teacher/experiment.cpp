#define SearchKey TeacherSearchKey
#define KeyHash TeacherKeyHash
#define Answer TeacherAnswer
#include "../v7/search.h"
#undef SearchKey
#undef KeyHash
#undef Answer
#include "../v8-search/mcts.h"
#include "../v7/pool.h"
#include <climits>
using namespace v2;
struct Outcome{bool win;int moves;double reward;int rank;};
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("FIRST COUNT PREFIX");init();Learner m;m.load("training/v7/base.ntd");MCTS policy(m);policy.budget=1000;policy.maxSimulations=640;Search teacher(m);teacher.maxNodes=INT_MAX;uint32_t first=std::stoul(argv[1]);int count=std::stoi(argv[2]);std::string prefix=argv[3];Pool pool;std::ofstream states(prefix+".states.jsonl"),rolls(prefix+".rollouts.jsonl");states<<std::setprecision(17);rolls<<std::setprecision(17);
 auto searchSeed=[](uint32_t seed,int step){return uint32_t(seed*2654435761u)^uint32_t(step*2246822519u);};
 auto continuation=[&](const Snapshot&s,int direction,uint32_t seed){Game g=s.game;Counts c=s.counts;RNG rng{seed};for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);int steps=0;double score=0;while(!g.over&&high(g.b)<12&&steps<6000){int d=steps?policy.choose(g.b,g.next,c,searchSeed(seed,steps)).direction:direction;auto a=move(g.b,d);if(!a.size)throw std::runtime_error("illegal continuation");score+=a.reward+g.advance(a,rng);c=observe(c,g.next);steps++;}if(!g.over&&high(g.b)<12)throw std::runtime_error("truncated continuation");return Outcome{high(g.b)>=12,steps,score,high(g.b)};};
 for(int j=0;j<count;j++){
  uint32_t seed=first+j;RNG rng{seed},sampling{seed^0x7abc42};Game g;g.reset(rng);Counts c=opening(g);Snapshot chosen;int seen=0;
  while(!g.over&&high(g.b)<12&&g.turns<6000){if(sampling.index(++seen)==0)chosen={g,c};auto a=policy.choose(g.b,g.next,c,searchSeed(seed,g.turns));g.advance(move(g.b,a.direction),rng);c=observe(c,g.next);}
  if(!g.over&&high(g.b)<12)throw std::runtime_error("truncated collection");if(!seen)throw std::runtime_error("empty game");pool.states.push_back(chosen);
  Counts actual{};for(int k=0;k<chosen.game.left;k++)actual[chosen.game.deck[k]-1]++;if(actual!=chosen.counts)throw std::runtime_error("public deck mismatch");
  auto begin=cpuNow();auto ta=teacher.choose(chosen.game.b,chosen.game.next,chosen.counts,4);double tcpu=cpuNow()-begin;if(teacher.completed!=4)throw std::runtime_error("incomplete teacher");begin=cpuNow();auto ba=policy.choose(chosen.game.b,chosen.game.next,chosen.counts,searchSeed(seed,chosen.game.turns));double bcpu=cpuNow()-begin;
  states<<"{\"seed\":"<<seed<<",\"turn\":"<<chosen.game.turns<<",\"rank\":"<<high(chosen.game.b)<<",\"sourceMoves\":"<<g.turns<<",\"sourceSuccess\":"<<(high(g.b)>=12?"true":"false")<<",\"teacherAction\":"<<ta.direction<<",\"baseAction\":"<<ba.direction<<",\"teacherCPU\":"<<tcpu<<",\"baseCPU\":"<<bcpu<<",\"teacherQGap\":"<<teacher.rootValues[ta.direction]-teacher.rootValues[ba.direction]<<"}"<<std::endl;
  if(ta.direction!=ba.direction){
   for(int r=0;r<32;r++){uint32_t rolloutSeed=11810001+uint32_t(seed-11710001)*32+r;auto b=continuation(chosen,ba.direction,rolloutSeed);auto t=continuation(chosen,ta.direction,rolloutSeed);if(r==0){auto check=continuation(chosen,ba.direction,rolloutSeed);if(check.win!=b.win||check.moves!=b.moves||check.reward!=b.reward)throw std::runtime_error("nonreproducible rollout");}
    rolls<<"{\"seed\":"<<seed<<",\"replicate\":"<<r<<",\"rolloutSeed\":"<<rolloutSeed<<",\"baseWin\":"<<(b.win?"true":"false")<<",\"teacherWin\":"<<(t.win?"true":"false")<<",\"baseMoves\":"<<b.moves<<",\"teacherMoves\":"<<t.moves<<",\"baseReward\":"<<b.reward<<",\"teacherReward\":"<<t.reward<<",\"baseRank\":"<<b.rank<<",\"teacherRank\":"<<t.rank<<"}"<<std::endl;
   }
  }
 }
 saveEarlyPool(pool,prefix+".snapshots.txt");
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
