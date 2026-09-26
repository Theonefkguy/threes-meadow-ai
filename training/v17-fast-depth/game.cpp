// game DEPTH THRESHOLD FIRST_SEED COUNT OUT
// Plays COUNT normal openings (seeds FIRST_SEED..) to first 1536 or real death with
// v17 search at a fixed complete depth (no node/time cutoff). threshold=0 is the
// original complete expectimax. One JSON line per game.
#include "search.h"
#include <climits>
#include <ctime>
using namespace v2;
int main(int argc,char**argv){try{
 if(argc!=6)throw std::runtime_error("DEPTH THRESHOLD FIRST_SEED COUNT OUT");
 init();Learner m;m.load("training/v7/base.ntd");int depth=std::stoi(argv[1]);double thr=std::stod(argv[2]);uint32_t first=std::stoul(argv[3]);int count=std::stoi(argv[4]);
 std::ofstream out(argv[5],std::ios::app);out<<std::setprecision(17);v17::Search s(m);s.maxNodes=INT_MAX;s.threshold=thr;
 for(int k=0;k<count;k++){uint32_t seed=first+k;RNG rng{seed};Game g;g.reset(rng);Counts c=opening(g);std::vector<double>times;double cpu=0;uint64_t nodes=0;
  while(!g.over&&high(g.b)<12&&g.turns<6000){auto t0=std::clock();auto a=s.choose(g.b,g.next,c,depth);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;
   if(s.completed!=depth)throw std::runtime_error("incomplete depth");cpu+=el;times.push_back(el);nodes+=s.nodes;
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);c=observe(c,g.next);}
  if(!g.over&&high(g.b)<12)throw std::runtime_error("truncated game");
  std::sort(times.begin(),times.end());
  out<<"{\"seed\":"<<seed<<",\"depth\":"<<depth<<",\"threshold\":"<<thr<<",\"success\":"<<(high(g.b)>=12?"true":"false")<<",\"moves\":"<<g.turns<<",\"rank\":"<<high(g.b)
     <<",\"searchCPU\":"<<cpu<<",\"meanMoveCPU\":"<<cpu/times.size()<<",\"p95MoveCPU\":"<<times[std::min(times.size()-1,size_t(times.size()*.95))]<<",\"maxMoveCPU\":"<<times.back()<<",\"nodes\":"<<nodes<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
