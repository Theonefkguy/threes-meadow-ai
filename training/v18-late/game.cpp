// game DEPTH THRESHOLD FIRST_SEED COUNT OUT
// Normal opening -> first 6144 or real death, v17 search at a fixed complete depth
// (no node/time cutoff). Records the first turn each milestone is reached and CPU per
// phase. threshold=0 is the original complete expectimax. One JSON line per game.
#include "../v17-fast-depth/search.h"
#include <climits>
#include <ctime>
using namespace v2;
int main(int argc,char**argv){try{
 if(argc!=6)throw std::runtime_error("DEPTH THRESHOLD FIRST_SEED COUNT OUT");
 init();Learner m;m.load("training/v7/base.ntd");int depth=std::stoi(argv[1]);double thr=std::stod(argv[2]);uint32_t first=std::stoul(argv[3]);int count=std::stoi(argv[4]);
 std::ofstream out(argv[5],std::ios::app);out<<std::setprecision(17);v17::Search s(m);s.maxNodes=INT_MAX;s.threshold=thr;
 for(int k=0;k<count;k++){uint32_t seed=first+k;RNG rng{seed};Game g;g.reset(rng);Counts c=opening(g);
  std::array<int,3>at{-1,-1,-1};std::array<double,3>cpu{0,0,0};double mx=0;uint64_t nodes=0;int moves=0;double score=0;
  while(!g.over&&high(g.b)<14&&g.turns<20000){
   int phase=stage(g.b);auto t0=std::clock();auto a=s.choose(g.b,g.next,c,depth);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;
   if(s.completed!=depth)throw std::runtime_error("incomplete depth");cpu[phase]+=el;mx=std::max(mx,el);nodes+=s.nodes;moves++;
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);c=observe(c,g.next);
   for(int r=12;r<=14;r++)if(at[r-12]<0&&high(g.b)>=r)at[r-12]=g.turns;}
  if(!g.over&&high(g.b)<14)throw std::runtime_error("truncated game");
  score=v2::score(g.b);
  out<<"{\"seed\":"<<seed<<",\"depth\":"<<depth<<",\"threshold\":"<<thr<<",\"r1536\":"<<(at[0]>=0?"true":"false")<<",\"r3072\":"<<(at[1]>=0?"true":"false")<<",\"r6144\":"<<(at[2]>=0?"true":"false")
     <<",\"turn1536\":"<<at[0]<<",\"turn3072\":"<<at[1]<<",\"turn6144\":"<<at[2]<<",\"moves\":"<<moves<<",\"rank\":"<<high(g.b)<<",\"finalScore\":"<<score
     <<",\"cpuEarly\":"<<cpu[0]<<",\"cpuMid\":"<<cpu[1]<<",\"cpuLate\":"<<cpu[2]<<",\"maxMoveCPU\":"<<mx<<",\"nodes\":"<<nodes<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
