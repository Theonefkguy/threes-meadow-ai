// V38 continuation from cycle-rule first-6144 states: the unseen basic deck is reshuffled with
// RNG(ROLLOUT_BASE + index); draws counter, hidden slot and public cycle continue from the state.
// Cycle-aware website search (depth 5, cutoff 0.01, clamp) to 12288 or death.
//  cont MODEL RES_LIST BIAS POOL FIRST COUNT ROLLOUT_BASE OUT
#include "cgame.h"
#include <climits>
#include <ctime>
using namespace v38;
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("cont MODEL RES_LIST BIAS POOL FIRST COUNT ROLLOUT_BASE OUT");
 v31::init();ModelP m;m.load(argv[1],argv[2],argv[3]);auto pool=readSnaps(argv[4]);int first=std::stoi(argv[5]),count=std::stoi(argv[6]);
 uint32_t base=std::stoul(argv[7]);std::ofstream out(argv[8],std::ios::app);out<<std::setprecision(17);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.01;s.clampLeaf=true;s.cycleAware=true;
 for(int idx=first;idx<first+count&&idx<int(pool.size());idx++){
  CSnap sn=pool[idx];CGame g=sn.g;Counts c=sn.c;Cycle cy=sn.cy;RNG rng{base+uint32_t(idx)};
  for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
  double start=score(g.b),cpu=0;int moves=0,maxSecond=0;
  while(!g.over&&moves<30000){auto t0=std::clock();auto a=s.choose(g.b,g.next,c,cy,5);cpu+=double(std::clock()-t0)/CLOCKS_PER_SEC;
   if(s.completed!=5)throw std::runtime_error("incomplete depth");moves++;
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);
   if(!g.over){cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);}
   int a1=0,a2=0;for(int v:g.b){if(v>a1){a2=a1;a1=v;}else if(v>a2)a2=v;}maxSecond=std::max(maxSecond,a2);}
  if(!g.over)throw std::runtime_error("truncated");bool success=high(g.b)>=FINAL_RANK;
  out<<"{\"index\":"<<idx<<",\"sourceSeed\":"<<sn.seed<<",\"success\":"<<(success?"true":"false")<<",\"moves\":"<<moves<<",\"maxSecondRank\":"<<maxSecond
     <<",\"scoreGained\":"<<score(g.b)-start<<",\"searchCPU\":"<<cpu<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
