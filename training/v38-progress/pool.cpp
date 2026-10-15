// V38 cycle-rule first-6144 pool: whole games from standard openings with the cycle-aware website
// search (depth 5, cutoff 0.01, clamp) on MODEL (no bias), stopped at the first 6144. The full
// state is written (cgame.h line, including the hidden bonus slot) unless the game is over.
//  pool MODEL RES_LIST FIRST_SEED GAMES OUT_STATES OUT_LOG
#include "cgame.h"
#include <climits>
using namespace v38;
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("pool MODEL RES_LIST FIRST_SEED GAMES OUT_STATES OUT_LOG");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");uint32_t first=std::stoul(argv[3]);int games=std::stoi(argv[4]);
 std::ofstream states(argv[5],std::ios::app),log(argv[6],std::ios::app);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.01;s.clampLeaf=true;s.cycleAware=true;
 for(int k=0;k<games;k++){uint32_t seed=first+k;RNG rng{seed};CGame g;g.reset(rng);
  Counts c{4,4,4};for(auto r:g.b)if(r>=1&&r<=3)c[r-1]--;c=observe(c,g.next);Cycle cy;
  while(!g.over&&high(g.b)<14&&g.turns<50000){auto a=s.choose(g.b,g.next,c,cy,5);if(s.completed!=5)throw std::runtime_error("incomplete depth");
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);
   if(!g.over){cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);}}
  if(!g.over&&high(g.b)<14)throw std::runtime_error("move guard");
  bool reached=high(g.b)>=14,kept=reached&&!g.over;if(kept)writeSnap(states,seed,g,c,cy);
  log<<"{\"seed\":"<<seed<<",\"reached\":"<<(reached?"true":"false")<<",\"kept\":"<<(kept?"true":"false")<<",\"turn\":"<<g.turns<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
