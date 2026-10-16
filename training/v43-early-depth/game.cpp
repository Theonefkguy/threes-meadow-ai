// V43: training/v41-stage3-depth/game.cpp with the search before the first 6144 also configurable:
//  game MODEL RES_LIST EARLY_DEPTH EARLY_THR LATE_DEPTH LATE_THR FIRST_SEED COUNT OUT SNAP_OUT
// With EARLY 5 0.01 the games equal the V41 tool (checked against V41 arm D records).
// V41 note:
// V41 whole games: training/v35-cycle/game.cpp (cycle rule, cycle-aware website search) where the
// search switches to LATE_DEPTH / LATE_THR once the board holds a 6144, plus (per the project rule)
// a first-6144 checkpoint line (training/v38-progress/cgame.h format) and board combos.
//  game MODEL RES_LIST LATE_DEPTH LATE_THR FIRST_SEED COUNT OUT SNAP_OUT
// With LATE 5 0.01 the games equal training/v35-cycle/bin/game AWARE=1 (same search, same RNG use).
// Output per game: seed, r1536/r3072/r6144/r12288, turn6144, c2/c3/c4 (first move with 6144+3072,
// +1536, +768 on the board), moves, score, bonuses, searchCPU, lateMoves, lateCPU.
#include "../v38-progress/cgame.h"
#include <climits>
#include <ctime>
using namespace v38;
int main(int argc,char**argv){try{
 if(argc!=11)throw std::runtime_error("MODEL RES_LIST EARLY_DEPTH EARLY_THR LATE_DEPTH LATE_THR FIRST_SEED COUNT OUT SNAP_OUT");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");const int earlyDepth=std::stoi(argv[3]);const double earlyThr=std::stod(argv[4]);const int lateDepth=std::stoi(argv[5]);const double lateThr=std::stod(argv[6]);
 uint32_t first=std::stoul(argv[7]);int count=std::stoi(argv[8]);std::ofstream out(argv[9],std::ios::app),snaps(argv[10],std::ios::app);out<<std::setprecision(17);
 Search s(m);s.maxNodes=INT_MAX;s.clampLeaf=true;s.cycleAware=true;
 for(int k=0;k<count;k++){uint32_t seed=first+k;RNG rng{seed};CGame g;g.reset(rng);
  Counts c{4,4,4};for(auto r:g.b)if(r>=1&&r<=3)c[r-1]--;c=observe(c,g.next);Cycle cy;
  std::array<int,4>at{-1,-1,-1,-1};std::array<int,3>combo{-1,-1,-1};double cpu=0,lateCpu=0;int moves=0,lateMoves=0;bool snapped=false;
  while(!g.over&&g.turns<50000){bool late=high(g.b)>=14;int depth=late?lateDepth:earlyDepth;s.threshold=late?lateThr:earlyThr;
   if(late&&!snapped){writeSnap(snaps,seed,g,c,cy);snapped=true;}
   auto t0=std::clock();auto a=s.choose(g.b,g.next,c,cy,depth);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;cpu+=el;if(late){lateCpu+=el;lateMoves++;}
   if(s.completed!=depth)throw std::runtime_error("incomplete depth");moves++;
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);
   if(!g.over){cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);}
   for(int r=12;r<=15;r++)if(at[r-12]<0&&high(g.b)>=r)at[r-12]=g.turns;
   {int has=0;for(auto x:g.b)if(x>=11&&x<=14)has|=1<<(x-11);const int need[3]={0b1100,0b1110,0b1111};for(int i=0;i<3;i++)if(combo[i]<0&&(has&need[i])==need[i])combo[i]=g.turns;}}
  if(!g.over)throw std::runtime_error("truncated game");
  out<<"{\"seed\":"<<seed<<",\"r1536\":"<<(at[0]>=0?"true":"false")<<",\"r3072\":"<<(at[1]>=0?"true":"false")<<",\"r6144\":"<<(at[2]>=0?"true":"false")<<",\"r12288\":"<<(at[3]>=0?"true":"false")
     <<",\"turn6144\":"<<at[2]<<",\"c2\":"<<combo[0]<<",\"c3\":"<<combo[1]<<",\"c4\":"<<combo[2]<<",\"moves\":"<<moves<<",\"score\":"<<score(g.b)<<",\"bonuses\":"<<g.bonusCount
     <<",\"searchCPU\":"<<cpu<<",\"lateMoves\":"<<lateMoves<<",\"lateCPU\":"<<lateCpu<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
