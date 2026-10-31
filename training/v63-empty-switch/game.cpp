// V63: training/v43-early-depth/game.cpp (6-ply / 0.005 from the first move) where a move whose board has at least
// EMPTY_K empty cells is searched with the cheap setting CHEAP_DEPTH / CHEAP_THR instead:
//  game MODEL RES_LIST CHEAP_DEPTH CHEAP_THR EMPTY_K FIRST_SEED COUNT OUT SNAP_OUT
// EMPTY_K 17 never switches, which must equal the V43 arm E records (same search, same RNG use).
// Output per game: the V43 fields plus cheapMoves, cheapCPU.
#include "../v38-progress/cgame.h"
#include <climits>
#include <ctime>
using namespace v38;
int main(int argc,char**argv){try{
 if(argc!=10)throw std::runtime_error("MODEL RES_LIST CHEAP_DEPTH CHEAP_THR EMPTY_K FIRST_SEED COUNT OUT SNAP_OUT");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");const int cheapDepth=std::stoi(argv[3]);const double cheapThr=std::stod(argv[4]);const int emptyK=std::stoi(argv[5]);
 uint32_t first=std::stoul(argv[6]);int count=std::stoi(argv[7]);std::ofstream out(argv[8],std::ios::app),snaps(argv[9],std::ios::app);out<<std::setprecision(17);
 Search s(m);s.maxNodes=INT_MAX;s.clampLeaf=true;s.cycleAware=true;
 for(int k=0;k<count;k++){uint32_t seed=first+k;RNG rng{seed};CGame g;g.reset(rng);
  Counts c{4,4,4};for(auto r:g.b)if(r>=1&&r<=3)c[r-1]--;c=observe(c,g.next);Cycle cy;
  std::array<int,4>at{-1,-1,-1,-1};std::array<int,3>combo{-1,-1,-1};double cpu=0,lateCpu=0,cheapCpu=0;int moves=0,lateMoves=0,cheapMoves=0;bool snapped=false;
  while(!g.over&&g.turns<50000){bool late=high(g.b)>=14;int empty=0;for(auto x:g.b)empty+=x==0;bool cheap=empty>=emptyK;int depth=cheap?cheapDepth:6;s.threshold=cheap?cheapThr:0.005;
   if(late&&!snapped){writeSnap(snaps,seed,g,c,cy);snapped=true;}
   auto t0=std::clock();auto a=s.choose(g.b,g.next,c,cy,depth);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;cpu+=el;if(late){lateCpu+=el;lateMoves++;}if(cheap){cheapCpu+=el;cheapMoves++;}
   if(s.completed!=depth)throw std::runtime_error("incomplete depth");moves++;
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);
   if(!g.over){cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);}
   for(int r=12;r<=15;r++)if(at[r-12]<0&&high(g.b)>=r)at[r-12]=g.turns;
   {int has=0;for(auto x:g.b)if(x>=11&&x<=14)has|=1<<(x-11);const int need[3]={0b1100,0b1110,0b1111};for(int i=0;i<3;i++)if(combo[i]<0&&(has&need[i])==need[i])combo[i]=g.turns;}}
  if(!g.over)throw std::runtime_error("truncated game");
  out<<"{\"seed\":"<<seed<<",\"r1536\":"<<(at[0]>=0?"true":"false")<<",\"r3072\":"<<(at[1]>=0?"true":"false")<<",\"r6144\":"<<(at[2]>=0?"true":"false")<<",\"r12288\":"<<(at[3]>=0?"true":"false")
     <<",\"turn6144\":"<<at[2]<<",\"c2\":"<<combo[0]<<",\"c3\":"<<combo[1]<<",\"c4\":"<<combo[2]<<",\"moves\":"<<moves<<",\"score\":"<<score(g.b)<<",\"bonuses\":"<<g.bonusCount
     <<",\"searchCPU\":"<<cpu<<",\"lateMoves\":"<<lateMoves<<",\"lateCPU\":"<<lateCpu<<",\"cheapMoves\":"<<cheapMoves<<",\"cheapCPU\":"<<cheapCpu<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
