// V35 whole games under the bonus-cycle rule measured in the official app.
//   game MODEL RES_LIST DEPTH THRESHOLD CLAMP(0/1) AWARE(0/1) FIRST_SEED COUNT OUT
// Rule: the opening preview is draw 0; draws 1, 2, ... form blocks of 21 (draws 1-21, 22-42, ...).
// At the start of each block one position is drawn uniformly (hidden); at that draw the preview is
// a bonus if one is possible (max card >= 48, window rule as before), otherwise the slot is lost.
// All other draws come from the 12-card basic deck as before. AWARE selects the search's chance
// model (search5c.h); the AI's cycle state is built only from previews it has seen.
#include "cgame.h"
#include <climits>
#include <ctime>
using namespace v35;
int main(int argc,char**argv){try{
 if(argc!=10)throw std::runtime_error("MODEL RES_LIST DEPTH THRESHOLD CLAMP AWARE FIRST_SEED COUNT OUT");
 v31::init();ModelS m;m.load(argv[1],argv[2]);int depth=std::stoi(argv[3]);double thr=std::stod(argv[4]);bool clamp=std::stoi(argv[5]);bool aware=std::stoi(argv[6]);
 uint32_t first=std::stoul(argv[7]);int count=std::stoi(argv[8]);std::ofstream out(argv[9],std::ios::app);out<<std::setprecision(17);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=thr;s.clampLeaf=clamp;s.cycleAware=aware;
 for(int k=0;k<count;k++){uint32_t seed=first+k;RNG rng{seed};CGame g;g.reset(rng);
  Counts c{4,4,4};for(auto r:g.b)if(r>=1&&r<=3)c[r-1]--;c=observe(c,g.next);   // opening(): the opening preview is draw 0
  Cycle cy;   // cycle state of the next draw, from public previews only
  std::array<int,4>at{-1,-1,-1,-1};double cpu=0;int moves=0;
  while(!g.over&&g.turns<50000){auto t0=std::clock();auto a=s.choose(g.b,g.next,c,cy,depth);cpu+=double(std::clock()-t0)/CLOCKS_PER_SEC;
   if(s.completed!=depth)throw std::runtime_error("incomplete depth");moves++;
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");
   g.advance(p,rng);
   if(!g.over){bool possible=!bonuses(high(g.b)).empty();cy=cy.after(g.next.bonus,possible);c=observe(c,g.next);}
   for(int r=12;r<=15;r++)if(at[r-12]<0&&high(g.b)>=r)at[r-12]=g.turns;}
  if(!g.over)throw std::runtime_error("truncated game");
  out<<"{\"seed\":"<<seed<<",\"aware\":"<<aware<<",\"r1536\":"<<(at[0]>=0?"true":"false")<<",\"r3072\":"<<(at[1]>=0?"true":"false")<<",\"r6144\":"<<(at[2]>=0?"true":"false")<<",\"r12288\":"<<(at[3]>=0?"true":"false")
     <<",\"turn6144\":"<<at[2]<<",\"moves\":"<<moves<<",\"finalRank\":"<<high(g.b)<<",\"score\":"<<score(g.b)<<",\"bonuses\":"<<g.bonusCount<<",\"searchCPU\":"<<cpu<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
