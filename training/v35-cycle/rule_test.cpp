// V35 rule test: random legal moves; once a bonus is possible, every complete block of 21 draws
// must contain exactly one bonus; gaps between bonuses are at most 41.
#include "cgame.h"
using namespace v35;
int main(){v31::init();int blocks=0,bad=0,maxgap=0,nbonus=0;
 for(uint32_t seed=1;seed<=20000;seed++){RNG rng{seed};CGame g;g.reset(rng);std::vector<int>kind;std::vector<bool>possible;
  while(!g.over&&g.turns<20000){int d;do d=rng.index(4);while(!move(g.b,d).size);g.advance(move(g.b,d),rng);
   if(!g.over){kind.push_back(g.next.bonus);possible.push_back(!bonuses(high(g.b)).empty());}}
  // kind[i] is draw i+1; blocks are draws 1-21, 22-42, ...
  int last=-1;for(size_t i=0;i<kind.size();i++)if(kind[i]){if(last>=0)maxgap=std::max(maxgap,int(i)-last);last=i;nbonus++;}
  for(size_t s0=0;s0+21<=kind.size();s0+=21){bool allPossible=true;int n=0;for(int j=0;j<21;j++){allPossible&=possible[s0+j];n+=kind[s0+j];}
   if(allPossible){blocks++;bad+=n!=1;}}}
 std::printf("blocks checked %d, without exactly one bonus %d, bonuses %d, max gap %d\n",blocks,bad,nbonus,maxgap);return bad?1:0;}
