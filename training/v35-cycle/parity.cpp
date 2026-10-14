// V35 parity: with cycleAware=false, search5c.h must equal training/v31-stages/search5s.h bit for bit
// (direction, completed depth and all root values) on the 320 V25 browser states, for any cycle state.
#include "../v31-stages/search5s.h"
#include "search5c.h"
#include <climits>
int main(int argc,char**argv){
 v31::init();v31::ModelS m;m.load(argv[1],argv[2]);
 v31::Search a(m);a.maxNodes=INT_MAX;a.threshold=0.01;a.clampLeaf=true;
 v35::Search b(m);b.maxNodes=INT_MAX;b.threshold=0.01;b.clampLeaf=true;b.cycleAware=false;
 v35::Search c(m);c.maxNodes=INT_MAX;c.threshold=0.01;c.clampLeaf=true;c.cycleAware=true;
 std::ifstream in(argv[3]);v31::Board bd;int first,n=0,bad=0,differ=0;
 while(in>>first){bd[0]=first;for(int i=1;i<16;i++){int v;in>>v;bd[i]=v;}v31::Preview p;int bonus;in>>p.size>>bonus;p.bonus=bonus;
  for(int i=0;i<3;i++){int card;in>>card>>p.p[i];p.cards[i]=card;}v31::Counts cn;for(int i=0;i<3;i++)in>>cn[i];
  v35::Cycle cy;cy.pos=n%21;cy.excl=cy.pos/2;cy.used=n%3==0;
  auto ra=a.choose(bd,p,cn,5);auto rb=b.choose(bd,p,cn,cy,5);auto rc=c.choose(bd,p,cn,cy,5);
  if(ra.direction!=rb.direction||a.completed!=b.completed||a.rootValues!=b.rootValues)bad++;
  if(rc.direction!=ra.direction)differ++;n++;}
 std::printf("states %d, unaware mismatches %d, aware direction changes %d\n",n,bad,differ);return bad?1:0;}
