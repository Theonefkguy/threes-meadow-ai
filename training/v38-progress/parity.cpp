// V38 parity: v38::Search with ModelP and no bias ("-") or a zero bias must equal the V35
// cycle-aware search (search5c.h) bit for bit (direction, depth, root values) on the 320 V25
// browser states with the V35 parity cycle states; also checks massKey on hand-made boards.
//  parity MODEL RES_LIST STATES_TXT
#include "../v35-cycle/search5c.h"
#include "search5p.h"
#include <climits>
int main(int argc,char**argv){
 v31::init();v31::ModelS ms;ms.load(argv[1],argv[2]);v38::ModelP a,z;a.load(argv[1],argv[2],"-");z.load(argv[1],argv[2],"zero");
 v35::Search r(ms);r.maxNodes=INT_MAX;r.threshold=0.01;r.clampLeaf=true;r.cycleAware=true;
 v38::Search sa(a);sa.maxNodes=INT_MAX;sa.threshold=0.01;sa.clampLeaf=true;sa.cycleAware=true;
 v38::Search sz(z);sz.maxNodes=INT_MAX;sz.threshold=0.01;sz.clampLeaf=true;sz.cycleAware=true;
 std::ifstream in(argv[3]);v31::Board bd;int first,n=0,bad=0;
 while(in>>first){bd[0]=first;for(int i=1;i<16;i++){int v;in>>v;bd[i]=v;}v31::Preview p;int bonus;in>>p.size>>bonus;p.bonus=bonus;
  for(int i=0;i<3;i++){int card;in>>card>>p.p[i];p.cards[i]=card;}v31::Counts cn;for(int i=0;i<3;i++)in>>cn[i];
  v35::Cycle c5;c5.pos=n%21;c5.excl=c5.pos/2;c5.used=n%3==0;v38::Cycle c8;c8.pos=c5.pos;c8.excl=c5.excl;c8.used=c5.used;
  auto x=r.choose(bd,p,cn,c5,5);auto y=sa.choose(bd,p,cn,c8,5);auto w=sz.choose(bd,p,cn,c8,5);
  if(x.direction!=y.direction||x.direction!=w.direction||r.completed!=sa.completed||r.completed!=sz.completed||r.rootValues!=sa.rootValues||r.rootValues!=sz.rootValues)bad++;n++;}
 v31::Board b{};b[0]=14;b[1]=14;b[2]=13;b[3]=12;b[4]=11;b[5]=9;b[6]=8;   // second 6144 (32) + 16 + 8 + 4 + 1 -> capped 40
 v31::Board c{};c[0]=14;c[1]=12;c[2]=10;c[3]=9;c[4]=9;c[5]=3;          // 8 + 2 + 1 + 1 = 12
 int k1=v38::massKey(b),k2=v38::massKey(c);
 std::printf("states %d, mismatches %d, massKey %d %d\n",n,bad,k1,k2);return bad||k1!=40||k2!=12?1:0;}
