// V21 search with a 3-stage file (stage 3 = copy of stage 2) must equal V17 search.
#include "search.h"
#include "../v17-fast-depth/search.h"
#include <climits>
using namespace v2;
int main(int argc,char**argv){init();Learner a;a.load("training/v7/base.ntd");v21::Model4 b;b.load("training/v7/base.ntd");
 v17::Search s1(a);v21::Search s2(b);s1.maxNodes=s2.maxNodes=INT_MAX;s1.threshold=s2.threshold=0.01;
 std::ifstream f(argv[1]);int n,bad=0,cnt=0;Board bd;Preview p;Counts c;int turn,x;
 while(f>>x){bd[0]=x;for(int i=1;i<16;i++){f>>x;bd[i]=x;}f>>p.size>>x;p.bonus=x;for(int i=0;i<3;i++){f>>x>>p.p[i];p.cards[i]=x;}for(auto&v:c)f>>v;f>>turn;
  auto r1=s1.choose(bd,p,c,5);auto r2=s2.choose(bd,p,c,5);cnt++;if(r1.direction!=r2.direction||s1.rootValues!=s2.rootValues)bad++;}
 std::printf("states %d mismatches %d\n",cnt,bad);return bad!=0;}
