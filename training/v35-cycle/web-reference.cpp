// V35 web parity reference: training/web-release/reference.cpp with the cycle-aware search
// (search5c.h, cycleAware=true). State i gets the cycle state pos=i%21, excl=pos/2, used=(i%3==0),
// the same as parity.cpp and training/web-release/parity.mjs with CYCLE=1.
//  web-reference MODEL RES_LIST STATES_TXT OUT_JSONL
#include "search5c.h"
#include <climits>
using namespace v35;
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("MODEL RES_LIST STATES OUT");
 v31::init();ModelS m;m.load(argv[1],argv[2]);Search s(m);s.maxNodes=INT_MAX;s.threshold=0.01;s.clampLeaf=true;s.cycleAware=true;
 std::ifstream in(argv[3]);std::ofstream out(argv[4]);out<<std::setprecision(17);Board b;int first,index=0;
 while(in>>first){b[0]=first;for(int i=1;i<16;i++){int v;in>>v;b[i]=v;}
  Preview p;int bonus;in>>p.size>>bonus;p.bonus=bonus;for(int i=0;i<3;i++){int card;in>>card>>p.p[i];p.cards[i]=card;}Counts c;for(int i=0;i<3;i++)in>>c[i];
  if(!in)throw std::runtime_error("bad state");
  Cycle cy;cy.pos=index%21;cy.excl=cy.pos/2;cy.used=index%3==0;
  auto r=s.choose(b,p,c,cy,5);if(s.completed!=5)throw std::runtime_error("incomplete depth");
  out<<"{\"index\":"<<index<<",\"direction\":"<<r.direction<<",\"depth\":"<<s.completed<<",\"rootValues\":[";
  for(int d=0;d<4;d++){if(d)out<<",";if(s.rootValues[d]<=-1e299)out<<"null";else out<<s.rootValues[d];}
  out<<"]}\n";index++;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
