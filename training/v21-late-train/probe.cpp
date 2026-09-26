// probe MODEL STATES DEPTH THRESH OUT: same output as v17 probe, four-stage model.
#include "search.h"
#include <climits>
#include <ctime>
using namespace v21;
int main(int argc,char**argv){init();Model4 m;m.load(argv[1]);std::ifstream f(argv[2]);int depth=std::stoi(argv[3]);Search s(m);s.maxNodes=INT_MAX;s.threshold=std::stod(argv[4]);s.clampLeaf=std::getenv("CLAMP")&&std::getenv("CLAMP")[0]=='1';
 std::ofstream out(argv[5]);out<<std::setprecision(17);Board b;Preview p;Counts c;int x,turn,i=0;
 while(f>>x){b[0]=x;for(int k=1;k<16;k++){f>>x;b[k]=x;}f>>p.size>>x;p.bonus=x;for(int k=0;k<3;k++){f>>x>>p.p[k];p.cards[k]=x;}for(auto&v:c)f>>v;f>>turn;
  auto t0=std::clock();auto a=s.choose(b,p,c,depth);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;
  out<<"{\"i\":"<<i++<<",\"high\":"<<high(b)<<",\"dir\":"<<a.direction<<",\"sec\":"<<el<<",\"nodes\":"<<s.nodes<<",\"q\":[";
  for(int d=0;d<4;d++)out<<(d?",":"")<<(s.rootValues[d]<-1e299?-1:s.rootValues[d]);out<<"]}\n";}}
