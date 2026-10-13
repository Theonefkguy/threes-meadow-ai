// Web release parity reference: native website search (depth 5, cutoff 0.01, clamped leaves,
// complete depth) with the V31 model (four-cell tables plus per-stage five-cell residuals).
//  reference MODEL RES_LIST STATES_TXT OUT_JSONL      (RES_LIST "-" or comma-separated .res files)
// STATES_TXT: training/v25-adaptive-cutoff/browser/states.txt lines (board ranks x16, preview
// size, bonus, (card p)x3, remaining counts x3). One JSON line per state: index, direction
// (0 left, 1 right, 2 up, 3 down), completed depth, and the four root values (null = illegal).
#include "../v31-stages/search5s.h"
#include <climits>
using namespace v31;
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("MODEL RES_LIST STATES OUT");
 v31::init();ModelS m;m.load(argv[1],argv[2]);Search s(m);s.maxNodes=INT_MAX;s.threshold=0.01;s.clampLeaf=true;
 std::ifstream in(argv[3]);std::ofstream out(argv[4]);out<<std::setprecision(17);Board b;int first,index=0;
 while(in>>first){b[0]=first;for(int i=1;i<16;i++){int v;in>>v;b[i]=v;}
  Preview p;int bonus;in>>p.size>>bonus;p.bonus=bonus;for(int i=0;i<3;i++){int card;in>>card>>p.p[i];p.cards[i]=card;}Counts c;for(int i=0;i<3;i++)in>>c[i];
  if(!in)throw std::runtime_error("bad state");
  auto r=s.choose(b,p,c,5);if(s.completed!=5)throw std::runtime_error("incomplete depth");
  out<<"{\"index\":"<<index<<",\"direction\":"<<r.direction<<",\"depth\":"<<s.completed<<",\"rootValues\":[";
  for(int d=0;d<4;d++){if(d)out<<",";if(s.rootValues[d]<=-1e299)out<<"null";else out<<s.rootValues[d];}
  out<<"]}\n";index++;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
