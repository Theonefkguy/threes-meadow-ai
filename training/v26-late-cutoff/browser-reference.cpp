// Untimed native directions on the 128 archived 6144-phase browser states, all V26 arms.
//  browser-reference MODEL STATES_TXT OUT   (repository root)
#include "../v21-late-train/search.h"
#include <climits>
using namespace v21;
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("MODEL STATES OUT");
 init();Model4 m;m.load(argv[1]);Search s(m);s.maxNodes=INT_MAX;s.clampLeaf=true;
 const char*names[3]={"B","A1","A2"};const int depth[3]={5,5,6};const double thr[3]={.01,.005,.005};
 std::ifstream in(argv[2]);std::ofstream out(argv[3]);Board b;int first,index=0;
 while(in>>first){b[0]=first;for(int i=1;i<16;i++){int v;in>>v;b[i]=v;}
  Preview p;int bonus;in>>p.size>>bonus;p.bonus=bonus;for(int i=0;i<3;i++){int card;in>>card>>p.p[i];p.cards[i]=card;}Counts c;for(int i=0;i<3;i++)in>>c[i];if(!in)throw std::runtime_error("bad state");
  if(high(b)<14)continue;
  for(int a=0;a<3;a++){s.threshold=thr[a];auto r=s.choose(b,p,c,depth[a]);
   out<<"{\"index\":"<<index<<",\"arm\":\""<<names[a]<<"\",\"direction\":"<<r.direction<<",\"depth\":"<<s.completed<<"}\n";}
  index++;}
 if(index!=128)throw std::runtime_error("expected 128 6144-phase states");
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
