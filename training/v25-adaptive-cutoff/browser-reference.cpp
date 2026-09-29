#include "source/training/v21-late-train/search.h"
#include <climits>
using namespace v21;
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("MODEL STATES OUT");
 init();Model4 m;m.load(argv[1]);Search s(m);s.maxNodes=4000000;s.clampLeaf=true;
 std::ifstream in(argv[2]);std::ofstream out(argv[3]);Board b;int first,index=0;
 while(in>>first){b[0]=first;for(int i=1;i<16;i++){int v;in>>v;b[i]=v;}
 Preview p;int bonus;in>>p.size>>bonus;p.bonus=bonus;for(int i=0;i<3;i++){int card;in>>card>>p.p[i];p.cards[i]=card;}Counts c;for(int i=0;i<3;i++)in>>c[i];if(!in)throw std::runtime_error("bad state");
 for(int arm=0;arm<2;arm++){s.threshold=arm&&high(b)>=13?.005:.01;auto a=s.choose(b,p,c,5);
 out<<"{\"index\":"<<index<<",\"arm\":\""<<(arm?"adaptive":"baseline")<<"\",\"direction\":"<<a.direction<<",\"depth\":"<<s.completed<<"}\n";}
 index++;
 }
 if(index!=320)throw std::runtime_error("expected 320 states");
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
