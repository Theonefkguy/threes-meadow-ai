// V26 first-6144 pool: fresh standard openings played by the website default AI
// (V23 model, five-ply cutoff 0.01, leaf clamp 0) until the first 6144 appears.
//  pool MODEL FIRST_SEED GAMES OUT_STATES OUT_LOG
// OUT_STATES uses the training/v21-late-train/snap.cpp pool line format, so its
// `cont` mode can continue these states. OUT_LOG has one JSON line per opening.
#include "../v21-late-train/search.h"
#include <climits>
using namespace v21;
static void write(std::ostream&o,uint32_t seed,const Game&g,Counts c){
 o<<std::setprecision(17)<<seed<<' '<<g.turns<<' '<<g.left;for(int i=0;i<g.left;i++)o<<' '<<int(g.deck[i]);
 o<<' '<<g.next.size<<' '<<int(g.next.bonus);for(int i=0;i<3;i++)o<<' '<<int(g.next.cards[i])<<' '<<g.next.p[i];
 for(auto v:c)o<<' '<<v;for(auto r:g.b)o<<' '<<int(r);o<<'\n';o.flush();}
int main(int argc,char**argv){try{
 if(argc!=7||std::string(argv[1])!="pool")throw std::runtime_error("pool MODEL FIRST_SEED GAMES OUT_STATES OUT_LOG");
 init();Model4 m;m.load(argv[2]);uint32_t first=std::stoul(argv[3]);int games=std::stoi(argv[4]);
 std::ofstream states(argv[5],std::ios::app),log(argv[6],std::ios::app);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.01;s.clampLeaf=true;
 for(int k=0;k<games;k++){uint32_t seed=first+k;RNG rng{seed};Game g;g.reset(rng);Counts c=opening(g);
  while(!g.over&&high(g.b)<14&&g.turns<50000){auto a=s.choose(g.b,g.next,c,5);if(s.completed!=5)throw std::runtime_error("incomplete depth");
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);if(g.over)break;c=observe(c,g.next);}
  if(!g.over&&high(g.b)<14)throw std::runtime_error("move guard");
  bool reached=high(g.b)>=14,kept=reached&&!g.over;if(kept)write(states,seed,g,c);
  log<<"{\"seed\":"<<seed<<",\"reached6144\":"<<(reached?"true":"false")<<",\"kept\":"<<(kept?"true":"false")<<",\"turn\":"<<g.turns<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
