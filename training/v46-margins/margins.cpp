// V46: how decisive are the website AI's choices, and in what positions? Whole games (cycle rule) with
// the website search (6-ply / 0.005, cycle-aware, clamp, float16 weights). Every move logs the state and,
// per direction, human-readable features. One JSON line per move:
//  seed, move, maxRank, board (16 ranks), next {cards, bonus}, counts, cycle, chosen,
//  dirs[4]: {legal, value, pDeath (chance of no legal move right after the new card enters), big (cards of
//  768 or more that change cell), merges, reward (merge points), lanes (possible entry cells)}
// plus a per-game line in GAMES: seed, success, moves, score.
//  margins MODEL RES_LIST FIRST_SEED COUNT OUT GAMES
#include "../v38-progress/cgame.h"
#include <climits>
using namespace v38;
static double pDeath(const Projection&m,const Preview&p){if(high(m.b)>=FINAL_RANK)return 0;double d=0;
 for(int e=0;e<m.size;e++)for(int i=0;i<p.size;i++){Board b=m.b;b[m.entries[e]]=p.cards[i];if(!legal(b))d+=p.p[i]/m.size;}return d;}
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("margins MODEL RES_LIST FIRST_SEED COUNT OUT GAMES");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");uint32_t first=std::stoul(argv[3]);int count=std::stoi(argv[4]);
 std::ofstream out(argv[5],std::ios::app),gm(argv[6],std::ios::app);out<<std::setprecision(10);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.005;s.clampLeaf=true;s.cycleAware=true;
 for(int k=0;k<count;k++){uint32_t seed=first+k;RNG rng{seed};CGame g;g.reset(rng);
  Counts c{4,4,4};for(auto r:g.b)if(r>=1&&r<=3)c[r-1]--;c=observe(c,g.next);Cycle cy;int moves=0;
  while(!g.over&&g.turns<50000){auto a=s.choose(g.b,g.next,c,cy,6);if(s.completed!=6)throw std::runtime_error("incomplete depth");
   out<<"{\"seed\":"<<seed<<",\"move\":"<<moves<<",\"maxRank\":"<<high(g.b)<<",\"board\":[";for(int i=0;i<16;i++)out<<(i?",":"")<<int(g.b[i]);
   out<<"],\"next\":{\"cards\":[";for(int i=0;i<g.next.size;i++)out<<(i?",":"")<<int(g.next.cards[i]);out<<"],\"bonus\":"<<(g.next.bonus?"true":"false")<<"}";
   out<<",\"counts\":["<<c[0]<<","<<c[1]<<","<<c[2]<<"],\"cycle\":["<<cy.pos<<","<<cy.used<<","<<cy.excl<<"],\"chosen\":"<<a.direction<<",\"dirs\":[";
   for(int d=0;d<4;d++){auto p=move(g.b,d);out<<(d?",":"")<<"{\"legal\":"<<(p.size?1:0);
    if(p.size){int big=0,merges=0;for(int i=0;i<16;i++){if(g.b[i]>=11&&p.b[i]!=g.b[i])big++;}
     int before=0,after=0;for(int i=0;i<16;i++){before+=g.b[i]>0;after+=p.b[i]>0;}merges=before-after;
     out<<",\"value\":"<<s.rootValues[d]<<",\"pDeath\":"<<pDeath(p,g.next)<<",\"big\":"<<big<<",\"merges\":"<<merges<<",\"reward\":"<<p.reward<<",\"lanes\":"<<p.size;}
    out<<"}";}
   out<<"]}\n";
   auto p=move(g.b,a.direction);g.advance(p,rng);moves++;if(!g.over){cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);}}
  gm<<"{\"seed\":"<<seed<<",\"success\":"<<(high(g.b)>=FINAL_RANK?"true":"false")<<",\"moves\":"<<moves<<",\"score\":"<<score(g.b)<<"}"<<std::endl;out.flush();}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
