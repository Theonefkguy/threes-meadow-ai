// V45 scan: continue first-6144 states with the website search (6-ply / 0.005, cycle-aware, clamp) and
// log moves that rearrange big cards. For every move and direction: legal, root value and how many
// big cards (768 or larger) change cell content. A move is "rearranging" if the chosen direction moves
// two or more big cards while a legal direction moves none; "cornerLeave" if the largest card leaves a
// corner while a legal direction keeps it there. Logged: every rearranging or cornerLeave move, plus
// every 50th move as a baseline sample, each with the full state (cgame.h line) for later rollouts.
//  scan MODEL RES_LIST POOL FIRST COUNT ROLLOUT_BASE OUT_MOVES OUT_GAMES
#include "../v38-progress/cgame.h"
#include <climits>
using namespace v38;
static bool corner(int c){return c==0||c==3||c==12||c==15;}
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("scan MODEL RES_LIST POOL FIRST COUNT ROLLOUT_BASE OUT_MOVES OUT_GAMES");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");auto pool=readSnaps(argv[3]);int first=std::stoi(argv[4]),count=std::stoi(argv[5]);
 uint32_t base=std::stoul(argv[6]);std::ofstream mv(argv[7],std::ios::app),gm(argv[8],std::ios::app);mv<<std::setprecision(17);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.005;s.clampLeaf=true;s.cycleAware=true;
 for(int idx=first;idx<first+count&&idx<int(pool.size());idx++){
  CSnap sn=pool[idx];CGame g=sn.g;Counts c=sn.c;Cycle cy=sn.cy;RNG rng{base+uint32_t(idx)};
  for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
  int moves=0,logged=0,rearr=0,leave=0;double start=score(g.b);
  while(!g.over&&moves<30000){
   std::ostringstream st;writeSnap(st,sn.seed,g,c,cy);std::string line=st.str();line.pop_back();
   auto a=s.choose(g.b,g.next,c,cy,6);if(s.completed!=6)throw std::runtime_error("incomplete depth");
   int big[4],legal[4],keep[4];int hi=high(g.b),hc=-1;for(int i=0;i<16;i++)if(g.b[i]==hi&&corner(i))hc=i;
   for(int d=0;d<4;d++){auto p=move(g.b,d);legal[d]=p.size>0;big[d]=0;keep[d]=0;
    if(legal[d]){for(int i=0;i<16;i++)if(g.b[i]>=11&&p.b[i]!=g.b[i])big[d]++;
     for(int i=0;i<16;i++)if(corner(i)&&p.b[i]==hi)keep[d]=1;}}
   int ch=a.direction;bool calmExists=false,keepExists=false;for(int d=0;d<4;d++)if(legal[d]){if(big[d]==0)calmExists=true;if(keep[d])keepExists=true;}
   bool re=big[ch]>=2&&calmExists,cl=hc>=0&&!keep[ch]&&keepExists,sample=moves%50==0;
   if(re)rearr++;if(cl)leave++;
   if(re||cl||sample){logged++;
    mv<<"{\"index\":"<<idx<<",\"move\":"<<moves<<",\"rearranging\":"<<(re?"true":"false")<<",\"cornerLeave\":"<<(cl?"true":"false")<<",\"sample\":"<<(sample?"true":"false")
      <<",\"chosen\":"<<ch<<",\"values\":[";for(int d=0;d<4;d++)mv<<(d?",":"")<<(legal[d]?s.rootValues[d]:0);mv<<"],\"legal\":[";
    for(int d=0;d<4;d++)mv<<(d?",":"")<<legal[d];mv<<"],\"bigMoved\":[";for(int d=0;d<4;d++)mv<<(d?",":"")<<big[d];mv<<"],\"keepsCorner\":[";
    for(int d=0;d<4;d++)mv<<(d?",":"")<<keep[d];mv<<"],\"state\":\""<<line<<"\"}\n";}
   moves++;auto p=move(g.b,ch);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);
   if(!g.over){cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);}}
  if(!g.over)throw std::runtime_error("truncated");
  gm<<"{\"index\":"<<idx<<",\"seed\":"<<sn.seed<<",\"success\":"<<(high(g.b)>=FINAL_RANK?"true":"false")<<",\"moves\":"<<moves<<",\"scoreGained\":"<<score(g.b)-start
    <<",\"rearranging\":"<<rearr<<",\"cornerLeave\":"<<leave<<",\"logged\":"<<logged<<"}"<<std::endl;mv.flush();}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
