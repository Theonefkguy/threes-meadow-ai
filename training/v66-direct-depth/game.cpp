// V66: training/v65-cascade-games/game.cpp (V43 whole games) with the iterations of the website search (6-ply /
// 0.005) started at depth START: 1 = Search::choose (V43 arm E), 6 = depth 6 only (arm D6).
//  game MODEL RES_LIST START FIRST_SEED COUNT OUT SNAP_OUT
#include "../v38-progress/cgame.h"
#include <climits>
#include <ctime>
using namespace v38;
// Search::choose to depth 6 written out, starting at depth start
static Answer search(Search&s,const CGame&g,Counts c,const Cycle&cy,int start){
 s.stamp++;s.nodes=0;s.completed=0;s.cutoffs=0;Search::Moves mv;for(int d=0;d<4;d++)mv.m[d]=move(g.b,d);Answer a;
 for(int depth=start;depth<=6;depth++){a=s.decision(g.b,mv,g.next,c,cy,depth,1.0,true);s.completed=depth;}
 return a;}
int main(int argc,char**argv){try{
 if(argc!=8)throw std::runtime_error("MODEL RES_LIST START FIRST_SEED COUNT OUT SNAP_OUT");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");int start=std::stoi(argv[3]);if(start<1||start>6)throw std::runtime_error("START 1-6");
 uint32_t first=std::stoul(argv[4]);int count=std::stoi(argv[5]);std::ofstream out(argv[6],std::ios::app),snaps(argv[7],std::ios::app);out<<std::setprecision(17);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.005;s.clampLeaf=true;s.cycleAware=true;
 for(int k=0;k<count;k++){uint32_t seed=first+k;RNG rng{seed};CGame g;g.reset(rng);
  Counts c{4,4,4};for(auto r:g.b)if(r>=1&&r<=3)c[r-1]--;c=observe(c,g.next);Cycle cy;
  std::array<int,4>at{-1,-1,-1,-1};std::array<int,3>combo{-1,-1,-1};double cpu=0,lateCpu=0;int moves=0,lateMoves=0;bool snapped=false;
  while(!g.over&&g.turns<50000){bool late=high(g.b)>=14;
   if(late&&!snapped){writeSnap(snaps,seed,g,c,cy);snapped=true;}
   auto t0=std::clock();auto a=search(s,g,c,cy,start);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;cpu+=el;if(late){lateCpu+=el;lateMoves++;}
   if(s.completed!=6)throw std::runtime_error("incomplete depth");moves++;
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);
   if(!g.over){cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);}
   for(int r=12;r<=15;r++)if(at[r-12]<0&&high(g.b)>=r)at[r-12]=g.turns;
   {int has=0;for(auto x:g.b)if(x>=11&&x<=14)has|=1<<(x-11);const int need[3]={0b1100,0b1110,0b1111};for(int i=0;i<3;i++)if(combo[i]<0&&(has&need[i])==need[i])combo[i]=g.turns;}}
  if(!g.over)throw std::runtime_error("truncated game");
  out<<"{\"seed\":"<<seed<<",\"r1536\":"<<(at[0]>=0?"true":"false")<<",\"r3072\":"<<(at[1]>=0?"true":"false")<<",\"r6144\":"<<(at[2]>=0?"true":"false")<<",\"r12288\":"<<(at[3]>=0?"true":"false")
     <<",\"turn6144\":"<<at[2]<<",\"c2\":"<<combo[0]<<",\"c3\":"<<combo[1]<<",\"c4\":"<<combo[2]<<",\"moves\":"<<moves<<",\"score\":"<<score(g.b)<<",\"bonuses\":"<<g.bonusCount
     <<",\"searchCPU\":"<<cpu<<",\"lateMoves\":"<<lateMoves<<",\"lateCPU\":"<<lateCpu<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
