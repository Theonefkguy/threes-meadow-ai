// V40 late deeper search: training/v39-endgame/diag.cpp where the search switches from the website
// setting (depth 5, cutoff 0.01) to DEEP_DEPTH / DEEP_THRESHOLD once TRIGGER is reached and keeps it:
//   TRIGGER start  every move from the first 6144 (the whole of stage 3)
//           m768   from the first move at which 3072+1536+768 (or more, see milestones) are on the board
//           never  website setting throughout (must equal the V38 W31 arm and V39)
//  deep MODEL RES_LIST POOL FIRST COUNT ROLLOUT_BASE TRIGGER DEEP_DEPTH DEEP_THRESHOLD OUT
// V39 note:
// V39 endgame diagnostic: training/v38-progress/cont.cpp (W31, no bias) with milestone tracking.
// Same pool, rollout seeds and search as the V38 W31 arm, so every game must equal its V38 record.
// Milestones (first move at which the board holds, besides one 6144):
//   m3072  a 3072                      m1536  3072+1536
//   m768   3072+1536+768 (final-768 chain cards)
//   ready  every card needed for 12288 by merges of cards already on the board:
//          a second 6144, or 3072+3072, or 3072+1536+1536, or 3072+1536+768+768
// chain768: at the first m768, 6144-3072-1536-768 lie on a path of orthogonally adjacent cells.
// Output per state: success, moves, and the move numbers of the milestones (-1 = never).
//  diag MODEL RES_LIST POOL FIRST COUNT ROLLOUT_BASE OUT
#include "../v38-progress/cgame.h"
#include <climits>
#include <ctime>
using namespace v38;
static bool adj(int a,int b){int ya=a/4,xa=a%4,yb=b/4,xb=b%4;return std::abs(ya-yb)+std::abs(xa-xb)==1;}
static bool chained(const Board&b){
 for(int p=0;p<16;p++)if(b[p]==14)for(int q=0;q<16;q++)if(b[q]==13&&adj(p,q))for(int r=0;r<16;r++)if(b[r]==12&&adj(q,r))for(int s=0;s<16;s++)if(b[s]==11&&adj(r,s))return true;
 return false;}
int main(int argc,char**argv){try{
 if(argc!=11)throw std::runtime_error("deep MODEL RES_LIST POOL FIRST COUNT ROLLOUT_BASE TRIGGER DEEP_DEPTH DEEP_THRESHOLD OUT");
 const std::string trigger=argv[7];if(trigger!="start"&&trigger!="m768"&&trigger!="never")throw std::runtime_error("bad trigger");
 const int deepDepth=std::stoi(argv[8]);const double deepThr=std::stod(argv[9]);
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");auto pool=readSnaps(argv[3]);int first=std::stoi(argv[4]),count=std::stoi(argv[5]);
 uint32_t base=std::stoul(argv[6]);std::ofstream out(argv[10],std::ios::app);out<<std::setprecision(17);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.01;s.clampLeaf=true;s.cycleAware=true;
 for(int idx=first;idx<first+count&&idx<int(pool.size());idx++){
  CSnap sn=pool[idx];CGame g=sn.g;Counts c=sn.c;Cycle cy=sn.cy;RNG rng{base+uint32_t(idx)};
  for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
  double start=score(g.b);int moves=0,deepMoves=0;double cpu=0,deepCpu=0;int ms[4]={-1,-1,-1,-1};int chain=-1;
  auto mark=[&]{int n[16]={};bool skipped=false;for(int v:g.b){if(v==14&&!skipped){skipped=true;continue;}if(v<16)n[v]++;}
   bool k[4]={n[13]>=1,n[13]>=1&&n[12]>=1,n[13]>=1&&n[12]>=1&&n[11]>=1,
    n[14]>=1||n[13]>=2||(n[13]>=1&&n[12]>=2)||(n[13]>=1&&n[12]>=1&&n[11]>=2)};
   for(int i=0;i<4;i++)if(ms[i]<0&&(k[i]||(i<3&&n[14]>=1)))ms[i]=moves;
   if(ms[2]==moves&&chain<0)chain=chained(g.b)?1:0;};
  mark();
  while(!g.over&&moves<30000){bool deep=trigger=="start"||(trigger=="m768"&&ms[2]>=0);int depth=deep?deepDepth:5;s.threshold=deep?deepThr:0.01;
   auto t0=std::clock();auto a=s.choose(g.b,g.next,c,cy,depth);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;cpu+=el;if(deep){deepCpu+=el;deepMoves++;}
   if(s.completed!=depth)throw std::runtime_error("incomplete depth");moves++;
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);
   if(!g.over){cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);}
   if(high(g.b)<FINAL_RANK)mark();}
  if(!g.over)throw std::runtime_error("truncated");bool success=high(g.b)>=FINAL_RANK;
  out<<"{\"index\":"<<idx<<",\"success\":"<<(success?"true":"false")<<",\"moves\":"<<moves<<",\"scoreGained\":"<<score(g.b)-start
     <<",\"m3072\":"<<ms[0]<<",\"m1536\":"<<ms[1]<<",\"m768\":"<<ms[2]<<",\"ready\":"<<ms[3]<<",\"chain768\":"<<chain<<",\"deepMoves\":"<<deepMoves<<",\"searchCPU\":"<<cpu<<",\"deepCPU\":"<<deepCpu<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
