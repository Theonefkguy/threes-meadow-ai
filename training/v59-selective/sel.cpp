// V59 selective deepening: training/v40-late-depth/deep.cpp (TRIGGER start) where each move first searches
// to BASE_DEPTH and goes on to DEEP_DEPTH only if the root margin is small:
//   margin = (best - second best) / |best| over the legal root values of the BASE_DEPTH iteration.
// The deeper iterations run inside the same search (same transposition-table stamp), so a deepened move
// is exactly Search::choose(..., DEEP_DEPTH) and an undeepened one exactly choose(..., BASE_DEPTH).
// BASE_DEPTH = DEEP_DEPTH gives the plain search (BASE 6 / DEEP 6 must equal the V40 S arm game by game).
//  sel MODEL RES_LIST POOL FIRST COUNT ROLLOUT_BASE BASE_DEPTH DEEP_DEPTH THRESHOLD MARGIN OUT
#include "../v38-progress/cgame.h"
#include <climits>
#include <ctime>
using namespace v38;
static double margin(const std::array<double,4>&v){
 double b1=-1e300,b2=-1e300;for(double x:v){if(x<=-1e299)continue;if(x>b1){b2=b1;b1=x;}else if(x>b2)b2=x;}
 if(b2<=-1e299)return 1e300;return (b1-b2)/std::max(std::abs(b1),1e-9);}
static Answer chooseSel(Search&s,const Board&b,const Preview&p,Counts c,const Cycle&cy,int base,int deep,double m,bool&deepened){
 s.stamp++;s.nodes=0;s.completed=0;s.cutoffs=0;Answer result;deepened=false;
 Search::Moves mv;for(int d=0;d<4;d++)mv.m[d]=move(b,d);
 for(int depth=1;depth<=deep;depth++){
  if(depth>base){if(margin(s.rootValues)>=m)break;deepened=true;}
  result=s.decision(b,mv,p,c,cy,depth,1.0,true);s.completed=depth;}
 return result;}
int main(int argc,char**argv){try{
 if(argc!=12)throw std::runtime_error("sel MODEL RES_LIST POOL FIRST COUNT ROLLOUT_BASE BASE_DEPTH DEEP_DEPTH THRESHOLD MARGIN OUT");
 const int baseDepth=std::stoi(argv[7]),deepDepth=std::stoi(argv[8]);const double thr=std::stod(argv[9]),mar=std::stod(argv[10]);
 if(baseDepth<1||deepDepth<baseDepth)throw std::runtime_error("need 1 <= BASE_DEPTH <= DEEP_DEPTH");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");auto pool=readSnaps(argv[3]);int first=std::stoi(argv[4]),count=std::stoi(argv[5]);
 uint32_t seedBase=std::stoul(argv[6]);std::ofstream out(argv[11],std::ios::app);out<<std::setprecision(17);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=thr;s.clampLeaf=true;s.cycleAware=true;
 for(int idx=first;idx<first+count&&idx<int(pool.size());idx++){
  CSnap sn=pool[idx];CGame g=sn.g;Counts c=sn.c;Cycle cy=sn.cy;RNG rng{seedBase+uint32_t(idx)};
  for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
  double start=score(g.b);int moves=0,deepMoves=0;double cpu=0,deepCpu=0;
  while(!g.over&&moves<30000){bool deepened;
   auto t0=std::clock();auto a=chooseSel(s,g.b,g.next,c,cy,baseDepth,deepDepth,mar,deepened);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;
   cpu+=el;if(deepened){deepCpu+=el;deepMoves++;}moves++;
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);
   if(!g.over){cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);}}
  if(!g.over)throw std::runtime_error("truncated");bool success=high(g.b)>=FINAL_RANK;
  out<<"{\"index\":"<<idx<<",\"success\":"<<(success?"true":"false")<<",\"moves\":"<<moves<<",\"scoreGained\":"<<score(g.b)-start
     <<",\"deepMoves\":"<<deepMoves<<",\"searchCPU\":"<<cpu<<",\"deepCPU\":"<<deepCpu<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
