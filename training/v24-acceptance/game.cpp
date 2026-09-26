// V24 whole-game acceptance: normal opening -> 12288 (final card, game ends) or death.
//   game MODEL DEPTH THRESHOLD MAX_NODES CLAMP(0/1) FIRST_SEED COUNT OUT
// MAX_NODES=0 means complete depth every move (no node cap); otherwise iterative deepening
// with that node budget keeps the last completed depth (as the website's 3-ply/24000-node
// search). Official-modern rules (default engine). One JSON line per game.
#include "../v21-late-train/search.h"
#include <climits>
#include <ctime>
using namespace v21;
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("MODEL DEPTH THRESHOLD MAX_NODES CLAMP FIRST_SEED COUNT OUT");
 init();Model4 m;m.load(argv[1]);int depth=std::stoi(argv[2]);double thr=std::stod(argv[3]);int maxNodes=std::stoi(argv[4]);bool clamp=std::stoi(argv[5]);
 uint32_t first=std::stoul(argv[6]);int count=std::stoi(argv[7]);std::ofstream out(argv[8],std::ios::app);out<<std::setprecision(17);
 Search s(m);s.maxNodes=maxNodes>0?maxNodes:INT_MAX;s.threshold=thr;s.clampLeaf=clamp;
 for(int k=0;k<count;k++){uint32_t seed=first+k;RNG rng{seed};Game g;g.reset(rng);Counts c=opening(g);
  std::array<int,4>at{-1,-1,-1,-1};double cpu=0,mx=0;int moves=0;std::array<int,6>depthHist{};
  while(!g.over&&g.turns<50000){auto t0=std::clock();auto a=s.choose(g.b,g.next,c,depth);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;
   if(maxNodes<=0&&s.completed!=depth)throw std::runtime_error("incomplete depth");depthHist[std::min(5,s.completed)]++;
   cpu+=el;mx=std::max(mx,el);moves++;auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");
   g.advance(p,rng);if(!g.over)c=observe(c,g.next);for(int r=12;r<=15;r++)if(at[r-12]<0&&high(g.b)>=r)at[r-12]=g.turns;}
  if(!g.over)throw std::runtime_error("truncated game");
  out<<"{\"seed\":"<<seed<<",\"r1536\":"<<(at[0]>=0?"true":"false")<<",\"r3072\":"<<(at[1]>=0?"true":"false")<<",\"r6144\":"<<(at[2]>=0?"true":"false")<<",\"r12288\":"<<(at[3]>=0?"true":"false")
     <<",\"turn6144\":"<<at[2]<<",\"turn12288\":"<<at[3]<<",\"moves\":"<<moves<<",\"finalRank\":"<<high(g.b)<<",\"score\":"<<score(g.b)<<",\"searchCPU\":"<<cpu<<",\"maxMoveCPU\":"<<mx
     <<",\"depthHist\":["<<depthHist[0]<<","<<depthHist[1]<<","<<depthHist[2]<<","<<depthHist[3]<<","<<depthHist[4]<<","<<depthHist[5]<<"]}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
