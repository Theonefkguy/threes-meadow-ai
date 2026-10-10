// V27 stage-3 (>=6144) training: the V23 stage-2 method applied to stage 3, with a
// hardware-independent budget.
//   train MODEL_IN MODEL_OUT MOVES SEED LOG STRONG_MOVE_SHARE POOL [POOL...]
// Derived from training/v23-stage2/train.cpp. Differences:
//  * only stage 3 is updated; episodes run to 12288 (terminal, value 0) or death, so there
//    is no bootstrap boundary;
//  * learning rates are the V21 stage-3 rates (.02/.008/.003);
//  * the budget is MOVES environment moves (not CPU seconds) and the rate steps down at 50%
//    and 85% of it;
//  * an episode is played by the strong player (complete 3-ply expectimax, clamped leaves)
//    when strong moves so far are below STRONG_MOVE_SHARE of all moves so far; otherwise by
//    the two-move policy. V23 defined the share on CPU time, which depends on the machine.
// With the same compiler and flags the run is therefore deterministic given SEED.
// TD(lambda=.5) five-step truncated score return and the multiplicity-normalised update are
// unchanged from V21/V23. Hidden deck order is never used by either player.
#include "../v21-late-train/search.h"
#include <ctime>
#include <climits>
using namespace v21;
struct Rec{Board b;double value,reward;};
static double pts(int r){return points(r);}
// Two-move score policy on the public state (identical to training/v23-stage2/train.cpp).
static Projection policy(const Game&g,const Model4&m){
 double best=-1e300;Projection choice;
 for(int d=0;d<4;d++){auto mv=move(g.b,d);if(!mv.size)continue;double q=mv.reward;
  if(high(mv.b)<FINAL_RANK){double ex=0;
   for(int e=0;e<mv.size;e++)for(int c=0;c<g.next.size;c++){Board b=mv.b;int card=g.next.cards[c];b[mv.entries[e]]=card;double v=-1e300;
    for(int d2=0;d2<4;d2++){auto n=move(b,d2);if(!n.size)continue;double w=n.reward+(high(n.b)>=FINAL_RANK?0:m.value(n.b));v=std::max(v,w);}
    ex+=g.next.p[c]*(pts(card)+(v==-1e300?0:v));}
   q+=ex/mv.size;}
  if(q>best){best=q;choice=mv;}}
 return choice;}
struct Snap{Game g;Counts c;};
static std::vector<Snap> readPool(const std::string&path){std::ifstream f(path);if(!f)throw std::runtime_error("no pool");std::vector<Snap>out;std::string line;
 while(std::getline(f,line)){std::istringstream in(line);Snap s;uint32_t seed;int n;in>>seed>>s.g.turns>>s.g.left;for(int i=0;i<s.g.left;i++){in>>n;s.g.deck[i]=n;}
  in>>s.g.next.size>>n;s.g.next.bonus=n;for(int i=0;i<3;i++){in>>n>>s.g.next.p[i];s.g.next.cards[i]=n;}for(auto&v:s.c)in>>v;for(auto&r:s.g.b){in>>n;r=n;}
  if(!in)throw std::runtime_error("bad pool line");if(high(s.g.b)!=14)throw std::runtime_error("pool state is not a 6144-phase state");out.push_back(s);}return out;}
int main(int argc,char**argv){try{
 if(argc<8)throw std::runtime_error("MODEL_IN MODEL_OUT MOVES SEED LOG STRONG_MOVE_SHARE POOL...");
 init();Model4 m;m.load(argv[1]);const uint64_t budget=std::stoull(argv[3]);RNG rng{uint32_t(std::stoul(argv[4]))};std::ofstream log(argv[5]);
 double strongShare=std::stod(argv[6]);
 std::vector<std::vector<Snap>>pools;for(int i=7;i<argc;i++)pools.push_back(readPool(argv[i]));
 const double alphas[3]={.02,.008,.003};
 Search strong(m);strong.maxNodes=INT_MAX;strong.threshold=0;strong.clampLeaf=true;
 const double W[5]={.5,.25,.125,.0625,.0625};
 auto t0=std::clock();double strongCPU=0;
 uint64_t episodes=0,steps=0,strongSteps=0,strongEpisodes=0,updates=0,cleared=0,deaths=0,winUpdates=0;double errSum=0;
 while(steps<budget){
  double alpha=steps<budget/2?alphas[0]:steps<budget*17/20?alphas[1]:alphas[2];
  auto&pool=pools[rng.index(pools.size())];Snap s=pool[rng.index(pool.size())];Game g=s.g;Counts c=s.c;
  for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
  std::vector<Rec>t;auto e0=std::clock();bool useStrong=double(strongSteps)<strongShare*double(steps);
  // Record k: afterstate of move k; reward = points of the card that then enters
  // + merge points of move k+1. Terminal afterstates (death after spawn, 12288) end it.
  while(!g.over){Projection mv;if(useStrong){auto a=strong.choose(g.b,g.next,c,3);mv=move(g.b,a.direction);}else mv=policy(g,m);
   if(!mv.size)throw std::runtime_error("no legal move");
   if(!t.empty())t.back().reward+=mv.reward;
   steps++;if(useStrong)strongSteps++;
   if(high(mv.b)>=FINAL_RANK){g.advance(mv,rng);cleared++;break;}
   if(stage4(mv.b)!=3)throw std::runtime_error("left stage 3");
   t.push_back({mv.b,m.value(mv.b,3),0});
   double spawn=g.advance(mv,rng);t.back().reward+=spawn;if(!g.over)c=observe(c,g.next);else deaths++;}
  for(size_t i=0;i<t.size();i++){
   double target=0,sum=0;size_t at=i;
   for(int n=0;n<5;n++){if(at<t.size()){sum+=t[at].reward;at++;}
    double boot=at<t.size()?t[at].value:0;target+=W[n]*(sum+boot);}
   errSum+=std::fabs(target-t[i].value);winUpdates++;m.update(t[i].b,3,target,alpha);updates++;}
  if(useStrong){strongCPU+=double(std::clock()-e0)/CLOCKS_PER_SEC;strongEpisodes++;}
  episodes++;
  if(episodes%2000==0){log<<"{\"episodes\":"<<episodes<<",\"cpu\":"<<double(std::clock()-t0)/CLOCKS_PER_SEC<<",\"steps\":"<<steps<<",\"updates\":"<<updates<<",\"cleared\":"<<cleared<<",\"deaths\":"<<deaths<<",\"meanAbsTDerr\":"<<(winUpdates?errSum/winUpdates:0)<<",\"alpha\":"<<alpha<<",\"strongEpisodes\":"<<strongEpisodes<<",\"strongSteps\":"<<strongSteps<<",\"strongCPU\":"<<strongCPU<<"}"<<std::endl;errSum=0;winUpdates=0;}
 }
 for(auto&tb:m.w)for(float v:tb)if(!std::isfinite(v))throw std::runtime_error("non-finite weight");
 m.save(argv[2]);
 log<<"{\"final\":true,\"episodes\":"<<episodes<<",\"steps\":"<<steps<<",\"updates\":"<<updates<<",\"cleared\":"<<cleared<<",\"deaths\":"<<deaths<<",\"strongEpisodes\":"<<strongEpisodes<<",\"strongSteps\":"<<strongSteps<<",\"strongCPU\":"<<strongCPU<<",\"cpu\":"<<double(std::clock()-t0)/CLOCKS_PER_SEC<<"}"<<std::endl;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
