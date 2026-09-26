// V23 stage-2 training: V21 trainer plus (1) a share of training CPU spent on episodes
// played by complete 3-ply expectimax (clamped leaves) instead of the two-move policy, and
// (2) optional TC (temporal coherence) per-weight learning rates with 2.5x base alphas.
//   train MODEL_IN MODEL_OUT CPU_SECONDS SEED LOG STRONG_SHARE TC(0/1) POOL [POOL...]
// Only stage 2 is updated; a 6144 afterstate is a bootstrap boundary (frozen stage 3).
// Derived from training/v21-late-train/train.cpp:
//   train MODEL_IN MODEL_OUT STAGES CPU_SECONDS SEED LOG POOL [POOL...]
// STAGES: which stage tables are updated, e.g. "12" (1536 and 3072) or "3" (>=6144).
// Episodes start from real snapshots (uniform pool, then uniform snapshot; the unseen
// deck is reshuffled). Behaviour policy: the two-move score policy used since V2
// (move, public preview spawn, best next move by merge points + afterstate value).
// Objective: future score (spawn points + merge points), undiscounted, as V1-V7.
// If stage 3 is not trained, an afterstate with a 6144 is a bootstrap boundary: its
// (frozen) value closes the return and the episode stops. 12288 is terminal (value 0).
// Targets: five-step truncated lambda return, lambda=.5 (weights .5,.25,.125,.0625,.0625),
// using values recorded when the trajectory was played. Learning rate steps down at
// 50% and 85% of the CPU budget. Hidden deck order is never used by the policy.
#include "../v21-late-train/search.h"
#include <ctime>
#include <climits>
using namespace v21;
struct Rec{Board b;int stage;double value,reward;bool boundary;};
static double pts(int r){return points(r);}
// Two-move score policy on the public state (same as v2::scorePolicy, Model4 leaves).
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
  if(!in)throw std::runtime_error("bad pool line");out.push_back(s);}return out;}
int main(int argc,char**argv){try{
 if(argc<9)throw std::runtime_error("MODEL_IN MODEL_OUT CPU_SECONDS SEED LOG STRONG_SHARE TC POOL...");
 init();Model4 m;m.load(argv[1]);double budget=std::stod(argv[3]);RNG rng{uint32_t(std::stoul(argv[4]))};std::ofstream log(argv[5]);
 double strongShare=std::stod(argv[6]);bool tc=std::stoi(argv[7])!=0;
 std::array<bool,4>train{};train[2]=true;
 std::vector<std::vector<Snap>>pools;for(int i=8;i<argc;i++)pools.push_back(readPool(argv[i]));
 const double k=tc?2.5:1;const double alphas[3]={.008*k,.003*k,.001*k};
 std::vector<float>tcE,tcA;if(tc){tcE.assign(WEIGHTS,0);tcA.assign(WEIGHTS,0);}
 Search strong(m);strong.maxNodes=INT_MAX;strong.threshold=0;strong.clampLeaf=true;double strongCPU=0;uint64_t strongEpisodes=0;
 const double W[5]={.5,.25,.125,.0625,.0625};
 auto t0=std::clock();uint64_t episodes=0,steps=0,updates=0,cleared=0,deaths=0,boundaries=0,winUpdates=0;double errSum=0;
 while(true){double used=double(std::clock()-t0)/CLOCKS_PER_SEC;if(used>=budget)break;
  double alpha=used<.5*budget?alphas[0]:used<.85*budget?alphas[1]:alphas[2];
  auto&pool=pools[rng.index(pools.size())];Snap s=pool[rng.index(pool.size())];Game g=s.g;Counts c=s.c;
  for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
  std::vector<Rec>t;auto e0=std::clock();bool useStrong=strongCPU<strongShare*used;
  // Record k: afterstate of move k; reward = points of the card that then enters
  // + merge points of move k+1. Terminal afterstates (death after spawn, 12288) end it.
  while(!g.over){Projection mv;if(useStrong){auto a=strong.choose(g.b,g.next,c,3);mv=move(g.b,a.direction);}else mv=policy(g,m);
   if(!mv.size)throw std::runtime_error("no legal move");
   if(!t.empty())t.back().reward+=mv.reward;
   if(high(mv.b)>=FINAL_RANK){g.advance(mv,rng);cleared++;break;}
   int st=stage4(mv.b);bool bound=st==3&&!train[3];
   t.push_back({mv.b,st,m.value(mv.b,st),0,bound});
   if(bound){boundaries++;break;}
   double spawn=g.advance(mv,rng);t.back().reward+=spawn;steps++;if(!g.over)c=observe(c,g.next);else deaths++;}
  for(size_t i=0;i<t.size();i++){if(t[i].boundary||!train[t[i].stage])continue;
   double target=0,sum=0;size_t at=i;
   for(int n=0;n<5;n++){if(at<t.size()&&!t[at].boundary){sum+=t[at].reward;at++;}
    double boot=at<t.size()?t[at].value:0;target+=W[n]*(sum+boot);}
   errSum+=std::fabs(target-t[i].value);winUpdates++;m.update(t[i].b,t[i].stage,target,alpha,tc?&tcE:nullptr,tc?&tcA:nullptr);updates++;}
  if(useStrong){strongCPU+=double(std::clock()-e0)/CLOCKS_PER_SEC;strongEpisodes++;}
  episodes++;
  if(episodes%2000==0){log<<"{\"episodes\":"<<episodes<<",\"cpu\":"<<used<<",\"steps\":"<<steps<<",\"updates\":"<<updates<<",\"cleared\":"<<cleared<<",\"deaths\":"<<deaths<<",\"boundaries\":"<<boundaries<<",\"meanAbsTDerr\":"<<(winUpdates?errSum/winUpdates:0)<<",\"alpha\":"<<alpha<<",\"strongEpisodes\":"<<strongEpisodes<<",\"strongCPU\":"<<strongCPU<<"}"<<std::endl;errSum=0;winUpdates=0;}
 }
 for(auto&tb:m.w)for(float v:tb)if(!std::isfinite(v))throw std::runtime_error("non-finite weight");
 m.save(argv[2]);
 log<<"{\"final\":true,\"episodes\":"<<episodes<<",\"steps\":"<<steps<<",\"updates\":"<<updates<<",\"cleared\":"<<cleared<<",\"deaths\":"<<deaths<<",\"boundaries\":"<<boundaries<<",\"strongEpisodes\":"<<strongEpisodes<<",\"strongCPU\":"<<strongCPU<<",\"cpu\":"<<double(std::clock()-t0)/CLOCKS_PER_SEC<<"}"<<std::endl;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
