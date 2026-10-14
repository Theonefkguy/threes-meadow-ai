// V29 stage-2 (3072 -> 6144) training with an optional five-cell residual.
//   train MODEL_IN RES MOVES SEED LOG STRONG_MOVE_SHARE CHECKPOINTS OUT_PREFIX POOL [POOL...]
// RES: "-" trains the four-cell stage-2 table only (control); "zero" adds an all-zero
// five-cell residual on stage 2 and trains both jointly (model5.h). CHECKPOINTS: ascending
// comma-separated move counts below MOVES ("" for none).
// Writes OUT_PREFIX-<moves>.bin (four-stage model) and, with a residual, OUT_PREFIX-<moves>.res
// at each checkpoint (end of the first episode reaching it) and OUT_PREFIX-final.* at the end.
// Method: training/v23-stage2/train.cpp (stage 2 only; a 6144 afterstate is a bootstrap
// boundary closed by the frozen stage-3 value; two-move behaviour policy plus strong episodes
// by complete 3-ply expectimax with clamped leaves; TD(lambda=.5) five-step score return),
// with the move budget and move-share strong rule of training/v27-stage3/train.cpp, and
// learning rates .001/.0005/.00025 stepping at 50%/85% of the move budget (V28: large rates
// destroy an already-trained table; .001 is the rate the current stage 2 finished on).
// Deterministic given the arguments, compiler and flags.
#include "search5.h"
#include <ctime>
#include <climits>
using namespace v29;
struct Rec{Board b;double value,reward;bool boundary;};
static double pts(int r){return points(r);}
// Two-move score policy on the public state (as training/v23-stage2/train.cpp).
static Projection policy(const Game&g,const ModelR&m){
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
  if(!in)throw std::runtime_error("bad pool line");if(high(s.g.b)!=13)throw std::runtime_error("pool state is not a 3072-phase state");out.push_back(s);}return out;}
static void save(const ModelR&m,const std::string&prefix,const std::string&tag){m.base.save(prefix+"-"+tag+".bin");if(m.hasRes)m.saveRes(prefix+"-"+tag+".res");}
int main(int argc,char**argv){try{
 if(argc<10)throw std::runtime_error("MODEL_IN RES MOVES SEED LOG STRONG_MOVE_SHARE CHECKPOINTS OUT_PREFIX POOL...");
 v29::init();ModelR m;std::string resMode=argv[2];if(resMode!="-"&&resMode!="zero")throw std::runtime_error("RES must be - or zero");m.load(argv[1],resMode);
 const uint64_t budget=std::stoull(argv[3]);RNG rng{uint32_t(std::stoul(argv[4]))};std::ofstream log(argv[5]);double strongShare=std::stod(argv[6]);
 std::vector<uint64_t>checks;{std::stringstream cs(argv[7]);std::string tok;while(std::getline(cs,tok,','))if(!tok.empty())checks.push_back(std::stoull(tok));}
 for(size_t i=1;i<checks.size();i++)if(checks[i]<=checks[i-1])throw std::runtime_error("checkpoints must ascend");
 if(!checks.empty()&&checks.back()>=budget)throw std::runtime_error("checkpoint beyond budget");
 std::string prefix=argv[8];size_t nextCheck=0;
 std::vector<std::vector<Snap>>pools;for(int i=9;i<argc;i++)pools.push_back(readPool(argv[i]));
 const double alphas[3]={.001,.0005,.00025};
 Search strong(m);strong.maxNodes=INT_MAX;strong.threshold=0;strong.clampLeaf=true;
 const double W[5]={.5,.25,.125,.0625,.0625};
 auto t0=std::clock();double strongCPU=0;
 uint64_t episodes=0,steps=0,strongSteps=0,strongEpisodes=0,updates=0,boundaries=0,deaths=0,winUpdates=0;double errSum=0;
 while(steps<budget){
  double alpha=steps<budget/2?alphas[0]:steps<budget*17/20?alphas[1]:alphas[2];
  auto&pool=pools[rng.index(pools.size())];Snap s=pool[rng.index(pool.size())];Game g=s.g;Counts c=s.c;
  for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
  std::vector<Rec>t;auto e0=std::clock();bool useStrong=double(strongSteps)<strongShare*double(steps);
  // Record k: afterstate of move k; reward = points of the card that then enters + merge
  // points of move k+1. A 6144 afterstate closes the episode with its frozen stage-3 value.
  while(!g.over){Projection mv;if(useStrong){auto a=strong.choose(g.b,g.next,c,3);mv=move(g.b,a.direction);}else mv=policy(g,m);
   if(!mv.size)throw std::runtime_error("no legal move");
   if(!t.empty())t.back().reward+=mv.reward;
   steps++;if(useStrong)strongSteps++;
   int st=stage4(mv.b);if(st<2)throw std::runtime_error("left stage 2");
   t.push_back({mv.b,m.value(mv.b,st),0,st==3});
   if(st==3){boundaries++;break;}
   double spawn=g.advance(mv,rng);t.back().reward+=spawn;if(!g.over)c=observe(c,g.next);else deaths++;}
  for(size_t i=0;i<t.size();i++){if(t[i].boundary)continue;
   double target=0,sum=0;size_t at=i;
   for(int n=0;n<5;n++){if(at<t.size()&&!t[at].boundary){sum+=t[at].reward;at++;}
    double boot=at<t.size()?t[at].value:0;target+=W[n]*(sum+boot);}
   errSum+=std::fabs(target-t[i].value);winUpdates++;m.update(t[i].b,2,target,alpha);updates++;}
  if(useStrong){strongCPU+=double(std::clock()-e0)/CLOCKS_PER_SEC;strongEpisodes++;}
  episodes++;
  if(nextCheck<checks.size()&&steps>=checks[nextCheck]){save(m,prefix,std::to_string(checks[nextCheck]));
   log<<"{\"checkpoint\":"<<checks[nextCheck]<<",\"steps\":"<<steps<<",\"episodes\":"<<episodes<<"}"<<std::endl;nextCheck++;}
  if(episodes%2000==0){log<<"{\"episodes\":"<<episodes<<",\"cpu\":"<<double(std::clock()-t0)/CLOCKS_PER_SEC<<",\"steps\":"<<steps<<",\"updates\":"<<updates<<",\"boundaries\":"<<boundaries<<",\"deaths\":"<<deaths<<",\"meanAbsTDerr\":"<<(winUpdates?errSum/winUpdates:0)<<",\"alpha\":"<<alpha<<",\"strongEpisodes\":"<<strongEpisodes<<",\"strongSteps\":"<<strongSteps<<",\"strongCPU\":"<<strongCPU<<"}"<<std::endl;errSum=0;winUpdates=0;}
 }
 for(auto&tb:m.base.w)for(float v:tb)if(!std::isfinite(v))throw std::runtime_error("non-finite weight");
 for(float v:m.res)if(!std::isfinite(v))throw std::runtime_error("non-finite residual");
 save(m,prefix,"final");
 log<<"{\"final\":true,\"episodes\":"<<episodes<<",\"steps\":"<<steps<<",\"updates\":"<<updates<<",\"boundaries\":"<<boundaries<<",\"deaths\":"<<deaths<<",\"strongEpisodes\":"<<strongEpisodes<<",\"strongSteps\":"<<strongSteps<<",\"strongCPU\":"<<strongCPU<<",\"cpu\":"<<double(std::clock()-t0)/CLOCKS_PER_SEC<<"}"<<std::endl;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
