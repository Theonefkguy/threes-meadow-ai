// V32: training/v31-stages/train.cpp with (1) a configurable strong player (expectimax depth and
// probability cutoff, clamped leaves) and (2) a wall-clock budget instead of a move budget: the
// rate steps at 50%/85% of SECONDS, checkpoints are fractions of SECONDS, and threads stop at the
// first episode start after SECONDS. The strong-move share rule is unchanged, except that a share
// of 1 (or more) makes every episode strong.
//   train MODEL_IN RES_LIST STAGE THREADS SECONDS SEED LOG STRONG_MOVE_SHARE STRONG_DEPTH STRONG_THRESHOLD CHECKPOINT_FRACTIONS OUT_PREFIX POOL [POOL...]
// Original V31 note:
// V31 parallel training of one stage S (1, 2 or 3): its four-cell table plus its five-cell
// residual, Hogwild style. training/v30-parallel/train.cpp generalised to any stage:
//   train MODEL_IN RES_LIST STAGE THREADS MOVES SEED LOG STRONG_MOVE_SHARE CHECKPOINTS OUT_PREFIX POOL [POOL...]
// RES_LIST is a ModelS list (model5s.h), e.g. "W30.res,zero:1" keeps the V30 stage-2 residual
// fixed and trains a new zero-initialised stage-1 residual. Pool states must be in stage S
// (max card 1536/3072/6144 for S=1/2/3). An afterstate of a later stage closes the episode with
// its frozen value (including that stage's residual); 12288 is terminal (value 0). Only stage S
// is updated. Otherwise as V30 (V23 stage-2 method, two-move behaviour policy plus strong 3-ply clamped episodes,
// TD(lambda=.5) five-step score return, rates .001/.0005/.00025 stepping at 50%/85% of MOVES),
// but THREADS threads play episodes concurrently and update one shared model without locks.
// Counters (moves, strong moves) are global atomics: the rate and the strong-episode rule use
// the global totals, and each thread stops at the first episode start after MOVES is reached.
// Thread t uses RNG seed SEED+t. Lock-free updates race (lost or interleaved float writes),
// so the result is NOT bit-reproducible; it is the usual trade made by parallel n-tuple TD.
// A checkpoint is written by the thread whose episode first reaches it, while others continue.
#include "../v31-stages/search5s.h"
#include <atomic>
#include <ctime>
#include <climits>
#include <mutex>
#include <thread>
#include <chrono>
using namespace v31;
struct Rec{Board b;double value,reward;bool boundary;};
static double pts(int r){return points(r);}
// Two-move score policy on the public state (as training/v29-five/train.cpp).
static Projection policy(const Game&g,const ModelS&m){
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
static std::vector<Snap> readPool(const std::string&path,int rank){std::ifstream f(path);if(!f)throw std::runtime_error("no pool");std::vector<Snap>out;std::string line;
 while(std::getline(f,line)){std::istringstream in(line);Snap s;uint32_t seed;int n;in>>seed>>s.g.turns>>s.g.left;for(int i=0;i<s.g.left;i++){in>>n;s.g.deck[i]=n;}
  in>>s.g.next.size>>n;s.g.next.bonus=n;for(int i=0;i<3;i++){in>>n>>s.g.next.p[i];s.g.next.cards[i]=n;}for(auto&v:s.c)in>>v;for(auto&r:s.g.b){in>>n;r=n;}
  if(!in)throw std::runtime_error("bad pool line");if(high(s.g.b)!=rank)throw std::runtime_error("pool state is not in the trained stage");out.push_back(s);}return out;}
static int STAGE=2;
static void save(const ModelS&m,const std::string&prefix,const std::string&tag){m.base.save(prefix+"-"+tag+".bin");if(m.has[STAGE])m.saveRes(STAGE,prefix+"-"+tag+".res");}
int main(int argc,char**argv){try{
 if(argc<14)throw std::runtime_error("MODEL_IN RES_LIST STAGE THREADS SECONDS SEED LOG STRONG_MOVE_SHARE STRONG_DEPTH STRONG_THRESHOLD CHECKPOINT_FRACTIONS OUT_PREFIX POOL...");
 v31::init();ModelS m;m.load(argv[1],argv[2]);STAGE=std::stoi(argv[3]);if(STAGE<1||STAGE>3)throw std::runtime_error("STAGE must be 1, 2 or 3");
 const int threads=std::stoi(argv[4]);const double seconds=std::stod(argv[5]);const uint32_t seed=std::stoul(argv[6]);std::ofstream log(argv[7]);
 const double strongShare=std::stod(argv[8]);const int strongDepth=std::stoi(argv[9]);const double strongThreshold=std::stod(argv[10]);
 std::vector<double>checks;{std::stringstream cs(argv[11]);std::string tok;while(std::getline(cs,tok,','))if(!tok.empty())checks.push_back(std::stod(tok));}
 for(size_t i=1;i<checks.size();i++)if(checks[i]<=checks[i-1])throw std::runtime_error("checkpoints must ascend");
 if(!checks.empty()&&(checks.front()<=0||checks.back()>=1))throw std::runtime_error("checkpoint fractions must be in (0,1)");
 const std::string prefix=argv[12];
 std::vector<std::vector<Snap>>pools;for(int i=13;i<argc;i++)pools.push_back(readPool(argv[i],11+STAGE));
 const double alphas[3]={.001,.0005,.00025};const double W[5]={.5,.25,.125,.0625,.0625};
 std::atomic<uint64_t>steps{0},strongSteps{0},episodes{0},strongEpisodes{0},updates{0},boundaries{0},deaths{0},cleared{0};std::atomic<size_t>nextCheck{0};
 std::atomic<bool>failed{false};std::mutex logMutex,saveMutex;std::string error;
 auto start=std::chrono::steady_clock::now();
 auto worker=[&](int id){try{
  RNG rng{seed+uint32_t(id)};Search strong(m);strong.maxNodes=INT_MAX;strong.threshold=strongThreshold;strong.clampLeaf=true;double errSum=0;uint64_t errN=0,mine=0;
  auto elapsed=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
  while(!failed){double frac=elapsed()/seconds;if(frac>=1)break;uint64_t now=steps.load(std::memory_order_relaxed);
   double alpha=frac<.5?alphas[0]:frac<.85?alphas[1]:alphas[2];
   auto&pool=pools[rng.index(pools.size())];Snap s=pool[rng.index(pool.size())];Game g=s.g;Counts c=s.c;
   for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
   bool useStrong=strongShare>=1||double(strongSteps.load(std::memory_order_relaxed))<strongShare*double(now);
   std::vector<Rec>t;uint64_t n=0;
   while(!g.over){Projection mv;if(useStrong){auto a=strong.choose(g.b,g.next,c,strongDepth);if(strong.completed!=strongDepth)throw std::runtime_error("incomplete strong depth");mv=move(g.b,a.direction);}else mv=policy(g,m);
    if(!mv.size)throw std::runtime_error("no legal move");
    if(!t.empty())t.back().reward+=mv.reward;n++;
    if(high(mv.b)>=FINAL_RANK){g.advance(mv,rng);cleared++;break;}
    int st=stage4(mv.b);if(st<STAGE)throw std::runtime_error("left the trained stage");
    t.push_back({mv.b,m.value(mv.b,st),0,st>STAGE});
    if(st>STAGE){boundaries++;break;}
    double spawn=g.advance(mv,rng);t.back().reward+=spawn;if(!g.over)c=observe(c,g.next);else deaths++;}
   for(size_t i=0;i<t.size();i++){if(t[i].boundary)continue;
    double target=0,sum=0;size_t at=i;
    for(int k=0;k<5;k++){if(at<t.size()&&!t[at].boundary){sum+=t[at].reward;at++;}
     double boot=at<t.size()?t[at].value:0;target+=W[k]*(sum+boot);}
    errSum+=std::fabs(target-t[i].value);errN++;m.update(t[i].b,STAGE,target,alpha);updates++;}
   uint64_t total=steps.fetch_add(n)+n;if(useStrong){strongSteps+=n;strongEpisodes++;}uint64_t ep=++episodes;mine++;
   size_t ck=nextCheck.load();
   double fnow=elapsed()/seconds;
   if(ck<checks.size()&&fnow>=checks[ck]){std::lock_guard<std::mutex>lk(saveMutex);ck=nextCheck.load();
    if(ck<checks.size()&&fnow>=checks[ck]){save(m,prefix,"f"+std::to_string(int(checks[ck]*100+.5)));nextCheck=ck+1;
     std::lock_guard<std::mutex>ll(logMutex);log<<"{\"checkpoint\":"<<checks[ck]<<",\"steps\":"<<total<<",\"episodes\":"<<ep<<"}"<<std::endl;}}
   if(id==0&&mine%500==0){double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::lock_guard<std::mutex>ll(logMutex);log<<"{\"wall\":"<<wall<<",\"steps\":"<<total<<",\"episodes\":"<<ep<<",\"updates\":"<<updates.load()<<",\"boundaries\":"<<boundaries.load()<<",\"deaths\":"<<deaths.load()<<",\"cleared\":"<<cleared.load()
     <<",\"meanAbsTDerrThread0\":"<<(errN?errSum/errN:0)<<",\"alpha\":"<<alpha<<",\"strongEpisodes\":"<<strongEpisodes.load()<<",\"strongSteps\":"<<strongSteps.load()<<",\"movesPerSecond\":"<<total/wall<<"}"<<std::endl;errSum=0;errN=0;}
  }}catch(const std::exception&e){std::lock_guard<std::mutex>ll(logMutex);error=e.what();failed=true;}};
 std::vector<std::thread>pool;for(int i=0;i<threads;i++)pool.emplace_back(worker,i);for(auto&th:pool)th.join();
 if(failed)throw std::runtime_error(error);
 for(auto&tb:m.base.w)for(float v:tb)if(!std::isfinite(v))throw std::runtime_error("non-finite weight");
 for(auto&r:m.res)for(float v:r)if(!std::isfinite(v))throw std::runtime_error("non-finite residual");
 save(m,prefix,"final");
 double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 log<<"{\"final\":true,\"stage\":"<<STAGE<<",\"threads\":"<<threads<<",\"wall\":"<<wall<<",\"episodes\":"<<episodes.load()<<",\"steps\":"<<steps.load()<<",\"updates\":"<<updates.load()<<",\"boundaries\":"<<boundaries.load()<<",\"deaths\":"<<deaths.load()<<",\"cleared\":"<<cleared.load()<<",\"strongEpisodes\":"<<strongEpisodes.load()<<",\"strongSteps\":"<<strongSteps.load()<<"}"<<std::endl;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
