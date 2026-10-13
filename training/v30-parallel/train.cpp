// V30 parallel stage-2 training (four-cell table plus five-cell residual), Hogwild style.
//   train MODEL_IN RES THREADS MOVES SEED LOG STRONG_MOVE_SHARE CHECKPOINTS OUT_PREFIX POOL [POOL...]
// Same per-episode method as training/v29-five/train.cpp (V23 stage-2 method, 6144 bootstrap
// boundary with frozen stage 3, two-move behaviour policy plus strong 3-ply clamped episodes,
// TD(lambda=.5) five-step score return, rates .001/.0005/.00025 stepping at 50%/85% of MOVES),
// but THREADS threads play episodes concurrently and update one shared model without locks.
// Counters (moves, strong moves) are global atomics: the rate and the strong-episode rule use
// the global totals, and each thread stops at the first episode start after MOVES is reached.
// Thread t uses RNG seed SEED+t. Lock-free updates race (lost or interleaved float writes),
// so the result is NOT bit-reproducible; it is the usual trade made by parallel n-tuple TD.
// A checkpoint is written by the thread whose episode first reaches it, while others continue.
#include "../v29-five/search5.h"
#include <atomic>
#include <ctime>
#include <climits>
#include <mutex>
#include <thread>
#include <chrono>
using namespace v29;
struct Rec{Board b;double value,reward;bool boundary;};
static double pts(int r){return points(r);}
// Two-move score policy on the public state (as training/v29-five/train.cpp).
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
 if(argc<11)throw std::runtime_error("MODEL_IN RES THREADS MOVES SEED LOG STRONG_MOVE_SHARE CHECKPOINTS OUT_PREFIX POOL...");
 v29::init();ModelR m;std::string resMode=argv[2];if(resMode!="-"&&resMode!="zero")throw std::runtime_error("RES must be - or zero");m.load(argv[1],resMode);
 const int threads=std::stoi(argv[3]);const uint64_t budget=std::stoull(argv[4]);const uint32_t seed=std::stoul(argv[5]);std::ofstream log(argv[6]);
 const double strongShare=std::stod(argv[7]);
 std::vector<uint64_t>checks;{std::stringstream cs(argv[8]);std::string tok;while(std::getline(cs,tok,','))if(!tok.empty())checks.push_back(std::stoull(tok));}
 for(size_t i=1;i<checks.size();i++)if(checks[i]<=checks[i-1])throw std::runtime_error("checkpoints must ascend");
 if(!checks.empty()&&checks.back()>=budget)throw std::runtime_error("checkpoint beyond budget");
 const std::string prefix=argv[9];
 std::vector<std::vector<Snap>>pools;for(int i=10;i<argc;i++)pools.push_back(readPool(argv[i]));
 const double alphas[3]={.001,.0005,.00025};const double W[5]={.5,.25,.125,.0625,.0625};
 std::atomic<uint64_t>steps{0},strongSteps{0},episodes{0},strongEpisodes{0},updates{0},boundaries{0},deaths{0};std::atomic<size_t>nextCheck{0};
 std::atomic<bool>failed{false};std::mutex logMutex,saveMutex;std::string error;
 auto start=std::chrono::steady_clock::now();
 auto worker=[&](int id){try{
  RNG rng{seed+uint32_t(id)};Search strong(m);strong.maxNodes=INT_MAX;strong.threshold=0;strong.clampLeaf=true;double errSum=0;uint64_t errN=0,mine=0;
  while(!failed){uint64_t now=steps.load(std::memory_order_relaxed);if(now>=budget)break;
   double alpha=now<budget/2?alphas[0]:now<budget*17/20?alphas[1]:alphas[2];
   auto&pool=pools[rng.index(pools.size())];Snap s=pool[rng.index(pool.size())];Game g=s.g;Counts c=s.c;
   for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
   bool useStrong=double(strongSteps.load(std::memory_order_relaxed))<strongShare*double(now);
   std::vector<Rec>t;uint64_t n=0;
   while(!g.over){Projection mv;if(useStrong){auto a=strong.choose(g.b,g.next,c,3);mv=move(g.b,a.direction);}else mv=policy(g,m);
    if(!mv.size)throw std::runtime_error("no legal move");
    if(!t.empty())t.back().reward+=mv.reward;n++;
    int st=stage4(mv.b);if(st<2)throw std::runtime_error("left stage 2");
    t.push_back({mv.b,m.value(mv.b,st),0,st==3});
    if(st==3){boundaries++;break;}
    double spawn=g.advance(mv,rng);t.back().reward+=spawn;if(!g.over)c=observe(c,g.next);else deaths++;}
   for(size_t i=0;i<t.size();i++){if(t[i].boundary)continue;
    double target=0,sum=0;size_t at=i;
    for(int k=0;k<5;k++){if(at<t.size()&&!t[at].boundary){sum+=t[at].reward;at++;}
     double boot=at<t.size()?t[at].value:0;target+=W[k]*(sum+boot);}
    errSum+=std::fabs(target-t[i].value);errN++;m.update(t[i].b,2,target,alpha);updates++;}
   uint64_t total=steps.fetch_add(n)+n;if(useStrong){strongSteps+=n;strongEpisodes++;}uint64_t ep=++episodes;mine++;
   size_t ck=nextCheck.load();
   if(ck<checks.size()&&total>=checks[ck]){std::lock_guard<std::mutex>lk(saveMutex);ck=nextCheck.load();
    if(ck<checks.size()&&total>=checks[ck]){save(m,prefix,std::to_string(checks[ck]));nextCheck=ck+1;
     std::lock_guard<std::mutex>ll(logMutex);log<<"{\"checkpoint\":"<<checks[ck]<<",\"steps\":"<<total<<",\"episodes\":"<<ep<<"}"<<std::endl;}}
   if(id==0&&mine%500==0){double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::lock_guard<std::mutex>ll(logMutex);log<<"{\"wall\":"<<wall<<",\"steps\":"<<total<<",\"episodes\":"<<ep<<",\"updates\":"<<updates.load()<<",\"boundaries\":"<<boundaries.load()<<",\"deaths\":"<<deaths.load()
     <<",\"meanAbsTDerrThread0\":"<<(errN?errSum/errN:0)<<",\"alpha\":"<<alpha<<",\"strongEpisodes\":"<<strongEpisodes.load()<<",\"strongSteps\":"<<strongSteps.load()<<",\"movesPerSecond\":"<<total/wall<<"}"<<std::endl;errSum=0;errN=0;}
  }}catch(const std::exception&e){std::lock_guard<std::mutex>ll(logMutex);error=e.what();failed=true;}};
 std::vector<std::thread>pool;for(int i=0;i<threads;i++)pool.emplace_back(worker,i);for(auto&th:pool)th.join();
 if(failed)throw std::runtime_error(error);
 for(auto&tb:m.base.w)for(float v:tb)if(!std::isfinite(v))throw std::runtime_error("non-finite weight");
 for(float v:m.res)if(!std::isfinite(v))throw std::runtime_error("non-finite residual");
 save(m,prefix,"final");
 double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 log<<"{\"final\":true,\"threads\":"<<threads<<",\"wall\":"<<wall<<",\"episodes\":"<<episodes.load()<<",\"steps\":"<<steps.load()<<",\"updates\":"<<updates.load()<<",\"boundaries\":"<<boundaries.load()<<",\"deaths\":"<<deaths.load()<<",\"strongEpisodes\":"<<strongEpisodes.load()<<",\"strongSteps\":"<<strongSteps.load()<<"}"<<std::endl;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
