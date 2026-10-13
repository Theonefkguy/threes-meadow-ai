// V31 tools: training/v29-five/tools.cpp with ModelS; RES is "-" or a comma-separated residual list.
// V29 tools (official-modern rules; four-stage model plus optional five-cell stage-2 residual).
// Helpers are copied from training/v21-late-train/snap.cpp; the model type is v29::ModelR.
// RES is a residual file, "-" (none) or "zero" (all-zero residual; must equal "-").
//  pool   MODEL RES FIRST_SEED GAMES RANK OUT_STATES OUT_LOG
//         website default AI (depth 5, cutoff 0.01, clamp) from standard openings until the first
//         card of rank RANK (13 = 3072); the state is appended unless the game is already over.
//  sample MODEL RES POOL FIRST COUNT ROLLOUT_BASE EVERY RANK TARGET OUT
//         continue with the website search to rank TARGET or death, appending the state every
//         EVERY moves (from move 0) while the max card is exactly rank RANK.
//  cont   MODEL RES DEPTH THR POOL FIRST COUNT ROLLOUT_BASE TARGET OUT  (env CLAMP=1)
//         snap.cpp cont: continue to rank TARGET or death, one JSON line per state.
// Line format: seed turn left deck[left] next.size bonus (card p)x3 counts[3] board[16]
#include "search5s.h"
#include <climits>
#include <ctime>
using namespace v31;
struct Snap{uint32_t seed=0;Game g;Counts c{};};
static void write(std::ostream&o,uint32_t seed,const Game&g,Counts c){
 o<<std::setprecision(17)<<seed<<' '<<g.turns<<' '<<g.left;for(int i=0;i<g.left;i++)o<<' '<<int(g.deck[i]);
 o<<' '<<g.next.size<<' '<<int(g.next.bonus);for(int i=0;i<3;i++)o<<' '<<int(g.next.cards[i])<<' '<<g.next.p[i];
 for(auto v:c)o<<' '<<v;for(auto r:g.b)o<<' '<<int(r);o<<'\n';o.flush();}
static std::vector<Snap> readPool(const std::string&path){std::ifstream f(path);if(!f)throw std::runtime_error("no pool "+path);std::vector<Snap>out;std::string line;
 while(std::getline(f,line)){std::istringstream in(line);Snap s;int n;in>>s.seed>>s.g.turns>>s.g.left;
  if(s.g.left<0||s.g.left>12)throw std::runtime_error("bad deck");for(int i=0;i<s.g.left;i++){in>>n;s.g.deck[i]=n;}
  in>>s.g.next.size>>n;s.g.next.bonus=n;for(int i=0;i<3;i++){in>>n>>s.g.next.p[i];s.g.next.cards[i]=n;}
  for(auto&v:s.c)in>>v;for(auto&r:s.g.b){in>>n;r=n;}if(!in)throw std::runtime_error("bad pool line");
  Counts d{0,0,0};for(int i=0;i<s.g.left;i++)d[s.g.deck[i]-1]++;
  if(d!=s.c&&!(s.c==Counts{0,0,0}&&s.g.left==0))throw std::runtime_error("counts do not match unseen deck");
  out.push_back(s);}return out;}
static int SAMPLE_EVERY=0,SAMPLE_RANK=13;static std::ofstream*SAMPLE_OUT=nullptr;static uint32_t SAMPLE_SEED=0;
struct Result{int risky=0,riskyBig=0;double riskySum=0,riskyMax=0;int lastKind=0;bool success=false;int moves=0,maxSecond=0;double gained=0,cpu=0,mx=0;uint64_t nodes=0;Game g;Counts c;};
// Second-largest card rank on the board (a 6144 plus a 3072 gives 13).
static int second(const Board&b){int a=0,c=0;for(int v:b){if(v>a){c=a;a=v;}else if(v>c)c=v;}return c;}
// Probability that a move leaves no legal move right after the new card enters
// (public information only: lanes uniform, card by the shown preview).
static double pDeath(const Projection&m,const Preview&p){if(high(m.b)>=FINAL_RANK)return 0;double d=0;
 for(int e=0;e<m.size;e++)for(int i=0;i<p.size;i++){Board b=m.b;b[m.entries[e]]=p.cards[i];if(!legal(b))d+=p.p[i]/m.size;}return d;}
static Result play(Search&s,Snap sn,int depth,uint32_t seed,int target){
 Result r;Game&g=r.g;g=sn.g;Counts c=sn.c;RNG rng{seed};for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
 double start=score(g.b);
 while(!g.over&&high(g.b)<target&&r.moves<30000){if(SAMPLE_EVERY&&r.moves%SAMPLE_EVERY==0&&high(g.b)==SAMPLE_RANK)write(*SAMPLE_OUT,SAMPLE_SEED,g,c);auto t0=std::clock();auto a=s.choose(g.b,g.next,c,depth);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;
  if(s.completed!=depth)throw std::runtime_error("incomplete depth");r.cpu+=el;r.mx=std::max(r.mx,el);r.nodes+=s.nodes;r.moves++;
  auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");
  {double mine=pDeath(p,g.next),best=1;for(int d=0;d<4;d++){auto q=move(g.b,d);if(q.size)best=std::min(best,pDeath(q,g.next));}
   // lastKind: 1 = certain death with no alternative, 2 = certain death although a safer move existed
   r.lastKind=mine>=1-1e-12?(best>=1-1e-12?1:2):0;
   if(mine>best+1e-12){r.risky++;r.riskySum+=mine-best;r.riskyMax=std::max(r.riskyMax,mine-best);if(p.reward>=177147)r.riskyBig++;}}
  g.advance(p,rng);if(!g.over)c=observe(c,g.next);r.maxSecond=std::max(r.maxSecond,second(g.b));}
 if(!g.over&&high(g.b)<target)throw std::runtime_error("truncated");
 r.success=high(g.b)>=target;r.gained=score(g.b)-start;r.c=c;return r;}
int main(int argc,char**argv){try{
 v31::init();std::string mode=argv[1];
 if(mode=="pool"){if(argc!=9)throw std::runtime_error("pool MODEL RES FIRST_SEED GAMES RANK OUT_STATES OUT_LOG");
  ModelS m;m.load(argv[2],argv[3]);uint32_t first=std::stoul(argv[4]);int games=std::stoi(argv[5]),rank=std::stoi(argv[6]);
  std::ofstream states(argv[7],std::ios::app),log(argv[8],std::ios::app);Search s(m);s.maxNodes=INT_MAX;s.threshold=0.01;s.clampLeaf=true;
  for(int k=0;k<games;k++){uint32_t seed=first+k;RNG rng{seed};Game g;g.reset(rng);Counts c=opening(g);
   while(!g.over&&high(g.b)<rank&&g.turns<50000){auto a=s.choose(g.b,g.next,c,5);if(s.completed!=5)throw std::runtime_error("incomplete depth");
    auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);if(g.over)break;c=observe(c,g.next);}
   if(!g.over&&high(g.b)<rank)throw std::runtime_error("move guard");
   bool reached=high(g.b)>=rank,kept=reached&&!g.over;if(kept){write(states,seed,g,c);states.flush();}
   log<<"{\"seed\":"<<seed<<",\"reached\":"<<(reached?"true":"false")<<",\"kept\":"<<(kept?"true":"false")<<",\"turn\":"<<g.turns<<"}"<<std::endl;}
  return 0;}
 if(mode=="sample"){if(argc!=12)throw std::runtime_error("sample MODEL RES POOL FIRST COUNT ROLLOUT_BASE EVERY RANK TARGET OUT");
  ModelS m;m.load(argv[2],argv[3]);auto pool=readPool(argv[4]);int first=std::stoi(argv[5]),count=std::stoi(argv[6]);uint32_t base=std::stoul(argv[7]);
  SAMPLE_EVERY=std::stoi(argv[8]);SAMPLE_RANK=std::stoi(argv[9]);int target=std::stoi(argv[10]);std::ofstream out(argv[11],std::ios::app);SAMPLE_OUT=&out;
  Search s(m);s.maxNodes=INT_MAX;s.threshold=0.01;s.clampLeaf=true;
  for(int idx=first;idx<first+count&&idx<int(pool.size());idx++){SAMPLE_SEED=pool[idx].seed;play(s,pool[idx],5,base+uint32_t(idx),target);}
  return 0;}
 if(mode!="cont"||argc!=12)throw std::runtime_error("cont MODEL RES DEPTH THR POOL FIRST COUNT ROLLOUT_BASE TARGET OUT");
 ModelS m;m.load(argv[2],argv[3]);int depth=std::stoi(argv[4]);double thr=std::stod(argv[5]);auto pool=readPool(argv[6]);
 int first=std::stoi(argv[7]),count=std::stoi(argv[8]);uint32_t base=std::stoul(argv[9]);int target=std::stoi(argv[10]);std::ofstream out(argv[11],std::ios::app);out<<std::setprecision(17);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=thr;s.clampLeaf=std::getenv("CLAMP")&&std::getenv("CLAMP")[0]=='1';
 for(int idx=first;idx<first+count&&idx<int(pool.size());idx++){auto r=play(s,pool[idx],depth,base+uint32_t(idx),target);
  out<<"{\"index\":"<<idx<<",\"sourceSeed\":"<<pool[idx].seed<<",\"depth\":"<<depth<<",\"threshold\":"<<thr<<",\"startRank\":"<<high(pool[idx].g.b)
     <<",\"success\":"<<(r.success?"true":"false")<<",\"finalRank\":"<<high(r.g.b)<<",\"moves\":"<<r.moves<<",\"maxSecondRank\":"<<r.maxSecond<<",\"died\":"<<(r.g.over&&!r.success?"true":"false")<<",\"overAtEnd\":"<<(r.g.over?"true":"false")<<",\"lastKind\":"<<r.lastKind<<",\"riskyMoves\":"<<r.risky<<",\"riskyExtraP\":"<<r.riskySum<<",\"riskyMaxExtraP\":"<<r.riskyMax<<",\"riskyBigMerge\":"<<r.riskyBig<<",\"scoreGained\":"<<r.gained
     <<",\"searchCPU\":"<<r.cpu<<",\"maxMoveCPU\":"<<r.mx<<",\"nodes\":"<<r.nodes<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
