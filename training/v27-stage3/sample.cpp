// V27 on-policy stage-3 training starts. Continues first-6144 snapshots (V26 pool.cpp
// line format) with the website default search (MODEL, depth 5, cutoff 0.01, leaf clamp 0)
// until 12288 or death, and appends the state every EVERY moves (from move 0) while the
// board is alive and its maximum card is exactly 6144.
//  sample MODEL POOL FIRST COUNT ROLLOUT_BASE EVERY OUT_STATES OUT_LOG
// The unsent ordinary deck is reshuffled with ROLLOUT_BASE+index. OUT_LOG: one JSON line
// per source snapshot (moves, states written, outcome) for bookkeeping.
#include "../v21-late-train/search.h"
#include <climits>
using namespace v21;
struct Snap{uint32_t seed=0;Game g;Counts c{};};
static void write(std::ostream&o,uint32_t seed,const Game&g,Counts c){
 o<<std::setprecision(17)<<seed<<' '<<g.turns<<' '<<g.left;for(int i=0;i<g.left;i++)o<<' '<<int(g.deck[i]);
 o<<' '<<g.next.size<<' '<<int(g.next.bonus);for(int i=0;i<3;i++)o<<' '<<int(g.next.cards[i])<<' '<<g.next.p[i];
 for(auto v:c)o<<' '<<v;for(auto r:g.b)o<<' '<<int(r);o<<'\n';}
static std::vector<Snap> readPool(const std::string&path){std::ifstream f(path);if(!f)throw std::runtime_error("no pool "+path);std::vector<Snap>out;std::string line;
 while(std::getline(f,line)){std::istringstream in(line);Snap s;int n;in>>s.seed>>s.g.turns>>s.g.left;
  if(s.g.left<0||s.g.left>12)throw std::runtime_error("bad deck");for(int i=0;i<s.g.left;i++){in>>n;s.g.deck[i]=n;}
  in>>s.g.next.size>>n;s.g.next.bonus=n;for(int i=0;i<3;i++){in>>n>>s.g.next.p[i];s.g.next.cards[i]=n;}
  for(auto&v:s.c)in>>v;for(auto&r:s.g.b){in>>n;r=n;}if(!in)throw std::runtime_error("bad pool line");
  Counts d{0,0,0};for(int i=0;i<s.g.left;i++)d[s.g.deck[i]-1]++;
  if(d!=s.c&&!(s.c==Counts{0,0,0}&&s.g.left==0))throw std::runtime_error("counts do not match unseen deck");
  out.push_back(s);}return out;}
int main(int argc,char**argv){try{
 if(argc!=10||std::string(argv[1])!="sample")throw std::runtime_error("sample MODEL POOL FIRST COUNT ROLLOUT_BASE EVERY OUT_STATES OUT_LOG");
 init();Model4 m;m.load(argv[2]);auto pool=readPool(argv[3]);int first=std::stoi(argv[4]),count=std::stoi(argv[5]);
 uint32_t base=std::stoul(argv[6]);int every=std::stoi(argv[7]);std::ofstream states(argv[8],std::ios::app),log(argv[9],std::ios::app);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.01;s.clampLeaf=true;
 for(int idx=first;idx<first+count&&idx<int(pool.size());idx++){
  Game g=pool[idx].g;Counts c=pool[idx].c;RNG rng{base+uint32_t(idx)};for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
  int moves=0,written=0;
  while(!g.over&&high(g.b)<FINAL_RANK&&moves<30000){
   if(moves%every==0&&high(g.b)==14){write(states,pool[idx].seed,g,c);written++;}
   auto a=s.choose(g.b,g.next,c,5);if(s.completed!=5)throw std::runtime_error("incomplete depth");
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);moves++;if(!g.over)c=observe(c,g.next);}
  if(!g.over&&high(g.b)<FINAL_RANK)throw std::runtime_error("move guard");
  states.flush();
  log<<"{\"index\":"<<idx<<",\"sourceSeed\":"<<pool[idx].seed<<",\"moves\":"<<moves<<",\"written\":"<<written<<",\"reached12288\":"<<(high(g.b)>=FINAL_RANK?"true":"false")<<"}"<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
