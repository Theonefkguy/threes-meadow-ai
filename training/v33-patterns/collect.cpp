// V33 data: stage-3 afterstates with their realised future score under the website search.
//  collect MODEL RES_LIST POOL FIRST COUNT ROLLOUT_BASE EVERY OUT_BIN
// Plays each pool state (unsent deck reshuffled with ROLLOUT_BASE+index) with the website
// search (depth 5, cutoff 0.01, clamped leaves, complete depth) to 12288 or death. Afterstate k
// gets G_k = sum of rewards after it (points of the card that then enters + merge points of the
// next move, ..., the 12288 merge included), the quantity the stage-3 table estimates. Every
// EVERY-th afterstate (from the first) is written as a 32-byte record:
//   uint8 board[16], double G, uint32 episode index, uint32 move number.
#include "../v31-stages/search5s.h"
#include <climits>
using namespace v31;
struct Snap{uint32_t seed=0;Game g;Counts c{};};
static std::vector<Snap> readPool(const std::string&path){std::ifstream f(path);if(!f)throw std::runtime_error("no pool");std::vector<Snap>out;std::string line;
 while(std::getline(f,line)){std::istringstream in(line);Snap s;int n;in>>s.seed>>s.g.turns>>s.g.left;for(int i=0;i<s.g.left;i++){in>>n;s.g.deck[i]=n;}
  in>>s.g.next.size>>n;s.g.next.bonus=n;for(int i=0;i<3;i++){in>>n>>s.g.next.p[i];s.g.next.cards[i]=n;}for(auto&v:s.c)in>>v;for(auto&r:s.g.b){in>>n;r=n;}
  if(!in)throw std::runtime_error("bad pool line");out.push_back(s);}return out;}
#pragma pack(push,1)
struct Record{uint8_t b[16];double G;uint32_t episode,move;};
#pragma pack(pop)
static_assert(sizeof(Record)==32,"record size");
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("collect MODEL RES_LIST POOL FIRST COUNT ROLLOUT_BASE EVERY OUT_BIN");
 v31::init();ModelS m;m.load(argv[1],argv[2]);auto pool=readPool(argv[3]);int first=std::stoi(argv[4]),count=std::stoi(argv[5]);
 uint32_t base=std::stoul(argv[6]);int every=std::stoi(argv[7]);std::ofstream out(argv[8],std::ios::binary|std::ios::app);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.01;s.clampLeaf=true;
 for(int idx=first;idx<first+count&&idx<int(pool.size());idx++){
  Game g=pool[idx].g;Counts c=pool[idx].c;RNG rng{base+uint32_t(idx)};for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
  if(high(g.b)!=14)throw std::runtime_error("not a 6144-phase state");
  std::vector<Board>after;std::vector<double>reward;int moves=0;
  while(!g.over&&moves<30000){auto a=s.choose(g.b,g.next,c,5);if(s.completed!=5)throw std::runtime_error("incomplete depth");
   auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");
   if(!reward.empty())reward.back()+=p.reward;moves++;
   if(high(p.b)>=FINAL_RANK){g.advance(p,rng);break;}
   after.push_back(p.b);reward.push_back(0);
   double spawn=g.advance(p,rng);reward.back()+=spawn;if(!g.over)c=observe(c,g.next);}
  if(!g.over)throw std::runtime_error("truncated");
  double G=0;std::vector<double>ret(after.size());for(size_t k=after.size();k-->0;){G+=reward[k];ret[k]=G;}
  for(size_t k=0;k<after.size();k+=every){Record r;for(int i=0;i<16;i++)r.b[i]=after[k][i];r.G=ret[k];r.episode=uint32_t(idx);r.move=uint32_t(k);out.write((const char*)&r,sizeof r);}
  out.flush();}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
