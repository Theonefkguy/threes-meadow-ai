// Snapshot pool and continuation tool (reusable).
//   pool FIRST_SEED GAMES OUT1536 OUT3072
//       Normal openings played with complete 3-ply expectimax (threshold 0). The full
//       game state at the first 1536 and first 3072 is appended (one line each).
//   cont DEPTH THRESHOLD POOL FIRST_INDEX COUNT ROLLOUT_BASE TARGET_RANK OUT
//       Continue snapshots [FIRST_INDEX, FIRST_INDEX+COUNT) with v17 search at a fixed
//       complete depth until max rank >= TARGET_RANK or real death. Before continuing,
//       the unseen ordinary deck is reshuffled with RNG seed ROLLOUT_BASE+index (search
//       never reads it; the public remaining counts are unchanged). Every arm uses the
//       same rollout seed per snapshot, so arms are paired.
// Line format: seed turn left deck[left] | next.size bonus (card p)x3 | counts[3] | board[16]
#include "../v17-fast-depth/search.h"
#include <climits>
#include <ctime>
using namespace v2;
struct Snap{uint32_t seed=0;Game g;Counts c{};};
static void write(std::ostream&o,uint32_t seed,const Game&g,Counts c){
 o<<std::setprecision(17)<<seed<<' '<<g.turns<<' '<<g.left;for(int i=0;i<g.left;i++)o<<' '<<int(g.deck[i]);
 o<<' '<<g.next.size<<' '<<int(g.next.bonus);for(int i=0;i<3;i++)o<<' '<<int(g.next.cards[i])<<' '<<g.next.p[i];
 for(auto v:c)o<<' '<<v;for(auto r:g.b)o<<' '<<int(r);o<<'\n';}
static std::vector<Snap> readPool(const std::string&path){std::ifstream f(path);std::vector<Snap>out;std::string line;
 while(std::getline(f,line)){std::istringstream in(line);Snap s;int n;in>>s.seed>>s.g.turns>>s.g.left;
  if(s.g.left<0||s.g.left>12)throw std::runtime_error("bad deck");for(int i=0;i<s.g.left;i++){in>>n;s.g.deck[i]=n;}
  in>>s.g.next.size>>n;s.g.next.bonus=n;for(int i=0;i<3;i++){in>>n>>s.g.next.p[i];s.g.next.cards[i]=n;}
  for(auto&v:s.c)in>>v;for(auto&r:s.g.b){in>>n;r=n;}if(!in)throw std::runtime_error("bad pool line");
  // Public counts must describe exactly the unseen deck (bonus previews never consume it).
  Counts d{0,0,0};for(int i=0;i<s.g.left;i++)d[s.g.deck[i]-1]++;
  Counts pub=s.c; // the visible ordinary preview was already drawn from the deck and removed from counts
  if(d!=pub&&!(pub==Counts{0,0,0}&&s.g.left==0))throw std::runtime_error("counts do not match unseen deck");
  out.push_back(s);}return out;}
int main(int argc,char**argv){try{
 init();Learner m;m.load("training/v7/base.ntd");std::string mode=argv[1];
 if(mode=="pool"){uint32_t first=std::stoul(argv[2]);int games=std::stoi(argv[3]);std::ofstream o1(argv[4],std::ios::app),o2(argv[5],std::ios::app);
  v17::Search s(m);s.maxNodes=INT_MAX;s.threshold=0;
  for(int k=0;k<games;k++){uint32_t seed=first+k;RNG rng{seed};Game g;g.reset(rng);Counts c=opening(g);bool w1=false;
   while(!g.over&&high(g.b)<13&&g.turns<20000){auto a=s.choose(g.b,g.next,c,3);if(s.completed!=3)throw std::runtime_error("incomplete");
    auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal");g.advance(p,rng);c=observe(c,g.next);
    if(g.over)break;
    if(!w1&&high(g.b)>=12){write(o1,seed,g,c);w1=true;}
    if(high(g.b)>=13){write(o2,seed,g,c);o1.flush();o2.flush();}}}
  return 0;}
 if(mode=="cont"){int depth=std::stoi(argv[2]);double thr=std::stod(argv[3]);auto pool=readPool(argv[4]);int first=std::stoi(argv[5]),count=std::stoi(argv[6]);
  uint32_t base=std::stoul(argv[7]);int target=std::stoi(argv[8]);std::ofstream out(argv[9],std::ios::app);out<<std::setprecision(17);
  v17::Search s(m);s.maxNodes=INT_MAX;s.threshold=thr;
  for(int idx=first;idx<first+count;idx++){if(idx>=int(pool.size()))throw std::runtime_error("index out of pool");
   Game g=pool[idx].g;Counts c=pool[idx].c;int startRank=high(g.b),startTurn=g.turns;RNG rng{base+uint32_t(idx)};
   for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
   double cpu=0,mx=0;int moves=0;uint64_t nodes=0;
   while(!g.over&&high(g.b)<target&&moves<20000){auto t0=std::clock();auto a=s.choose(g.b,g.next,c,depth);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;
    if(s.completed!=depth)throw std::runtime_error("incomplete depth");cpu+=el;mx=std::max(mx,el);nodes+=s.nodes;moves++;
    auto p=move(g.b,a.direction);if(!p.size)throw std::runtime_error("illegal action");g.advance(p,rng);c=observe(c,g.next);}
   if(!g.over&&high(g.b)<target)throw std::runtime_error("truncated");
   out<<"{\"index\":"<<idx<<",\"sourceSeed\":"<<pool[idx].seed<<",\"depth\":"<<depth<<",\"threshold\":"<<thr<<",\"startRank\":"<<startRank<<",\"startTurn\":"<<startTurn
      <<",\"success\":"<<(high(g.b)>=target?"true":"false")<<",\"finalRank\":"<<high(g.b)<<",\"moves\":"<<moves<<",\"finalScore\":"<<v2::score(g.b)
      <<",\"searchCPU\":"<<cpu<<",\"maxMoveCPU\":"<<mx<<",\"nodes\":"<<nodes<<"}"<<std::endl;}
  return 0;}
 throw std::runtime_error("mode");
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
