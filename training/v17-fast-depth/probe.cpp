// states FIRST_SEED GAMES STRIDE OUT   : sample decision states from exact-3ply games (original search)
// probe STATES DEPTH THRESH OUT        : search every state with v17 (INT_MAX nodes), write per-state results
// parity STATES DEPTH                  : v17 threshold=0 must equal original v7 search (direction + values)
#include "search.h"
#include "../v7/search.h"
#include <climits>
#include <ctime>
#include <cstdio>
using namespace v2;
struct State{Board b;Preview p;Counts c;int turn;};
static void write(std::ostream&o,const State&s){o<<std::setprecision(17);for(auto r:s.b)o<<int(r)<<' ';o<<s.p.size<<' '<<int(s.p.bonus)<<' ';for(int i=0;i<3;i++)o<<int(s.p.cards[i])<<' '<<s.p.p[i]<<' ';for(auto v:s.c)o<<v<<' ';o<<s.turn<<'\n';}
static std::vector<State> readStates(const std::string&path){std::ifstream f(path);std::vector<State>out;State s;int n;
 while(f>>n){s.b[0]=n;for(int i=1;i<16;i++){f>>n;s.b[i]=n;}f>>s.p.size>>n;s.p.bonus=n;for(int i=0;i<3;i++){f>>n>>s.p.p[i];s.p.cards[i]=n;}for(auto&v:s.c)f>>v;f>>s.turn;out.push_back(s);}return out;}
int main(int argc,char**argv){try{
 init();Learner m;m.load("training/v7/base.ntd");std::string mode=argv[1];
 if(mode=="states"){uint32_t first=std::stoul(argv[2]);int games=std::stoi(argv[3]),stride=std::stoi(argv[4]);std::ofstream out(argv[5]);
  v2::Search s(m);s.maxNodes=INT_MAX;
  for(int gi=0;gi<games;gi++){RNG rng{first+uint32_t(gi)};Game g;g.reset(rng);Counts c=opening(g);RNG pick{0x51ed2701u+uint32_t(gi)};int offset=pick.index(stride);
   while(!g.over&&g.turns<6000){if(g.turns%stride==offset)write(out,{g.b,g.next,c,g.turns});auto a=s.choose(g.b,g.next,c,3);auto p=move(g.b,a.direction);g.advance(p,rng);c=observe(c,g.next);if(high(g.b)>=13)break;}}
  return 0;}
 if(mode=="lstates"){uint32_t first=std::stoul(argv[2]);int games=std::stoi(argv[3]),stride=std::stoi(argv[4]);std::ofstream out(argv[5]);
  v2::Search s(m);s.maxNodes=INT_MAX;
  for(int gi=0;gi<games;gi++){RNG rng{first+uint32_t(gi)};Game g;g.reset(rng);Counts c=opening(g);int k=0;
   while(!g.over&&high(g.b)<14&&g.turns<20000){if(high(g.b)>=12&&(k++%stride)==0)write(out,{g.b,g.next,c,g.turns});auto a=s.choose(g.b,g.next,c,3);auto p=move(g.b,a.direction);g.advance(p,rng);c=observe(c,g.next);}}
  return 0;}
 auto states=readStates(argv[2]);int depth=std::stoi(argv[3]);
 if(mode=="parity"){v2::Search a(m);a.maxNodes=INT_MAX;v17::Search b(m);b.maxNodes=INT_MAX;int bad=0;
  for(auto&s:states){auto x=a.choose(s.b,s.p,s.c,depth);auto y=b.choose(s.b,s.p,s.c,depth);
   if(x.direction!=y.direction||x.value!=y.value||a.rootValues!=b.rootValues){bad++;if(bad<5){std::cout<<std::setprecision(17)<<x.direction<<" "<<y.direction<<" "<<x.value<<" "<<y.value;for(int d=0;d<4;d++)std::cout<<" ["<<a.rootValues[d]<<" "<<b.rootValues[d]<<"]";std::cout<<"\n";}}}
  std::cout<<"parity states="<<states.size()<<" depth="<<depth<<" mismatches="<<bad<<"\n";return bad!=0;}
 if(mode=="orig"){v2::Search s(m);s.maxNodes=INT_MAX;double tot=0,mx=0;for(auto&st:states){auto t0=std::clock();s.choose(st.b,st.p,st.c,depth);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;tot+=el;mx=std::max(mx,el);}
  std::cout<<"orig depth="<<depth<<" meanMs="<<1000*tot/states.size()<<" maxMs="<<1000*mx<<"\n";return 0;}
 double thresh=std::stod(argv[4]);std::ofstream out(argv[5]);out<<std::setprecision(17);
 v17::Search s(m,getenv("BITS")?atoi(getenv("BITS")):20);s.maxNodes=INT_MAX;s.threshold=thresh;
 for(size_t i=0;i<states.size();i++){auto&st=states[i];auto t0=std::clock();auto a=s.choose(st.b,st.p,st.c,depth);double el=double(std::clock()-t0)/CLOCKS_PER_SEC;
  if(s.completed!=depth)throw std::runtime_error("incomplete");
  out<<"{\"i\":"<<i<<",\"high\":"<<high(st.b)<<",\"dir\":"<<a.direction<<",\"sec\":"<<el<<",\"nodes\":"<<s.nodes<<",\"cut\":"<<s.cutoffs<<",\"q\":[";
  for(int d=0;d<4;d++)out<<(d?",":"")<<(s.rootValues[d]<-1e299?-1:s.rootValues[d]);out<<"]}\n";}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
