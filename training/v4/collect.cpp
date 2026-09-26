#include "../v3/search.h"
using namespace v2;
int main(int argc,char**argv){init();Model model;model.load(argv[1]);Search search(model);int first=std::stoi(argv[2]),n=std::stoi(argv[3]);Pool pool;std::ofstream provenance(std::string(argv[4])+".sources.jsonl");
for(int i=0;i<n;i++){RNG rng{uint32_t(first+i)};Game g;g.reset(rng);Counts c=opening(g);int last=-100,kept=0;
while(!g.over&&high(g.b)<14&&g.turns<6000){if(high(g.b)==13&&g.turns-last>=40&&kept<8){pool.states.push_back({g,c});provenance<<"{\"index\":"<<pool.states.size()-1<<",\"seed\":"<<first+i<<",\"turn\":"<<g.turns<<"}\n";last=g.turns;kept++;}auto a=search.choose(g.b,g.next,c);g.advance(move(g.b,a.direction),rng);c=observe(c,g.next);}
std::cout<<"collected "<<i+1<<" states="<<pool.states.size()<<std::endl;}
savePool(pool,argv[4]);}
