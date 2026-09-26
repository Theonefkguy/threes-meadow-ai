#include "search.h"
using namespace v2;
int main(int argc,char**argv){init();Model model;model.load(argv[1]);Search policy(model);Pool pool;int first=std::stoi(argv[2]),n=std::stoi(argv[3]),dead=0;
 for(int i=0;i<n;i++){RNG rng{uint32_t(first+i)};Game g;g.reset(rng);Counts c=opening(g);while(!g.over&&high(g.b)<13&&g.turns<6000){auto a=policy.choose(g.b,g.next,c);g.advance(move(g.b,a.direction),rng);c=observe(c,g.next);}if(high(g.b)>=13){if(g.over)dead++;else pool.states.push_back({g,c});}std::cout<<"collected "<<i+1<<std::endl;}
 savePool(pool,argv[4]);std::ofstream meta(std::string(argv[4])+".json");meta<<"{\"firstSeed\":"<<first<<",\"normalGames\":"<<n<<",\"playable3072\":"<<pool.states.size()<<",\"terminalAt3072\":"<<dead<<"}";
}
