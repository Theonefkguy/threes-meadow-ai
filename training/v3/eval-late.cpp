#include "search.h"
using namespace v2;
int main(int argc,char**argv){try{init();Model model;model.load(argv[1]);GoalModel gm;GoalModel*goal=nullptr;if(std::string(argv[2])!="-"){gm.load(argv[2]);goal=&gm;}Pool pool=loadPool(argv[3]);Search policy(model,goal);std::ofstream out(argv[4]);int first=std::stoi(argv[5]),last=std::stoi(argv[6]);
for(int i=first;i<last&&i<int(pool.states.size());i++)for(int replicate=0;replicate<5;replicate++){auto s=pool.states[i];Game g=s.game;Counts c=s.counts;RNG rng{uint32_t(5910001+i*5+replicate)};for(int k=g.left-1;k>0;k--)std::swap(g.deck[k],g.deck[rng.index(k+1)]);int moves=0;
while(!g.over&&high(g.b)<14&&moves<6000){auto a=policy.choose(g.b,g.next,c);g.advance(move(g.b,a.direction),rng);c=observe(c,g.next);moves++;}out<<"{\"state\":"<<i<<",\"replicate\":"<<replicate<<",\"success\":"<<(high(g.b)>=14?"true":"false")<<",\"moves\":"<<moves<<",\"truncated\":"<<(!g.over&&high(g.b)<14?"true":"false")<<"}\n";out.flush();}
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
