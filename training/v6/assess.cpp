#include "search.h"
#include "pool.h"
using namespace v2;
int main(int argc,char**argv){try{
 if(argc!=6)throw std::runtime_error("BASE POOL FIRST COUNT OUTPUT");init();Learner model;model.load(argv[1]);Search search(model);Pool p=loadEarlyPool(argv[2]);int first=std::stoi(argv[3]),count=std::stoi(argv[4]);std::ofstream out(argv[5]);
 for(int i=first;i<first+count;i++){auto s=p.states.at(i);int total=0,wins=0,byAction[4]={-1,-1,-1,-1};int recommended=search.choose(s.game.b,s.game.next,s.counts).direction;
  for(int d=0;d<4;d++){auto initial=move(s.game.b,d);if(!initial.size)continue;byAction[d]=0;for(int k=0;k<4;k++){RNG rng{uint32_t(2026100000u+i*128+d*16+k)};Game g=s.game;Counts c=s.counts;for(int j=g.left-1;j>0;j--)std::swap(g.deck[j],g.deck[rng.index(j+1)]);g.advance(initial,rng);c=observe(c,g.next);int local=0;
    while(!g.over&&high(g.b)<12&&local<6000){auto a=search.choose(g.b,g.next,c);g.advance(move(g.b,a.direction),rng);c=observe(c,g.next);local++;}if(!g.over&&high(g.b)<12)throw std::runtime_error("assessment truncated");int win=high(g.b)>=12;byAction[d]+=win;wins+=win;total++;
   }}out<<"{\"index\":"<<i<<",\"trials\":"<<total<<",\"wins\":"<<wins<<",\"recommended\":"<<recommended<<",\"byAction\":["<<byAction[0]<<','<<byAction[1]<<','<<byAction[2]<<','<<byAction[3]<<"]}\n";out.flush();
 }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
