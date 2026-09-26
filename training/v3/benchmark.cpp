#include "search.h"
using namespace v2;
int main(int argc,char**argv){try{
 if(argc<7)throw std::runtime_error("score-model goal-model-or-dash start-seed games output probe(0/1)");
 init();Model model;model.load(argv[1]);GoalModel gm;GoalModel*goal=nullptr;if(std::string(argv[2])!="-"){gm.load(argv[2]);goal=&gm;}
 uint32_t first=std::stoul(argv[3]);int games=std::stoi(argv[4]);std::ofstream out(argv[5]);bool probe=std::stoi(argv[6]);Search search(model,goal);out<<std::setprecision(17);
 for(int i=0;i<games;i++){
  uint32_t seed=first+i;RNG rng{seed};Game g;g.reset(rng);Counts c=opening(g);auto start=std::chrono::steady_clock::now();uint64_t nodes=0;int depths[4]{};
  while(!g.over&&high(g.b)<14&&g.turns<6000){
   auto a=search.choose(g.b,g.next,c);if(a.direction<0)throw std::runtime_error("invalid search move");nodes+=search.nodes;depths[search.completed]++;
   if(probe&&(g.turns%47==0||high(g.b)>=12)){
    out<<"{\"seed\":"<<seed<<",\"turn\":"<<g.turns<<",\"ranks\":[";for(int j=0;j<16;j++)out<<(j?",":"")<<int(g.b[j]);out<<"],\"cards\":[";for(int j=0;j<g.next.size;j++)out<<(j?",":"")<<int(g.next.cards[j]);out<<"],\"weights\":[";for(int j=0;j<g.next.size;j++)out<<(j?",":"")<<g.next.p[j];out<<"],\"counts\":["<<c[0]<<','<<c[1]<<','<<c[2]<<"],\"direction\":"<<a.direction<<",\"depth\":"<<search.completed<<",\"nodes\":"<<search.nodes<<",\"value\":"<<a.value<<"}\n";
   }
   g.advance(move(g.b,a.direction),rng);c=observe(c,g.next);
  }
  if(!probe)out<<"{\"seed\":"<<seed<<",\"success\":"<<(high(g.b)>=14?"true":"false")<<",\"terminal\":"<<(g.over?"true":"false")<<",\"truncated\":"<<(!g.over&&high(g.b)<14?"true":"false")<<",\"maxRank\":"<<high(g.b)<<",\"scoreAtStop\":"<<score(g.b)<<",\"moves\":"<<g.turns<<",\"nodes\":"<<nodes<<",\"depthCounts\":["<<depths[1]<<','<<depths[2]<<','<<depths[3]<<"],\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
  out.flush();std::cout<<"completed "<<seed<<" success="<<(high(g.b)>=14)<<" moves="<<g.turns<<std::endl;
 }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
