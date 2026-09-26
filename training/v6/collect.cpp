#include "search.h"
#include "pool.h"
#include <set>
using namespace v2;
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("BASE FIRST COUNT PREFIX");init();Learner m;m.load(argv[1]);Search search(m);uint32_t first=std::stoul(argv[2]);int count=std::stoi(argv[3]);std::string out=argv[4];Pool regular,candidates;std::ofstream sources(out+".sources.jsonl"),teacher(out+".teacher.txt"),games(out+".games.jsonl");
 for(int n=0;n<count;n++){uint32_t seed=first+n;RNG rng{seed};Game g;g.reset(rng);Counts c=opening(g);std::vector<Snapshot>history;std::vector<Board>examples;int first768=-1;
  while(!g.over&&high(g.b)<12&&g.turns<6000){history.push_back({g,c});if(high(g.b)==11&&first768<0)first768=history.size()-1;auto a=search.choose(g.b,g.next,c);auto moveResult=move(g.b,a.direction);if(!moveResult.size)throw std::runtime_error("illegal teacher action");if(high(moveResult.b)<12&&(g.turns%20==0||high(g.b)==11&&g.turns%5==0))examples.push_back(moveResult.b);g.advance(moveResult,rng);c=observe(c,g.next);}
  bool success=high(g.b)>=12;if(!success&&!g.over)throw std::runtime_error("truncated collection");
  if(first768>=0){regular.states.push_back(history[first768]);sources<<"{\"type\":\"regular\",\"index\":"<<regular.states.size()-1<<",\"seed\":"<<seed<<",\"turn\":"<<history[first768].game.turns<<"}\n";}
  if(!success){std::set<int>indices;for(int back:{20,50,100})indices.insert(std::max(0,int(history.size())-back));if(first768>=0)indices.insert(first768);for(int i:indices){candidates.states.push_back(history[i]);sources<<"{\"type\":\"candidate\",\"index\":"<<candidates.states.size()-1<<",\"seed\":"<<seed<<",\"turn\":"<<history[i].game.turns<<"}\n";}}
  for(auto&b:examples){teacher<<int(success);for(auto r:b)teacher<<' '<<int(r);teacher<<'\n';}games<<"{\"seed\":"<<seed<<",\"success1536\":"<<(success?"true":"false")<<",\"moves\":"<<g.turns<<"}\n";
 }
 saveEarlyPool(regular,out+".regular.txt");saveEarlyPool(candidates,out+".candidates.txt");std::cout<<"collected "<<count<<" games, "<<candidates.states.size()<<" candidates\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
