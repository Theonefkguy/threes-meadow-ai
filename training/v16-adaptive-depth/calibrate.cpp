#include "../v7/search.h"
#include "../v7/pool.h"
#include <climits>
#include <ctime>
using namespace v2;

int emptyCells(const Board&b){int n=0;for(auto v:b)n+=v==0;return n;}
int legalMoves(const Board&b){int n=0;for(int d=0;d<4;d++)n+=move(b,d).size>0;return n;}
double rootMargin(const Search&s){std::vector<double>v;for(double x:s.rootValues)if(x>-1e299)v.push_back(x);std::sort(v.begin(),v.end(),std::greater<double>());return v.size()<2?1:(v[0]-v[1])/std::max(1.,std::abs(v[0]));}

int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("OUT MAX_SAMPLES");init();int limit=std::stoi(argv[2]);Pool pool=loadEarlyPool("training/v15-late-depth/snapshots.txt");Learner model;model.load("training/v7/base.ntd");
 std::ofstream out(argv[1]);out<<std::setprecision(17);int samples=0;
 for(int index=0;index<32&&samples<limit;index++){
  Game g=pool.states[index].game;Counts counts=pool.states[index].counts;RNG rng{33170001u+uint32_t(index)};for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);int local=0;
  while(!g.over&&high(g.b)<13&&local<1200&&samples<limit){
   Search s3(model);s3.maxNodes=INT_MAX;auto a3=s3.choose(g.b,g.next,counts,3);
   if(local%30==0){
    Search s4(model),s5(model);s4.maxNodes=s5.maxNodes=INT_MAX;auto begin=std::clock();auto a4=s4.choose(g.b,g.next,counts,4);double cpu4=double(std::clock()-begin)/CLOCKS_PER_SEC;begin=std::clock();auto a5=s5.choose(g.b,g.next,counts,5);double cpu5=double(std::clock()-begin)/CLOCKS_PER_SEC;
    out<<"{\"index\":"<<index<<",\"move\":"<<local<<",\"rank\":"<<high(g.b)<<",\"empty\":"<<emptyCells(g.b)<<",\"legal\":"<<legalMoves(g.b)<<",\"margin3\":"<<rootMargin(s3)<<",\"d3\":"<<a3.direction<<",\"d4\":"<<a4.direction<<",\"d5\":"<<a5.direction<<",\"cpu4\":"<<cpu4<<",\"cpu5\":"<<cpu5<<"}\n";out.flush();samples++;
   }
   auto m=move(g.b,a3.direction);g.advance(m,rng);counts=observe(counts,g.next);local++;
  }
 }
 std::cerr<<"samples="<<samples<<"\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
