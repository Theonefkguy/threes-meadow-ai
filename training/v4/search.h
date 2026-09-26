#pragma once
#include "model.h"
#include <cstring>
namespace v2 {
struct SearchKey{
 uint64_t board=0,params=0;std::array<uint64_t,3>weights{};
 bool operator==(const SearchKey&k)const{return board==k.board&&params==k.params&&weights==k.weights;}
};
struct KeyHash{size_t operator()(const SearchKey&k)const{uint64_t x=k.board^(k.params*0x9e3779b97f4a7c15ULL);for(auto w:k.weights)x^=w+0x9e3779b97f4a7c15ULL+(x<<6)+(x>>2);return x^(x>>31);}};
struct Answer{double value=0;int direction=-1;};
struct Search{
 const Learner&scoreModel;int maxNodes=24000,nodes=0,completed=0;std::unordered_map<SearchKey,Answer,KeyHash>cache;
 explicit Search(const Learner&m):scoreModel(m){cache.reserve(4096);}
 void tick(){nodes++;if(completed&&nodes>maxNodes)throw 1;}
 SearchKey key(const Board&b,const Preview&p,Counts c,int depth){SearchKey k;for(int i=0;i<16;i++)k.board|=uint64_t(b[i])<<(i*4);if(depth==1){k.params=1;return k;}k.params=depth|(p.size<<3)|(c[0]<<5)|(c[1]<<8)|(c[2]<<11);for(int i=0;i<3&&i<p.size;i++){k.params|=uint64_t(p.cards[i])<<(14+i*4);std::memcpy(&k.weights[i],&p.p[i],8);}return k;}
 double future(const Board&b,Counts counts,int depth){if(!legal(b))return 0;return hints(b,counts,[&](const Preview&p,Counts c){return decision(b,p,c,depth).value;});}
 Answer decision(const Board&b,const Preview&p,Counts counts,int depth){
  tick();auto k=key(b,p,counts,depth);auto found=cache.find(k);if(found!=cache.end())return found->second;
  double best=-1e300;int direction=-1;
  for(int d=0;d<4;d++){
   auto m=move(b,d);if(!m.size)continue;double q=m.reward;
   if(depth==1)q+=scoreModel.value(m.b,stage(m.b));
   else{double expected=0;for(int e=0;e<m.size;e++)for(int i=0;i<p.size;i++){
    tick();Board next=m.b;next[m.entries[e]]=p.cards[i];expected+=p.p[i]*(points(p.cards[i])+future(next,counts,depth-1));
   }q+=expected/m.size;}
   if(q>best){best=q;direction=d;}
  }
  Answer a{direction<0?0:best,direction};cache.emplace(k,a);return a;
 }
 Answer choose(const Board&b,const Preview&p,Counts c,int maxDepth=3){
  cache.clear();nodes=0;completed=0;Answer result;
  for(int depth=1;depth<=maxDepth;depth++){try{result=decision(b,p,c,depth);completed=depth;}catch(int){break;}}
  return result;
 }
};
}
