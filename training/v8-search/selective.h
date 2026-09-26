#pragma once
#include "reference.h"
namespace v2 {
struct SelectAnswer{double value=0;int direction=-1;std::array<double,4> q{};};
struct Selective {
 const Learner&model;double budget=.001,deadline=0,margin=0;int nodes=0,completed=0,pruned=0;bool adaptive=false;std::unordered_map<SearchKey,SelectAnswer,KeyHash>cache;Reference keys;
 explicit Selective(const Learner&m):model(m),keys(m){cache.reserve(4096);}
 void tick(){++nodes;if(completed&&nodes%32==0&&cpuNow()>=deadline)throw 1;}
 SelectAnswer decision(const Board&b,const Preview&p,Counts c,int depth,bool root=false){
  tick();auto key=keys.key(b,p,c,depth);if(root)key.params|=1ULL<<60;auto found=cache.find(key);if(found!=cache.end())return found->second;
  SelectAnswer a;a.q.fill(-1e300);std::array<bool,4>active{};for(int d=0;d<4;d++)active[d]=move(b,d).size>0;
  // Every legal move gets a full two-ply estimate before selective deepening.
  // This is heuristic forward pruning, not a proof that omitted moves are worse.
  if(depth>=3&&!root){auto shallow=decision(b,p,c,2);std::array<int,4>order{0,1,2,3};std::stable_sort(order.begin(),order.end(),[&](int x,int y){return shallow.q[x]>shallow.q[y];});for(int k=2;k<4;k++){int d=order[k];if(active[d]&&(!adaptive||shallow.q[d]<shallow.value-margin)){active[d]=false;pruned++;}}}
  a.value=-1e300;
  for(int d=0;d<4;d++)if(active[d]){auto m=move(b,d);double q=m.reward;
   if(depth==1)q+=model.value(m.b,stage(m.b));
   else{double expected=0;for(int e=0;e<m.size;e++)for(int i=0;i<p.size;i++){tick();Board next=m.b;next[m.entries[e]]=p.cards[i];double v=0;if(legal(next))v=hints(next,c,[&](const Preview&hint,Counts cc){return decision(next,hint,cc,depth-1).value;});expected+=p.p[i]*(points(p.cards[i])+v);}q+=expected/m.size;}
   a.q[d]=q;if(q>a.value){a.value=q;a.direction=d;}}
  if(a.direction<0)a.value=0;cache.emplace(key,a);return a;
 }
 Answer choose(const Board&b,const Preview&p,Counts c,int maxDepth=8){deadline=cpuNow()+budget;cache.clear();nodes=completed=pruned=0;Answer answer;for(int depth=1;depth<=maxDepth;depth++){try{auto a=decision(b,p,c,depth,true);answer={a.value,a.direction};completed=depth;}catch(int){break;}}return answer;}
};
}
