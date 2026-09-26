#pragma once
#include "reference.h"
#include <memory>
#include <random>
namespace v2 {
// Separate search RNG: simulations never consume real-game randomness.
struct MEdge{Projection move;double prior=0,initial=0,total=0;int visits=0;};
struct MNode{Board b;Preview p;Counts c;std::array<MEdge,4>edges;int visits=0;double leaf=0;};
struct MCTS {
 const Learner&model;Reference keys;double budget=.001,deadline=0,temperature=2000,exploration=4000;int simulations=0,maxReached=0,evaluations=0,horizon=8,maxSimulations=100000;std::mt19937 rng;std::vector<std::unique_ptr<MNode>>nodes;std::unordered_map<SearchKey,int,KeyHash>table;
 explicit MCTS(const Learner&m):model(m),keys(m){table.reserve(4096);nodes.reserve(4096);}
 double random(){return (double(rng())+.5)/4294967296.;}
 int node(const Board&b,const Preview&p,Counts c,int depth=0){auto key=keys.key(b,p,c,2);key.params|=uint64_t(depth)<<52;auto it=table.find(key);if(it!=table.end())return it->second;auto n=std::make_unique<MNode>();n->b=b;n->p=p;n->c=c;n->leaf=-1e300;for(int d=0;d<4;d++){auto&e=n->edges[d];e.move=move(b,d);if(e.move.size){e.initial=e.move.reward+model.value(e.move.b,stage(e.move.b));n->leaf=std::max(n->leaf,e.initial);evaluations++;}}if(n->leaf<-1e299)n->leaf=0;double sum=0;for(auto&e:n->edges)if(e.move.size){e.prior=std::exp(std::max(-20.,(e.initial-n->leaf)/temperature));sum+=e.prior;}for(auto&e:n->edges)if(e.move.size)e.prior/=sum;int id=nodes.size();nodes.push_back(std::move(n));table.emplace(key,id);return id;}
 Preview nextPreview(const Board&b,Counts&c){auto list=bonuses(high(b));if(!list.empty()&&random()<1./21){double u=random(),sum=0;for(auto&item:list){sum+=item.second;if(u<sum)return item.first;}return list.back().first;}if(c[0]+c[1]+c[2]==0)c={4,4,4};int total=c[0]+c[1]+c[2];double u=random()*total;int card=2;for(int k=0;k<3;k++){u-=c[k];if(u<0){card=k;break;}}c[card]--;Preview p;p.size=1;p.cards[0]=card+1;p.p[0]=1;return p;}
 double simulate(int id,int depth){MNode&n=*nodes[id];maxReached=std::max(maxReached,depth);if(depth>=horizon)return n.leaf;int chosen=-1;double best=-1e300;for(int d=0;d<4;d++){auto&e=n.edges[d];if(!e.move.size)continue;if(!e.visits){chosen=d;break;}double q=e.total/e.visits+exploration*e.prior*std::sqrt(double(n.visits)+1)/(1+e.visits);if(q>best){best=q;chosen=d;}}if(chosen<0)return 0;auto&e=n.edges[chosen];Board next=e.move.b;int entry=std::min(e.move.size-1,int(random()*e.move.size));double u=random(),sum=0;int card=n.p.cards[n.p.size-1];for(int i=0;i<n.p.size;i++){sum+=n.p.p[i];if(u<sum){card=n.p.cards[i];break;}}next[e.move.entries[entry]]=card;double value=e.move.reward+points(card);if(legal(next)){Counts c=n.c;Preview p=nextPreview(next,c);int before=nodes.size(),child=node(next,p,c,depth+1);maxReached=std::max(maxReached,depth+1);value+=child>=before?nodes[child]->leaf:simulate(child,depth+1);}e.visits++;e.total+=value;n.visits++;return value;}
 Answer choose(const Board&b,const Preview&p,Counts c,uint32_t seed){deadline=cpuNow()+budget;table.clear();nodes.clear();simulations=maxReached=evaluations=0;rng.seed(seed);int root=node(b,p,c);do{simulate(root,0);simulations++;}while((simulations<4||cpuNow()<deadline)&&simulations<maxSimulations);Answer a;int most=-1;for(int d=0;d<4;d++){auto&e=nodes[root]->edges[d];if(e.move.size&&e.visits>most){most=e.visits;a={e.visits?e.total/e.visits:e.initial,d};}}return a;}
};
}
