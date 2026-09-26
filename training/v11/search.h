#pragma once
#include "../v8-search/mcts.h"
#include "../v9/net.h"
namespace v2 {
struct RootPolicySearch: MCTS {
 const v9::Net& policy;
 RootPolicySearch(const Learner&m,const v9::Net&p):MCTS(m),policy(p){}
 void applyPrior(int root){
  auto&n=*nodes[root];if(high(n.b)>=12)return;
  std::array<double,4>logits{};double maxLog=-1e300,sum=0;
  for(int d=0;d<4;d++){auto&e=n.edges[d];if(!e.move.size)continue;
   double base=e.initial-e.move.reward;auto f=v9::features(e.move,n.p,n.c,base,policy.context);
   logits[d]=(e.initial-n.leaf)/temperature+policy.forward(f)[1];maxLog=std::max(maxLog,logits[d]);
  }
  for(int d=0;d<4;d++)if(n.edges[d].move.size){n.edges[d].prior=std::exp(std::max(-20.,logits[d]-maxLog));sum+=n.edges[d].prior;}
  for(auto&e:n.edges)if(e.move.size)e.prior/=sum;
 }
 Answer choose(const Board&b,const Preview&p,Counts c,uint32_t seed){
  deadline=cpuNow()+budget;table.clear();nodes.clear();simulations=maxReached=evaluations=0;rng.seed(seed);
  int root=node(b,p,c);applyPrior(root);
  do{simulate(root,0);simulations++;}while((simulations<4||cpuNow()<deadline)&&simulations<maxSimulations);
  Answer a;int most=-1;for(int d=0;d<4;d++){auto&e=nodes[root]->edges[d];if(e.move.size&&e.visits>most){most=e.visits;a={e.visits?e.total/e.visits:e.initial,d};}}return a;
 }
};
}
