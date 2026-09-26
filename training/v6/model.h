#pragma once
#include "../v3/goal.h"
#include <ctime>
namespace v2 {
struct Learner {
 Model base;bool probability=false;
 void load(const std::string&path,int mode=0){if(mode!=0&&mode!=1)throw std::runtime_error("invalid mode");probability=mode;base.load(path);}
 void save(const std::string&path)const{base.save(path);}
 double value(const Board&b,int s)const{return base.value(b,s);}
 double prob(const Board&b)const{return high(b)>=12?1:sigmoid(base.value(b,0));}
 void initializeProbability(){probability=true;std::fill(base.w[0].begin(),base.w[0].end(),std::log(9.)/FEATURES);}
 void update(const Board&b,double target,double alpha){
  if(high(b)>=12)throw std::runtime_error("attempt to update frozen stage");
  if(!probability){base.update(b,0,target,alpha);return;}
  if(target<0||target>1||!std::isfinite(target))throw std::runtime_error("invalid probability target");
  auto ids=base.indices(b),sorted=ids;std::sort(sorted.begin(),sorted.end());int norm=0;for(int i=0;i<FEATURES;){int j=i+1;while(j<FEATURES&&sorted[j]==sorted[i])j++;norm+=(j-i)*(j-i);i=j;}
  float delta=alpha*(target-prob(b))/norm;for(int i:ids)base.w[0][i]+=delta;
 }
};
inline Projection policyTwo(const Game&g,const Learner&model){
 if(!model.probability)return scorePolicy(g,model.base);
 double best=-1;Projection choice;for(int d=0;d<4;d++){auto m=move(g.b,d);if(!m.size)continue;double q=0;
  if(high(m.b)>=12)q=1;else{for(int e=0;e<m.size;e++)for(int i=0;i<g.next.size;i++){Board b=m.b;b[m.entries[e]]=g.next.cards[i];double v=0;for(int d2=0;d2<4;d2++){auto n=move(b,d2);if(n.size)v=std::max(v,model.prob(n.b));}q+=g.next.p[i]*v/m.size;}}
  if(q>best){best=q;choice=m;}
 }return choice;
}
}
