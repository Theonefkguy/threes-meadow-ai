#pragma once
#include "common.h"
namespace v2 {
constexpr double GOAL_BONUS=200000;
inline double sigmoid(double x){return x>=0?1/(1+std::exp(-x)):std::exp(x)/(1+std::exp(x));}
struct GoalModel {
 Model base;
 double value(const Board&b,const Preview&,Counts)const{return high(b)>=14?1:high(b)<13?0:sigmoid(base.value(b,2));}
 void seed(const Model&m){for(int i=0;i<WEIGHTS;i++)base.w[2][i]=m.w[2][i]/600000.-2.5/FEATURES;}
 void update(const Board&b,double target,double alpha){
  auto ids=base.indices(b),sorted=ids;std::sort(sorted.begin(),sorted.end());int norm=0;for(int i=0;i<FEATURES;){int j=i+1;while(j<FEATURES&&sorted[j]==sorted[i])j++;norm+=(j-i)*(j-i);i=j;}
  double p=sigmoid(base.value(b,2));float delta=alpha*(target-p)/norm;for(int id:ids)base.w[2][id]+=delta;
 }
 void save(const std::string&p)const{base.save(p);}void load(const std::string&p){base.load(p);}
};
template<class F>inline double hints(const Board&b,Counts counts,F fn){
 auto list=bonuses(high(b));double chance=list.empty()?0:1./21,total=0;Counts deck=counts;if(deck[0]+deck[1]+deck[2]==0)deck={4,4,4};int size=deck[0]+deck[1]+deck[2];
 for(int n=0;n<3;n++)if(deck[n]){Preview p;p.size=1;p.cards[0]=n+1;p.p[0]=1;Counts next=deck;next[n]--;total+=(1-chance)*double(deck[n])/size*fn(p,next);}
 for(auto item:list)total+=chance*item.second*fn(item.first,counts);return total;
}
}
