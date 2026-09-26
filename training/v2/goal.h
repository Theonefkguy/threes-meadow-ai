#pragma once
#include "common.h"
namespace v2 {
constexpr int CONTEXT=131072;
inline double sigmoid(double z){return z>=0?1/(1+std::exp(-z)):std::exp(z)/(1+std::exp(z));}
struct ContextBase{int empty=0,ones=0,twos=0;};
inline ContextBase contextBase(const Board&b){ContextBase x;for(auto r:b){x.empty+=r==0;x.ones+=r==1;x.twos+=r==2;}return x;}
inline std::array<int,2> contextIndices(ContextBase x,const Preview&p,Counts c){
 int hint=(p.size-1)*16+p.cards[0],bag=c[0]+5*c[1]+25*c[2];
 return {(x.empty*48+hint)*125+bag,((x.ones*17+x.twos)*48+hint)*9+c[0]-c[1]+4};
}
struct GoalModel{
 bool contextual=false;Model base;std::array<std::vector<float>,2>extra;
 explicit GoalModel(bool useContext=false):contextual(useContext){if(contextual)for(auto&w:extra)w.assign(2*CONTEXT,0);}
 void seed(const Model&score){for(int s=0;s<2;s++)for(int i=0;i<WEIGHTS;i++)base.w[s][i]=score.w[s][i]/300000.-2./FEATURES;}
 double probability(double logit,int stage,ContextBase b,const Preview&p,Counts counts)const{
  if(contextual){auto ids=contextIndices(b,p,counts);logit+=extra[stage][ids[0]]+extra[stage][CONTEXT+ids[1]];}return sigmoid(logit);
 }
 double value(const Board&b,const Preview&p,Counts c)const{if(high(b)>=13)return 1;int s=stage(b);return probability(base.value(b,s),s,contextBase(b),p,c);}
 void update(const Board&b,const Preview&p,Counts c,double target,double alpha){
  int s=stage(b);auto ids=base.indices(b),sorted=ids;std::sort(sorted.begin(),sorted.end());int norm=contextual?2:0;
  for(int i=0;i<FEATURES;){int j=i+1;while(j<FEATURES&&sorted[j]==sorted[i])j++;norm+=(j-i)*(j-i);i=j;}
  float delta=alpha*(target-value(b,p,c))/norm;for(int id:ids)base.w[s][id]+=delta;
  if(contextual){auto ci=contextIndices(contextBase(b),p,c);extra[s][ci[0]]+=delta;extra[s][CONTEXT+ci[1]]+=delta;}
 }
 void save(const std::string&path)const{
  std::ofstream f(path,std::ios::binary);uint32_t h[4]={0x324c4f47,2,WEIGHTS,uint32_t(contextual?2*CONTEXT:0)};f.write((char*)h,16);
  for(int s=0;s<2;s++){f.write((char*)base.w[s].data(),WEIGHTS*4);if(contextual)f.write((char*)extra[s].data(),2*CONTEXT*4);}if(!f)throw std::runtime_error("goal write failed");
 }
 void load(const std::string&path){
  std::ifstream f(path,std::ios::binary);uint32_t h[4]{};f.read((char*)h,16);if(!f||h[0]!=0x324c4f47||h[1]!=2||h[2]!=WEIGHTS||(h[3]&&h[3]!=2*CONTEXT))throw std::runtime_error("goal format invalid");
  contextual=h[3];for(int s=0;s<2;s++){f.read((char*)base.w[s].data(),WEIGHTS*4);if(contextual){extra[s].resize(2*CONTEXT);f.read((char*)extra[s].data(),2*CONTEXT*4);}}if(!f)throw std::runtime_error("goal truncated");
 }
};
template<class F>inline double hints(const Board&b,Counts counts,F fn){
 auto list=bonuses(high(b));double chance=list.empty()?0:1./21,total=0;Counts deck=counts;if(deck[0]+deck[1]+deck[2]==0)deck={4,4,4};int size=deck[0]+deck[1]+deck[2];
 for(int n=0;n<3;n++)if(deck[n]){Preview p;p.size=1;p.cards[0]=n+1;p.p[0]=1;Counts next=deck;next[n]--;total+=(1-chance)*double(deck[n])/size*fn(p,next);}
 for(auto item:list)total+=chance*item.second*fn(item.first,counts);return total;
}
constexpr double GOAL_BONUS=60000;
// The score head anchors play while the success head learns the rare goal.
// This is a combined utility, not a calibrated success-probability policy.
inline Projection goalPolicy(const Game&g,Counts counts,const GoalModel&model,const Model&scoreModel){
 double best=-1e300;Projection chosen;
 for(int d=0;d<4;d++){
  auto m=move(g.b,d);if(!m.size)continue;double expected=0;
  for(int e=0;e<m.size;e++)for(int c=0;c<g.next.size;c++){
   Board b=m.b;b[m.entries[e]]=g.next.cards[c];std::array<Projection,4>leaves;std::array<double,4>logits{},scores{};std::array<ContextBase,4>context;std::array<int,4>stages{};bool any=false;
   for(int d2=0;d2<4;d2++){
    leaves[d2]=move(b,d2);if(!leaves[d2].size)continue;any=true;stages[d2]=stage(leaves[d2].b);scores[d2]=leaves[d2].reward+scoreModel.value(leaves[d2].b,stages[d2]);
    if(stages[d2]<2){logits[d2]=model.base.value(leaves[d2].b,stages[d2]);context[d2]=contextBase(leaves[d2].b);}
   }
   double v=high(b)>=13?GOAL_BONUS:0;
   if(any){auto evaluate=[&](const Preview&p,Counts cs){double mx=-1e300;for(int d2=0;d2<4;d2++)if(leaves[d2].size){double probability=stages[d2]>=2?1:model.probability(logits[d2],stages[d2],context[d2],p,cs);mx=std::max(mx,scores[d2]+GOAL_BONUS*probability);}return mx;};v=model.contextual?hints(b,counts,evaluate):evaluate(g.next,counts);}
   expected+=g.next.p[c]*(points(g.next.cards[c])+v);
  }
  double q=m.reward+expected/m.size;if(q>best){best=q;chosen=m;}
 }return chosen;
}
}
