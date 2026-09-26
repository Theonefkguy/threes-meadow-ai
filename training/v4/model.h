#pragma once
#include "../v3/goal.h"
#include <ctime>
namespace v2 {
constexpr int FIVE_TABLE=1048576,FIVE_PATTERNS=2,FIVE_FEATURES=16;
constexpr int fivePatterns[2][5]={{0,1,2,3,4},{0,1,2,4,5}};
struct Learner {
 Model base;bool five=false,tc=false;std::vector<float> residual;std::vector<double> errors,absolute;int mapping[16][5];
 Learner(){for(int p=0;p<2;p++)for(int sym=0;sym<8;sym++)for(int j=0;j<5;j++){int y=fivePatterns[p][j]/4,x=fivePatterns[p][j]%4;if(sym>=4)x=3-x;for(int k=0;k<sym%4;k++){int ny=x;x=3-y;y=ny;}mapping[p*8+sym][j]=y*4+x;}}
 void enableFive(){five=true;residual.assign(FIVE_PATTERNS*FIVE_TABLE,0);}
 void enableTC(){tc=true;errors.assign(WEIGHTS,0);absolute.assign(WEIGHTS,0);}
 std::array<int,16> extraIndices(const Board&b)const{std::array<int,16>ids;for(int f=0;f<16;f++){int key=0;for(int j=0;j<5;j++)key|=std::min<int>(15,b[mapping[f][j]])<<(j*4);ids[f]=(f/8)*FIVE_TABLE+key;}return ids;}
 double value(const Board&b,int s)const{double v=base.value(b,s);if(five&&s==2){auto ids=extraIndices(b);for(int i:ids)v+=residual[i];}return v;}
 template<size_t N>static int norm(const std::array<int,N>&ids){auto sorted=ids;std::sort(sorted.begin(),sorted.end());int n=0;for(int i=0;i<int(N);){int j=i+1;while(j<int(N)&&sorted[j]==sorted[i])j++;n+=(j-i)*(j-i);i=j;}return n;}
 void update(const Board&b,double target,double alpha){
  if(five){auto ids=extraIndices(b);float delta=alpha*(target-value(b,2))/norm(ids);for(int id:ids)residual[id]+=delta;return;}
  if(!tc){base.update(b,2,target,alpha);return;}
  auto ids=base.indices(b);double error=target-base.value(b,2),delta=alpha*error/norm(ids);auto sorted=ids;std::sort(sorted.begin(),sorted.end());
  // TC multiplier uses the PREVIOUS signed/absolute error sums. One update per
  // unique parameter, including the actual symmetry multiplicity.
  for(int i=0;i<FEATURES;){int j=i+1;while(j<FEATURES&&sorted[j]==sorted[i])j++;int id=sorted[i],mult=j-i;double eta=absolute[id]>0?std::abs(errors[id])/absolute[id]:1;float change=delta*eta;for(int k=0;k<mult;k++)base.w[2][id]+=change;errors[id]+=mult*error;absolute[id]+=mult*std::abs(error);i=j;}
 }
 void load(const std::string&p,const std::string&extra="-"){base.load(p);if(extra!="-"){enableFive();std::ifstream f(extra,std::ios::binary);uint32_t h[4]{};f.read((char*)h,16);if(!f||h[0]!=0x3544544e||h[1]!=2||h[2]!=FIVE_TABLE||h[3]!=5)throw std::runtime_error("invalid five-table model");f.read((char*)residual.data(),residual.size()*4);if(!f)throw std::runtime_error("truncated five model");}}
 void save(const std::string&p)const{base.save(p+".ntd");if(five){std::ofstream f(p+".five",std::ios::binary);uint32_t h[4]={0x3544544e,2,FIVE_TABLE,5};f.write((char*)h,16);f.write((char*)residual.data(),residual.size()*4);if(!f)throw std::runtime_error("five write failed");}}
};
inline Projection policyTwo(const Game&g,const Learner&model){double best=-1e300;Projection chosen;for(int d=0;d<4;d++){auto m=move(g.b,d);if(!m.size)continue;double expected=0;for(int e=0;e<m.size;e++)for(int c=0;c<g.next.size;c++){Board b=m.b;int card=g.next.cards[c];b[m.entries[e]]=card;double value=-1e300;for(int d2=0;d2<4;d2++){auto n=move(b,d2);if(n.size)value=std::max(value,n.reward+model.value(n.b,stage(n.b)));}expected+=g.next.p[c]*(points(card)+(value==-1e300?0:value));}double q=m.reward+expected/m.size;if(q>best){best=q;chosen=m;}}return chosen;}
}
