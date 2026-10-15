#pragma once
// V38 model: the V31 model (model5s.h) plus an optional stage-3 "progress" bias (V37 screen key
// "mass"): one weight per value of massKey(b) = min(40, sum over cards >= 192, one 6144 removed,
// of value/192). value(b,3) = ModelS value + bias[massKey(b)]; other stages are unchanged.
// Bias file: header {0x3542444e, 41, 0, 0} then 41 float32. BIAS argument "-" = no bias,
// "zero" = all-zero bias (values equal "-" exactly).
#include "../v31-stages/model5s.h"
namespace v38 {
using namespace v31;
constexpr int MASS=41;
inline int massKey(const Board&b){int u=0;bool skipped=false;
 for(int x:b){if(x==14&&!skipped){skipped=true;continue;}if(x>=9)u+=1<<(x-9);}return std::min(40,u);}
struct ModelP{
 ModelS s;std::array<float,MASS>bias{};bool hasBias=false;
 void load(const std::string&path,const std::string&list,const std::string&biasPath){s.load(path,list);bias.fill(0);hasBias=false;
  if(biasPath=="-")return;hasBias=true;if(biasPath=="zero")return;
  std::ifstream f(biasPath,std::ios::binary);uint32_t h[4];f.read((char*)h,16);
  if(!f||h[0]!=0x3542444e||h[1]!=MASS)throw std::runtime_error("invalid bias "+biasPath);
  f.read((char*)bias.data(),MASS*4);if(!f)throw std::runtime_error("truncated bias "+biasPath);}
 void saveBias(const std::string&path)const{std::ofstream f(path,std::ios::binary);uint32_t h[4]={0x3542444e,MASS,0,0};
  f.write((char*)h,16);f.write((const char*)bias.data(),MASS*4);if(!f)throw std::runtime_error("bias write failed");}
 double value(const Board&b,int st)const{double v=s.value(b,st);if(hasBias&&st==3)v+=bias[massKey(b)];return v;}
 double value(const Board&b)const{return value(b,stage4(b));}
 // TD step on stage st: as ModelS::update with the bias as one more feature (multiplicity 1).
 // tables=false keeps the four-cell table and residual fixed and moves only the bias.
 void update(const Board&b,int st,double target,double alpha,bool tables){
  const bool useBias=hasBias&&st==3;
  if(!useBias){if(tables)s.update(b,st,target,alpha);return;}
  if(!s.has[st])throw std::runtime_error("stage 3 needs a residual");
  std::array<int,FEATURES>ids;for(int f=0;f<FEATURES;f++){const int*m=maps[f];ids[f]=(f>>3)*TABLE+(b[m[0]]|b[m[1]]<<4|b[m[2]]<<8|b[m[3]]<<12);}
  std::array<int,FEATURES5>ids5;for(int f=0;f<FEATURES5;f++)ids5[f]=(f>>3)*TABLE5+ModelS::key5(b,maps5[f]);
  int k=massKey(b);double v=0;for(int i:ids)v+=s.base.w[st][i];for(int i:ids5)v+=s.res[st][i];v+=bias[k];
  auto mult=[](auto sorted){std::sort(sorted.begin(),sorted.end());int n=0;for(size_t i=0;i<sorted.size();){size_t j=i+1;while(j<sorted.size()&&sorted[j]==sorted[i])j++;n+=int((j-i)*(j-i));i=j;}return n;};
  int norm=mult(ids)+mult(ids5)+1;
  float d=float(alpha*(target-v)/norm);bias[k]+=d;
  if(tables){for(int i:ids)s.base.w[st][i]+=d;for(int i:ids5)s.res[st][i]+=d;}}
};
}
