#pragma once
// V21: four-stage N-tuple afterstate model. Same 8 patterns x 8 symmetries as V1-V8.
// Stages by the afterstate's maximum card: 0: <1536, 1: 1536, 2: 3072, 3: >=6144.
// File: header {0x3144544e, stages, PATTERNS, TABLE} then stages x WEIGHTS float32.
// A 3-stage (V1-V8) file loads with stage 3 initialised as a copy of stage 2.
#include "../v7/model.h"
namespace v21 {
using namespace v2;
constexpr int STAGES=4;
inline int stage4(const Board&b){int r=high(b);return r>=14?3:r>=13?2:r>=12?1:0;}
struct Model4{
 std::array<std::vector<float>,STAGES>w;
 Model4(){for(auto&t:w)t.assign(WEIGHTS,0);}
 void load(const std::string&path){std::ifstream f(path,std::ios::binary);uint32_t h[4];f.read((char*)h,16);
  if(!f||h[0]!=0x3144544e||(h[1]!=3&&h[1]!=4)||h[2]!=PATTERNS||h[3]!=TABLE)throw std::runtime_error("invalid model");
  for(uint32_t s=0;s<h[1];s++)f.read((char*)w[s].data(),WEIGHTS*4);if(!f)throw std::runtime_error("truncated model");
  if(h[1]==3)w[3]=w[2];}
 void save(const std::string&path)const{std::ofstream f(path,std::ios::binary);uint32_t h[4]={0x3144544e,STAGES,PATTERNS,TABLE};f.write((char*)h,16);
  for(auto&t:w)f.write((const char*)t.data(),WEIGHTS*4);if(!f)throw std::runtime_error("model write failed");}
 double value(const Board&b,int s)const{const float*t=w[s].data();double v=0;
  for(int f=0;f<FEATURES;f++){const int*m=maps[f];v+=t[(f>>3)*TABLE+(b[m[0]]|b[m[1]]<<4|b[m[2]]<<8|b[m[3]]<<12)];}return v;}
 double value(const Board&b)const{return value(b,stage4(b));}
 // TD update with the squared-multiplicity normalisation used since V1.
 // Optional TC (temporal coherence) per-weight rates when tcE/tcA are provided.
 void update(const Board&b,int s,double target,double alpha,std::vector<float>*tcE=nullptr,std::vector<float>*tcA=nullptr){
  std::array<int,FEATURES>ids;for(int f=0;f<FEATURES;f++){const int*m=maps[f];ids[f]=(f>>3)*TABLE+(b[m[0]]|b[m[1]]<<4|b[m[2]]<<8|b[m[3]]<<12);}
  double v=0;for(int i:ids)v+=w[s][i];auto sorted=ids;std::sort(sorted.begin(),sorted.end());int norm=0;
  for(int i=0;i<FEATURES;){int j=i+1;while(j<FEATURES&&sorted[j]==sorted[i])j++;norm+=(j-i)*(j-i);i=j;}
  double err=target-v;
  if(!tcE){float d=float(alpha*err/norm);for(int i:ids)w[s][i]+=d;return;}
  for(int i:ids){float&E=(*tcE)[i],&A=(*tcA)[i];double rate=A>0?std::fabs(E)/A:1;w[s][i]+=float(alpha*rate*err/norm);E+=float(err);A+=float(std::fabs(err));}
 }
};
}
