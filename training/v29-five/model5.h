#pragma once
// V29: the four-stage four-cell model (V21 Model4) plus an optional five-cell residual on
// stage 2 (max card 3072). value(b) = Model4 value + residual (stage 2 only). With no or an
// all-zero residual the value equals Model4's value exactly (adding 0.0 is exact).
// Residual: PATTERNS5 five-cell patterns x 8 D4 symmetries; each table 16^5 float32.
// File: header {0x3554444e, PATTERNS5, TABLE5, stage} then PATTERNS5*TABLE5 float32.
#include "../v21-late-train/model4.h"
namespace v29 {
using namespace v21;
constexpr int PATTERNS5=4, TABLE5=1<<20, FEATURES5=PATTERNS5*8, WEIGHTS5=PATTERNS5*TABLE5, RES_STAGE=2;
// Row/corner shapes most used by 2048 n-tuple programs, cut to five cells.
constexpr int patterns5[PATTERNS5][5]={{0,1,2,3,4},{4,5,6,7,8},{0,1,2,4,5},{4,5,6,8,9}};
inline int maps5[FEATURES5][5];
inline void initMaps5(){
 for(int p=0;p<PATTERNS5;p++)for(int sym=0;sym<8;sym++)for(int j=0;j<5;j++){
  int pos=patterns5[p][j],y=pos/4,x=pos%4;
  if(sym>=4)x=3-x;
  for(int k=0;k<sym%4;k++){int ny=x;x=3-y;y=ny;}
  maps5[p*8+sym][j]=y*4+x;}}
struct ModelR{
 Model4 base;std::vector<float> res;bool hasRes=false;
 void load(const std::string&path,const std::string&resPath){base.load(path);
  if(resPath=="-"){res.clear();hasRes=false;return;}
  res.assign(WEIGHTS5,0);hasRes=true;if(resPath=="zero")return;
  std::ifstream f(resPath,std::ios::binary);uint32_t h[4];f.read((char*)h,16);
  if(!f||h[0]!=0x3554444e||h[1]!=PATTERNS5||h[2]!=TABLE5||h[3]!=RES_STAGE)throw std::runtime_error("invalid residual");
  f.read((char*)res.data(),WEIGHTS5*4);if(!f)throw std::runtime_error("truncated residual");}
 void saveRes(const std::string&path)const{std::ofstream f(path,std::ios::binary);uint32_t h[4]={0x3554444e,PATTERNS5,TABLE5,RES_STAGE};f.write((char*)h,16);
  f.write((const char*)res.data(),WEIGHTS5*4);if(!f)throw std::runtime_error("residual write failed");}
 static int key5(const Board&b,const int*m){return b[m[0]]|b[m[1]]<<4|b[m[2]]<<8|b[m[3]]<<12|b[m[4]]<<16;}
 double resValue(const Board&b)const{double v=0;for(int f=0;f<FEATURES5;f++)v+=res[(f>>3)*TABLE5+key5(b,maps5[f])];return v;}
 double value(const Board&b,int s)const{double v=base.value(b,s);if(hasRes&&s==RES_STAGE)v+=resValue(b);return v;}
 double value(const Board&b)const{return value(b,stage4(b));}
 // TD step on the joint linear model (four-cell stage-s table plus, on stage 2, the residual):
 // every active weight moves by alpha*err/norm, norm = sum of squared multiplicities of all
 // active features. Without a residual this is exactly Model4::update (non-TC branch).
 void update(const Board&b,int s,double target,double alpha){
  if(!(hasRes&&s==RES_STAGE)){base.update(b,s,target,alpha);return;}
  std::array<int,FEATURES>ids;for(int f=0;f<FEATURES;f++){const int*m=maps[f];ids[f]=(f>>3)*TABLE+(b[m[0]]|b[m[1]]<<4|b[m[2]]<<8|b[m[3]]<<12);}
  std::array<int,FEATURES5>ids5;for(int f=0;f<FEATURES5;f++)ids5[f]=(f>>3)*TABLE5+key5(b,maps5[f]);
  double v=0;for(int i:ids)v+=base.w[s][i];for(int i:ids5)v+=res[i];
  auto mult=[](auto sorted){std::sort(sorted.begin(),sorted.end());int n=0;for(size_t i=0;i<sorted.size();){size_t j=i+1;while(j<sorted.size()&&sorted[j]==sorted[i])j++;n+=int((j-i)*(j-i));i=j;}return n;};
  int norm=mult(ids)+mult(ids5);
  float d=float(alpha*(target-v)/norm);for(int i:ids)base.w[s][i]+=d;for(int i:ids5)res[i]+=d;}
};
inline void init(){v21::init();initMaps5();}
}
