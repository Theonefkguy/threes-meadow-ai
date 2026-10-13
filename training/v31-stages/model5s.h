#pragma once
// V31: the V29 model generalised to one optional five-cell residual per stage.
// value(b,s) = Model4 value + residual of stage s (if present). Residual files use the V29
// format (header {0x3554444e, PATTERNS5, TABLE5, stage}); the stage comes from the header, so
// V29/V30 stage-2 residuals load unchanged. RES arguments: "-" (none) or a comma-separated
// list of residual files, where "zero:S" adds an all-zero residual for stage S.
// With only a stage-2 residual this is exactly v29::ModelR (same arithmetic, same order).
#include "../v29-five/model5.h"
namespace v31 {
using namespace v29;
struct ModelS{
 Model4 base;std::array<std::vector<float>,STAGES>res;std::array<bool,STAGES>has{};
 void load(const std::string&path,const std::string&list){base.load(path);has.fill(false);for(auto&r:res)r.clear();
  if(list=="-")return;std::stringstream ss(list);std::string item;
  while(std::getline(ss,item,',')){
   if(item.rfind("zero:",0)==0){int s=std::stoi(item.substr(5));if(s<0||s>=STAGES||has[s])throw std::runtime_error("bad zero stage");res[s].assign(WEIGHTS5,0);has[s]=true;continue;}
   std::ifstream f(item,std::ios::binary);uint32_t h[4];f.read((char*)h,16);
   if(!f||h[0]!=0x3554444e||h[1]!=PATTERNS5||h[2]!=TABLE5||h[3]>=STAGES)throw std::runtime_error("invalid residual "+item);
   int s=int(h[3]);if(has[s])throw std::runtime_error("two residuals for one stage");res[s].assign(WEIGHTS5,0);
   f.read((char*)res[s].data(),WEIGHTS5*4);if(!f)throw std::runtime_error("truncated residual "+item);has[s]=true;}}
 void saveRes(int s,const std::string&path)const{if(!has[s])throw std::runtime_error("no residual to save");std::ofstream f(path,std::ios::binary);
  uint32_t h[4]={0x3554444e,PATTERNS5,TABLE5,uint32_t(s)};f.write((char*)h,16);f.write((const char*)res[s].data(),WEIGHTS5*4);if(!f)throw std::runtime_error("residual write failed");}
 static int key5(const Board&b,const int*m){return b[m[0]]|b[m[1]]<<4|b[m[2]]<<8|b[m[3]]<<12|b[m[4]]<<16;}
 double resValue(const Board&b,int s)const{const float*r=res[s].data();double v=0;for(int f=0;f<FEATURES5;f++)v+=r[(f>>3)*TABLE5+key5(b,maps5[f])];return v;}
 double value(const Board&b,int s)const{double v=base.value(b,s);if(has[s])v+=resValue(b,s);return v;}
 double value(const Board&b)const{return value(b,stage4(b));}
 // Joint TD step on stage s, as v29::ModelR::update (falls back to Model4::update without a residual).
 void update(const Board&b,int s,double target,double alpha){
  if(!has[s]){base.update(b,s,target,alpha);return;}
  std::array<int,FEATURES>ids;for(int f=0;f<FEATURES;f++){const int*m=maps[f];ids[f]=(f>>3)*TABLE+(b[m[0]]|b[m[1]]<<4|b[m[2]]<<8|b[m[3]]<<12);}
  std::array<int,FEATURES5>ids5;for(int f=0;f<FEATURES5;f++)ids5[f]=(f>>3)*TABLE5+key5(b,maps5[f]);
  double v=0;for(int i:ids)v+=base.w[s][i];for(int i:ids5)v+=res[s][i];
  auto mult=[](auto sorted){std::sort(sorted.begin(),sorted.end());int n=0;for(size_t i=0;i<sorted.size();){size_t j=i+1;while(j<sorted.size()&&sorted[j]==sorted[i])j++;n+=int((j-i)*(j-i));i=j;}return n;};
  int norm=mult(ids)+mult(ids5);
  float d=float(alpha*(target-v)/norm);for(int i:ids)base.w[s][i]+=d;for(int i:ids5)res[s][i]+=d;}
};
inline void init(){v29::init();}
}
