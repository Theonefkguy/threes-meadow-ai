// V68: the website model (v38::ModelP without bias = v31::ModelS) evaluated with the same arithmetic in the same order,
// so every value is bit-identical, but faster (FastModel: arm 1/2; NeonModel: arm 3/4):
//  - the feature maps are compile-time constants and the feature loops are unrolled (fold expressions keep the order);
//  - HALF: the five-cell residuals are stored as float16 with a power-of-two scale. The web residuals are already
//    float16 times 2^scaleExp (scripts/quantize-residual.py), so float(half) * 2^scaleExp is the stored float32 exactly;
//    from() checks this for every weight.
#pragma once
#include "searcht.h"
#include <utility>
#include <arm_neon.h>
namespace v68 {
template<int P,int N> struct Maps{int m[P*8][N];};
template<int P,int N> constexpr Maps<P,N> makeMaps(const int(&pat)[P][N]){Maps<P,N> r{};
 for(int p=0;p<P;p++)for(int sym=0;sym<8;sym++)for(int j=0;j<N;j++){
  int pos=pat[p][j],y=pos/4,x=pos%4;if(sym>=4)x=3-x;
  for(int k=0;k<sym%4;k++){int ny=x;x=3-y;y=ny;}
  r.m[p*8+sym][j]=y*4+x;}
 return r;}
inline constexpr Maps<threes::PATTERNS,4> M4=makeMaps(threes::patterns);
inline constexpr Maps<v29::PATTERNS5,5> M5=makeMaps(v29::patterns5);
template<bool HALF> struct FastModel{
 std::array<const float*,v21::STAGES>base{};std::array<std::vector<__fp16>,v21::STAGES>res16;std::array<const float*,v21::STAGES>res32{};
 std::array<float,v21::STAGES>scale{};std::array<bool,v21::STAGES>has{};
 void from(const ModelP&m){if(m.hasBias)throw std::runtime_error("bias not supported");
  for(int s=0;s<v21::STAGES;s++){base[s]=m.s.base.w[s].data();has[s]=m.s.has[s];if(!has[s])continue;res32[s]=m.s.res[s].data();
   if(!HALF)continue;const auto&r=m.s.res[s];bool ok=false;
   for(int e=0;e<8&&!ok;e++){float sc=std::ldexp(1.0f,e);res16[s].resize(r.size());ok=true;
    for(size_t i=0;i<r.size();i++){__fp16 h=__fp16(r[i]/sc);if(float(h)*sc!=r[i]){ok=false;break;}res16[s][i]=h;}
    if(ok)scale[s]=sc;}
   if(!ok)throw std::runtime_error("residual is not float16 times a power of two");}}
 template<size_t...F> static double baseSum(const float*t,const Board&b,std::index_sequence<F...>){double v=0;
  ((v+=t[(F>>3)*threes::TABLE+(b[M4.m[F][0]]|b[M4.m[F][1]]<<4|b[M4.m[F][2]]<<8|b[M4.m[F][3]]<<12)]),...);return v;}
 static int key5(const Board&b,int f){return b[M5.m[f][0]]|b[M5.m[f][1]]<<4|b[M5.m[f][2]]<<8|b[M5.m[f][3]]<<12|b[M5.m[f][4]]<<16;}
 template<size_t...F> double resSum(int s,const Board&b,std::index_sequence<F...>)const{double v=0;
  if constexpr(HALF){const __fp16*r=res16[s].data();const float sc=scale[s];((v+=float(r[(F>>3)*v29::TABLE5+key5(b,F)])*sc),...);}
  else{const float*r=res32[s];((v+=r[(F>>3)*v29::TABLE5+key5(b,F)]),...);}
  return v;}
 double value(const Board&b)const{int r=fastHigh(b),s=r>=14?3:r>=13?2:r>=12?1:0;double v=baseSum(base[s],b,std::make_index_sequence<threes::FEATURES>{});
  if(has[s])v+=resSum(s,b,std::make_index_sequence<v29::FEATURES5>{});return v;}
};

// NeonModel: the 8 rotated / mirrored boards are built with one table lookup each and packed to 64-bit nibble boards
// (cell c in bits 4c..4c+3); a feature index is then a few shifts of the packed board. Same sums in the same order.
struct Perms{uint8_t p[8][16];};
constexpr Perms makePerms(){Perms r{};for(int sym=0;sym<8;sym++)for(int c=0;c<16;c++){int y=c/4,x=c%4;if(sym>=4)x=3-x;
  for(int k=0;k<sym%4;k++){int ny=x;x=3-y;y=ny;}r.p[sym][c]=uint8_t(y*4+x);}return r;}
inline constexpr Perms PERMS=makePerms();
template<int N> constexpr uint32_t nib(uint64_t P,const int(&pat)[N]){uint32_t i=0;for(int j=0;j<N;j++)i|=uint32_t((P>>(4*pat[j]))&15)<<(4*j);return i;}
template<bool HALF> struct NeonModel:FastModel<HALF>{
 using FastModel<HALF>::base;using FastModel<HALF>::res16;using FastModel<HALF>::res32;using FastModel<HALF>::scale;using FastModel<HALF>::has;
 static void packed(const Board&b,uint64_t(&P)[8]){uint8x16_t v=vld1q_u8(b.data());
  for(int s=0;s<8;s++)P[s]=pack(vqtbl1q_u8(v,vld1q_u8(PERMS.p[s])));}
 template<size_t...F> static double baseSum(const float*t,const uint64_t(&P)[8],std::index_sequence<F...>){double v=0;
  ((v+=t[(F>>3)*threes::TABLE+nib(P[F&7],threes::patterns[F>>3])]),...);return v;}
 template<size_t...F> double resSum(int s,const uint64_t(&P)[8],std::index_sequence<F...>)const{double v=0;
  if constexpr(HALF){const __fp16*r=res16[s].data();const float sc=scale[s];((v+=float(r[(F>>3)*v29::TABLE5+nib(P[F&7],v29::patterns5[F>>3])])*sc),...);}
  else{const float*r=res32[s];((v+=r[(F>>3)*v29::TABLE5+nib(P[F&7],v29::patterns5[F>>3])]),...);}
  return v;}
 double value(const Board&b)const{int s=v21::stage4(b);uint64_t P[8];packed(b,P);double v=baseSum(base[s],P,std::make_index_sequence<threes::FEATURES>{});
  if(has[s])v+=resSum(s,P,std::make_index_sequence<v29::FEATURES5>{});return v;}
};
}
