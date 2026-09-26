// Independent, compact MS-TD prototype. No external model weights.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace threes {
using Board = std::array<uint8_t,16>;
constexpr int PATTERNS=8, TABLE=65536, FEATURES=64, WEIGHTS=PATTERNS*TABLE;
constexpr int patterns[PATTERNS][4]={
 {0,1,2,3},{4,5,6,7},{0,1,4,5},{1,2,5,6},
 {5,6,9,10},{0,1,2,4},{0,1,5,6},{0,1,4,8}
};
inline int maps[FEATURES][4];
inline double points(int r){return r<3?0:std::pow(3.,r-2);}
inline int high(const Board& b){return *std::max_element(b.begin(),b.end());}
inline int stage(const Board& b){int r=high(b);return r>=13?2:r>=12?1:0;}
inline double score(const Board& b){double s=0;for(auto r:b)s+=points(r);return s;}
inline bool mergeable(int a,int b){return a&&b&&(a+b==3||(a>=3&&a==b));}
inline void initMaps(){
 for(int p=0;p<PATTERNS;p++)for(int sym=0;sym<8;sym++)for(int j=0;j<4;j++){
  int pos=patterns[p][j],y=pos/4,x=pos%4;
  if(sym>=4)x=3-x;
  for(int k=0;k<sym%4;k++){int ny=x;x=3-y;y=ny;}
  maps[p*8+sym][j]=y*4+x;
 }
}
struct Projection{Board b{};std::array<int,4> entries{};int size=0;double reward=0;};
struct Row{std::array<uint8_t,4>b{};bool changed=false;double reward=0;};
inline std::array<Row,TABLE> rows;
inline Row slide(std::array<uint8_t,4> a){
 Row r;r.b=a;
 for(int j=1;j<4;j++)if(r.b[j]&&(!r.b[j-1]||mergeable(r.b[j],r.b[j-1]))){
  int a=r.b[j],b=r.b[j-1],c=!b?a:a+b==3?3:a+1;
  r.reward+=points(c)-points(a)-points(b);r.b[j-1]=c;r.b[j]=0;r.changed=true;
 }
 return r;
}
inline void init(){initMaps();for(int i=0;i<TABLE;i++)rows[i]=slide({uint8_t(i&15),uint8_t((i>>4)&15),uint8_t((i>>8)&15),uint8_t(i>>12)});}
inline Projection move(const Board& b,int dir){
 Projection m;m.b=b;
 for(int lane=0;lane<4;lane++){
  int idx[4];std::array<uint8_t,4>a;int key=0;bool small=true;
  for(int j=0;j<4;j++){
   idx[j]=dir==0?lane*4+j:dir==1?lane*4+3-j:dir==2?j*4+lane:(3-j)*4+lane;
   a[j]=b[idx[j]];key|=(a[j]&15)<<(j*4);small&=a[j]<16;
  }
  const Row r=small?rows[key]:slide(a);
  if(r.changed){for(int j=0;j<4;j++)m.b[idx[j]]=r.b[j];m.entries[m.size++]=idx[3];m.reward+=r.reward;}
 }
 return m;
}
inline bool legal(const Board& b){for(int d=0;d<4;d++)if(move(b,d).size)return true;return false;}
struct RNG{
 uint32_t state;
 double next(){state=state*1664525u+1013904223u;return double(state)/4294967296.;}
 int index(int n){return int(next()*n);}
};
struct Preview{std::array<uint8_t,3>cards{};std::array<double,3>p{};int size=0;bool bonus=false;};
inline std::vector<std::pair<Preview,double>> bonuses(int maxRank){
 int n=maxRank-6;std::vector<std::pair<Preview,double>>out;if(n<=0)return out;
 int windows=std::max(1,n-2),width=std::min(3,n);
 for(int w=0;w<windows;w++){
  Preview p;p.bonus=true;p.size=width;double mass=0;
  for(int j=0;j<width;j++){
   int v=w+j,count=0;for(int k=0;k<windows;k++)if(v>=k&&v<k+width)count++;
   p.cards[j]=4+v;p.p[j]=1./n/count;mass+=p.p[j];
  }
  for(int j=0;j<width;j++)p.p[j]/=mass;
  out.push_back({p,mass});
 }
 return out;
}
struct Game{
 Board b{};std::array<uint8_t,12>deck{};int left=0;Preview next{};int turns=0;bool over=false;
 void refill(RNG& rng){for(int i=0;i<12;i++)deck[i]=i/4+1;for(int i=11;i>0;i--)std::swap(deck[i],deck[rng.index(i+1)]);left=12;}
 int normal(RNG& rng){if(!left)refill(rng);return deck[--left];}
 void draw(RNG& rng){
  auto list=bonuses(high(b));
  if(!list.empty()&&rng.next()<1./21){double r=rng.next(),sum=0;next=list.back().first;for(auto& item:list){sum+=item.second;if(r<sum){next=item.first;break;}}}
  else{next={};next.size=1;next.cards[0]=normal(rng);next.p[0]=1;}
 }
 void reset(RNG& rng){
  b={};turns=0;over=false;refill(rng);for(int i=0;i<9;i++)b[i]=normal(rng);
  for(int i=15;i>0;i--)std::swap(b[i],b[rng.index(i+1)]);draw(rng);
 }
 double advance(const Projection& m,RNG& rng){
  b=m.b;int pos=m.entries[rng.index(m.size)],card=next.cards[0];
  if(next.size>1){double r=rng.next(),sum=0;card=next.cards[next.size-1];for(int i=0;i<next.size;i++){sum+=next.p[i];if(r<sum){card=next.cards[i];break;}}}
  b[pos]=card;turns++;over=!legal(b);if(!over)draw(rng);else next={};return points(card);
 }
};
// V(afterstate): undiscounted expected future score increments, including spawns.
// The learned table uses board patterns. Search separately conditions on the
// public next-card hint and remaining counts; no hidden deck enters evaluation.
struct Model{
 std::array<std::vector<float>,3>w;
 Model(){for(auto& t:w)t.assign(WEIGHTS,0);}
 std::array<int,FEATURES> indices(const Board& b)const{
  std::array<int,FEATURES> ids;
  for(int f=0;f<FEATURES;f++){
   int key=0;for(int j=0;j<4;j++)key|=std::min<int>(15,b[maps[f][j]])<<(j*4);
   ids[f]=(f/8)*TABLE+key;
  }return ids;
 }
 double value(const Board& b,int s)const{
  double v=0;for(int f=0;f<FEATURES;f++){
   int key=0;for(int j=0;j<4;j++)key|=std::min<int>(15,b[maps[f][j]])<<(j*4);
   v+=w[s][(f/8)*TABLE+key];
  }return v;
 }
 void update(const Board& b,int s,double target,double alpha){
  auto ids=indices(b);double v=0;for(int i:ids)v+=w[s][i];
  // Duplicated symmetry features really have multiplicity; normalise by the
  // squared feature norm so symmetric boards do not receive larger TD steps.
  auto sorted=ids;std::sort(sorted.begin(),sorted.end());int norm=0;
  for(int i=0;i<FEATURES;){int j=i+1;while(j<FEATURES&&sorted[j]==sorted[i])j++;norm+=(j-i)*(j-i);i=j;}
  float delta=alpha*(target-v)/norm;for(int i:ids)w[s][i]+=delta;
 }
 void save(const std::string& path)const{
  std::ofstream f(path,std::ios::binary);uint32_t head[4]={0x3144544e,3,PATTERNS,TABLE};f.write((char*)head,16);
  for(auto& t:w)f.write((char*)t.data(),t.size()*4);if(!f)throw std::runtime_error("checkpoint write failed");
 }
 void load(const std::string& path){
  std::ifstream f(path,std::ios::binary);uint32_t h[4];f.read((char*)h,16);
  if(!f||h[0]!=0x3144544e||h[1]!=3||h[2]!=PATTERNS||h[3]!=TABLE)throw std::runtime_error("invalid checkpoint");
  for(auto& t:w)f.read((char*)t.data(),t.size()*4);if(!f)throw std::runtime_error("truncated checkpoint");
 }
};
}
