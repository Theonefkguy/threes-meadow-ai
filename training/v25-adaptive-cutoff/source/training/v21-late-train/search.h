#pragma once
// V21 search: identical to training/v17-fast-depth/search.h (probability cutoff,
// compact transposition table, 12288 terminal) but evaluating leaves with the
// four-stage Model4 (stage 3 for afterstates with a 6144 or larger).
#include "model4.h"
#include <cstring>
namespace v21 {
struct Answer{double value=0;int direction=-1;};
struct Entry{uint64_t board=0,params=0;double value=0;uint32_t stamp=0;};
struct Search{
 const Model4&model;double threshold=0;bool clampLeaf=false; // clampLeaf: true future score is never negative
int maxNodes=24000;int nodes=0,completed=0;uint64_t cutoffs=0;
 std::array<double,4>rootValues{};std::vector<Entry>table;uint32_t stamp=1;size_t mask;
 explicit Search(const Model4&m,int bits=18):model(m),table(size_t(1)<<bits),mask((size_t(1)<<bits)-1){}
 void tick(){nodes++;if(completed&&nodes>maxNodes)throw 1;}
 static void key(const Board&b,const Preview&p,Counts c,int depth,Entry&k){
  k.board=0;for(int i=0;i<16;i++)k.board|=uint64_t(b[i])<<(i*4);
  if(depth==1){k.params=1;return;}
  k.params=depth|(p.size<<3)|(c[0]<<5)|(c[1]<<8)|(c[2]<<11);
  for(int i=0;i<3&&i<p.size;i++)k.params|=uint64_t(p.cards[i])<<(14+i*4);
 }
 static size_t hash(const Entry&k){uint64_t x=k.board^(k.params*0x9e3779b97f4a7c15ULL);x^=x>>31;x*=0xbf58476d1ce4e5b9ULL;return x^(x>>29);}
 Entry*find(const Entry&k,bool&hit){size_t h=hash(k);Entry*victim=nullptr;
  for(int i=0;i<8;i++){Entry&e=table[(h+i)&mask];
   if(e.stamp!=stamp){if(!victim)victim=&e;continue;}
   if(e.board==k.board&&e.params==k.params){hit=true;return &e;}}
  hit=false;return victim;}
 static const std::vector<std::pair<Preview,double>>&bonusList(int r){static std::array<std::vector<std::pair<Preview,double>>,32>c;static std::array<bool,32>done{};if(!done[r]){c[r]=bonuses(r);done[r]=true;}return c[r];}
 static double pts(int r){static const auto t=[]{std::array<double,32>a{};for(int i=0;i<32;i++)a[i]=points(i);return a;}();return t[r];}
 struct Moves{std::array<Projection,4>m;bool any=false;};
 double future(const Board&b,Counts counts,int depth,double prob){
  Moves mv;for(int d=0;d<4;d++){mv.m[d]=move(b,d);mv.any|=mv.m[d].size>0;}if(!mv.any)return 0;
  auto&list=bonusList(high(b));double chance=list.empty()?0:1./21,total=0;Counts deck=counts;if(deck[0]+deck[1]+deck[2]==0)deck={4,4,4};int size=deck[0]+deck[1]+deck[2];
  for(int n=0;n<3;n++)if(deck[n]){Preview p;p.size=1;p.cards[0]=n+1;p.p[0]=1;Counts next=deck;next[n]--;double w=(1-chance)*double(deck[n])/size;total+=w*decision(b,mv,p,next,depth,prob*w).value;}
  for(auto&item:list)total+=chance*item.second*decision(b,mv,item.first,counts,depth,prob*chance*item.second).value;
  return total;
 }
 Answer decision(const Board&b,const Moves&mv,const Preview&p,Counts counts,int depth,double prob,bool root=false){
  if(depth>1&&!root&&prob<threshold){depth=1;cutoffs++;}
  std::array<double,4>values;values.fill(-1e300);
  tick();Entry k;key(b,p,counts,depth,k);bool hit=false;Entry*slot=nullptr;if(!root){slot=find(k,hit);if(hit)return Answer{slot->value,-1};}
  double best=-1e300;int direction=-1;
  for(int d=0;d<4;d++){
   const auto&m=mv.m[d];if(!m.size)continue;double q=m.reward;
   if(high(m.b)>=FINAL_RANK){}
   else if(depth==1)q+=clampLeaf?std::max(0.0,model.value(m.b)):model.value(m.b);
   else{double expected=0;for(int e=0;e<m.size;e++)for(int i=0;i<p.size;i++){
    tick();Board next=m.b;next[m.entries[e]]=p.cards[i];
    expected+=p.p[i]*(pts(p.cards[i])+future(next,counts,depth-1,prob*p.p[i]/m.size));
   }q+=expected/m.size;}
   values[d]=q;if(q>best){best=q;direction=d;}
  }
  Answer a{direction<0?0:best,direction};
  if(slot){*slot=k;slot->value=a.value;slot->stamp=stamp;}
  if(root)rootValues=values;return a;
 }
 Answer choose(const Board&b,const Preview&p,Counts c,int maxDepth=3){
  stamp++;nodes=0;completed=0;cutoffs=0;Answer result;
  Moves mv;for(int d=0;d<4;d++)mv.m[d]=move(b,d);
  for(int depth=1;depth<=maxDepth;depth++){try{result=decision(b,mv,p,c,depth,1.0,true);completed=depth;}catch(int){break;}}
  return result;
 }
};
}
