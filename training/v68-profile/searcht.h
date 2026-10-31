// V68: training/v38-progress/search5p.h (struct Search) as a template on the model type, so faster models with
// bit-identical values can be dropped in. Generated copy; only the struct name, model type and namespace differ, plus
// three exact speed-ups: the board packing in key() (vector instructions), moves (fastMove) and the maximum card
// (fastHigh), all with the same results as the originals.
#pragma once
#include "../v38-progress/cgame.h"
#include <arm_neon.h>
#include <cstring>
#include <type_traits>
namespace v68 {
using namespace v38;
inline uint64_t pack(uint8x16_t v){uint16x8_t w=vreinterpretq_u16_u8(v);return vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(vorrq_u16(w,vshrq_n_u16(w,4)))),0);}
// V68: threes::move with the lane cells as compile-time constants per direction (same row table, same lane order,
// same result); boards with a card >= 16 fall back to threes::move.
template<int dir> inline Projection moveT(const Board&b){Projection m;m.b=b;
 auto lane=[&](auto L){constexpr int l=decltype(L)::value;
  constexpr int i0=dir==0?l*4:dir==1?l*4+3:dir==2?l:12+l,i1=dir==0?l*4+1:dir==1?l*4+2:dir==2?4+l:8+l,
                i2=dir==0?l*4+2:dir==1?l*4+1:dir==2?8+l:4+l,i3=dir==0?l*4+3:dir==1?l*4:dir==2?12+l:l;
  const Row&r=rows[b[i0]|b[i1]<<4|b[i2]<<8|b[i3]<<12];
  if(r.changed){m.b[i0]=r.b[0];m.b[i1]=r.b[1];m.b[i2]=r.b[2];m.b[i3]=r.b[3];m.entries[m.size++]=i3;m.reward+=r.reward;}};
 lane(std::integral_constant<int,0>{});lane(std::integral_constant<int,1>{});lane(std::integral_constant<int,2>{});lane(std::integral_constant<int,3>{});
 return m;}
inline int fastHigh(const Board&b){return vmaxvq_u8(vld1q_u8(b.data()));}   // = threes::high
inline Projection fastMove(const Board&b,int dir){uint64_t lo,hi;std::memcpy(&lo,b.data(),8);std::memcpy(&hi,b.data()+8,8);
 if((lo|hi)&0xF0F0F0F0F0F0F0F0ULL)return move(b,dir);
 switch(dir){case 0:return moveT<0>(b);case 1:return moveT<1>(b);case 2:return moveT<2>(b);default:return moveT<3>(b);}}
template<class M> struct SearchT{
 const M&model;double threshold=0;bool clampLeaf=false; // clampLeaf: true future score is never negative
 bool cycleAware=false;
int maxNodes=24000;int nodes=0,completed=0;uint64_t cutoffs=0;
 std::array<double,4>rootValues{};std::vector<Entry>table;uint32_t stamp=1;size_t mask;
 explicit SearchT(const M&m,int bits=18):model(m),table(size_t(1)<<bits),mask((size_t(1)<<bits)-1){}
 void tick(){nodes++;if(completed&&nodes>maxNodes)throw 1;}
 void key(const Board&b,const Preview&p,Counts c,const Cycle&cy,int depth,Entry&k)const{
  k.board=pack(vld1q_u8(b.data()));   // V68: same value as the loop over the 16 cells (every cell < 16)
  if(depth==1){k.params=1;return;}
  k.params=depth|(p.size<<3)|(c[0]<<5)|(c[1]<<8)|(c[2]<<11);
  for(int i=0;i<3&&i<p.size;i++)k.params|=uint64_t(p.cards[i])<<(14+i*4);
  if(cycleAware)k.params|=cy.pack()<<30;
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
 // cy: cycle state of the draw made at this chance node.
 double future(const Board&b,Counts counts,const Cycle&cy,int depth,double prob){
  Moves mv;for(int d=0;d<4;d++){mv.m[d]=fastMove(b,d);mv.any|=mv.m[d].size>0;}if(!mv.any)return 0;
  auto&list=bonusList(fastHigh(b));bool possible=!list.empty();
  double chance=!possible?0:!cycleAware?1./21:cy.used?0:1./(21-cy.excl),total=0;Counts deck=counts;if(deck[0]+deck[1]+deck[2]==0)deck={4,4,4};int size=deck[0]+deck[1]+deck[2];
  Cycle normal=cy.after(false,possible),bonus=cy.after(true,possible);
  if(chance<1)for(int n=0;n<3;n++)if(deck[n]){Preview p;p.size=1;p.cards[0]=n+1;p.p[0]=1;Counts next=deck;next[n]--;double w=(1-chance)*double(deck[n])/size;total+=w*decision(b,mv,p,next,normal,depth,prob*w).value;}
  if(chance>0)for(auto&item:list)total+=chance*item.second*decision(b,mv,item.first,counts,bonus,depth,prob*chance*item.second).value;
  return total;
 }
 // cy: cycle state of the next draw (the preview after the move being decided).
 Answer decision(const Board&b,const Moves&mv,const Preview&p,Counts counts,const Cycle&cy,int depth,double prob,bool root=false){
  if(depth>1&&!root&&prob<threshold){depth=1;cutoffs++;}
  std::array<double,4>values;values.fill(-1e300);
  tick();Entry k;key(b,p,counts,cy,depth,k);bool hit=false;Entry*slot=nullptr;if(!root){slot=find(k,hit);if(hit)return Answer{slot->value,-1};}
  double best=-1e300;int direction=-1;
  for(int d=0;d<4;d++){
   const auto&m=mv.m[d];if(!m.size)continue;double q=m.reward;
   if(fastHigh(m.b)>=FINAL_RANK){}
   else if(depth==1)q+=clampLeaf?std::max(0.0,model.value(m.b)):model.value(m.b);
   else{double expected=0;for(int e=0;e<m.size;e++)for(int i=0;i<p.size;i++){
    tick();Board next=m.b;next[m.entries[e]]=p.cards[i];
    expected+=p.p[i]*(pts(p.cards[i])+future(next,counts,cy,depth-1,prob*p.p[i]/m.size));
   }q+=expected/m.size;}
   values[d]=q;if(q>best){best=q;direction=d;}
  }
  Answer a{direction<0?0:best,direction};
  if(slot){*slot=k;slot->value=a.value;slot->stamp=stamp;}
  if(root)rootValues=values;return a;
 }
 Answer choose(const Board&b,const Preview&p,Counts c,const Cycle&cy,int maxDepth=3){
  stamp++;nodes=0;completed=0;cutoffs=0;Answer result;
  Moves mv;for(int d=0;d<4;d++)mv.m[d]=move(b,d);
  for(int depth=1;depth<=maxDepth;depth++){try{result=decision(b,mv,p,c,cy,depth,1.0,true);completed=depth;}catch(int){break;}}
  return result;
 }
};
}
