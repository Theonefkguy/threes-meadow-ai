// V67: training/v38-progress/search5p.h (struct Search -> LSearch) with a leaf cache that survives across moves.
// A depth-1 decision (a leaf, or a node cut off by the threshold) depends only on the board, so its value can be
// reused by any later search; the cache sits behind the transposition table and the table is filled exactly as
// before, so all values are bit-identical to Search.
#include "../v38-progress/cgame.h"
namespace v38 {
struct LSearch{
 const ModelP&model;double threshold=0;bool clampLeaf=false; // clampLeaf: true future score is never negative
 bool cycleAware=false;
int maxNodes=24000;int nodes=0,completed=0;uint64_t cutoffs=0;
 std::array<double,4>rootValues{};std::vector<Entry>table;uint32_t stamp=1;size_t mask;
 explicit LSearch(const ModelP&m,int bits=18):model(m),table(size_t(1)<<bits),mask((size_t(1)<<bits)-1){}
 // leaf cache: value of a depth-1 decision, a pure function of the board (key params = 1); looked up only after a
 // transposition-table miss and never changes what the table stores, so every search result is unchanged
 struct Leaf{uint64_t board=0;double value=0;};std::vector<Leaf>leaves;size_t leafMask=0;uint64_t leafHits=0,leafMisses=0;
 void leafCache(int bits){leaves.assign(bits?size_t(1)<<bits:0,Leaf{});leafMask=bits?(size_t(1)<<bits)-1:0;}
 void clearLeaves(){std::fill(leaves.begin(),leaves.end(),Leaf{});}
 void tick(){nodes++;if(completed&&nodes>maxNodes)throw 1;}
 void key(const Board&b,const Preview&p,Counts c,const Cycle&cy,int depth,Entry&k)const{
  k.board=0;for(int i=0;i<16;i++)k.board|=uint64_t(b[i])<<(i*4);
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
  Moves mv;for(int d=0;d<4;d++){mv.m[d]=move(b,d);mv.any|=mv.m[d].size>0;}if(!mv.any)return 0;
  auto&list=bonusList(high(b));bool possible=!list.empty();
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
  Leaf*leaf=nullptr;
  if(depth==1&&!root&&!leaves.empty()){leaf=&leaves[hash(k)&leafMask];
   if(leaf->board==k.board){leafHits++;if(slot){*slot=k;slot->value=leaf->value;slot->stamp=stamp;}return Answer{leaf->value,-1};}leafMisses++;}
  double best=-1e300;int direction=-1;
  for(int d=0;d<4;d++){
   const auto&m=mv.m[d];if(!m.size)continue;double q=m.reward;
   if(high(m.b)>=FINAL_RANK){}
   else if(depth==1)q+=clampLeaf?std::max(0.0,model.value(m.b)):model.value(m.b);
   else{double expected=0;for(int e=0;e<m.size;e++)for(int i=0;i<p.size;i++){
    tick();Board next=m.b;next[m.entries[e]]=p.cards[i];
    expected+=p.p[i]*(pts(p.cards[i])+future(next,counts,cy,depth-1,prob*p.p[i]/m.size));
   }q+=expected/m.size;}
   values[d]=q;if(q>best){best=q;direction=d;}
  }
  Answer a{direction<0?0:best,direction};
  if(leaf){leaf->board=k.board;leaf->value=a.value;}
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
