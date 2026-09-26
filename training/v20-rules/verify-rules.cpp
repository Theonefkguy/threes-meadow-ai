// Checks the C++ engine rules against RULES.md (official-modern default, legacy switch).
// Run: g++ -std=c++17 -O2 training/v20-rules/verify-rules.cpp -o /tmp/v20-verify && /tmp/v20-verify
//      THREES_BONUS_RULE=legacy /tmp/v20-verify legacy
#include "../v17-fast-depth/search.h"
#include <climits>
#include <cstdio>
using namespace v2;
static int fails=0;
#define CHECK(c,msg) do{if(!(c)){std::printf("FAIL %s\n",msg);fails++;}}while(0)
int main(int argc,char**argv){
 init();bool legacy=argc>1;CHECK(BONUS_RULE==(legacy?0:1),"rule selection");
 // Marginals at max rank 11 (768): modern [1,2,3,2,1]/9, legacy uniform 1/5.
 for(int r=7;r<=14;r++){auto l=bonuses(r);double tot=0;std::array<double,16>m{};
  for(auto&x:l){tot+=x.second;double s=0;for(int j=0;j<x.first.size;j++){m[x.first.cards[j]]+=x.second*x.first.p[j];s+=x.first.p[j];}CHECK(std::abs(s-1)<1e-12,"window probs sum");
   if(!legacy){CHECK(std::abs(x.second-1./l.size())<1e-15,"uniform window");for(int j=0;j<x.first.size;j++)CHECK(std::abs(x.first.p[j]-1./x.first.size)<1e-15,"uniform in window");}}
  CHECK(std::abs(tot-1)<1e-12,"window mass");
  if(r==11){double e[5]={1,2,3,2,1};for(int k=0;k<5;k++)CHECK(std::abs(m[4+k]-(legacy?.2:e[k]/9))<1e-12,"marginals at 768");}}
 // Engine frequencies at max 384 (rank 10).
 RNG rng{20260925};std::array<int,16>cnt{};int bonus=0,N=420000;
 for(int i=0;i<N;i++){Game g;g.b={};g.b[1]=10;g.left=12;for(int k=0;k<12;k++)g.deck[k]=k/4+1;g.draw(rng);if(g.next.bonus){bonus++;
   double r=rng.next(),s=0;int card=g.next.cards[g.next.size-1];for(int j=0;j<g.next.size;j++){s+=g.next.p[j];if(r<s){card=g.next.cards[j];break;}}cnt[card]++;}}
 CHECK(std::abs(bonus-N/21.)<5*std::sqrt(N/21.),"1/21 bonus rate");
 double sh[4]={1/6.,1/3.,1/3.,1/6.};for(int k=0;k<4;k++){double exp=bonus*(legacy?.25:sh[k]);CHECK(std::abs(cnt[4+k]-exp)<5*std::sqrt(exp),"bonus card frequency");}
 // 12288 ending: two 6144 (rank 14) merge -> rank 15, game over, no spawn, no preview.
 Game g;g.b={14,14,3,1, 2,4,5,6, 7,8,9,10, 11,12,13,1};g.next={};g.next.size=1;g.next.cards[0]=2;g.next.p[0]=1;g.left=3;g.deck={1,2,3};
 auto m=move(g.b,0);CHECK(m.size==1,"one lane moves");double sc=score(g.b);g.advance(m,rng);
 CHECK(g.over&&g.b[0]==15&&g.next.size==0&&g.left==3,"12288 ends game, no draw");
 int twos=0;for(auto v:g.b)twos+=v==2;CHECK(twos==1,"no card enters after 12288");CHECK(std::abs(score(g.b)-sc-(std::pow(3.,13)-2*std::pow(3.,12)))<1e-6,"12288 score");
 // Search: the 12288 move is terminal (merge points only) at every depth.
 Learner mdl;mdl.load("training/v7/base.ntd");v17::Search s(mdl);s.maxNodes=INT_MAX;s.threshold=0.01;
 Board b{14,14,3,1, 2,4,5,6, 7,8,9,10, 11,12,13,1};Preview p;p.size=1;p.cards[0]=2;p.p[0]=1;
 for(int d=1;d<=5;d++){s.choose(b,p,Counts{3,3,3},d);CHECK(s.rootValues[0]==std::pow(3.,12),"terminal value in search");}
 std::printf("%s rule: %s (bonus draws %d; 6:%d 12:%d 24:%d 48:%d)\n",legacy?"legacy":"modern",fails?"FAILED":"all checks passed",bonus,cnt[4],cnt[5],cnt[6],cnt[7]);return fails!=0;}
