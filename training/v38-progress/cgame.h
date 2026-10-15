// V38 game state under the bonus-cycle rule: training/v35-cycle/cgame.h for v38 (same code), plus
// the state-line format of the cycle-rule first-6144 pool.
#pragma once
#include "search5p.h"
namespace v38 {
struct CGame{
 Board b{};std::array<uint8_t,12>deck{};int left=0;Preview next{};int turns=0;bool over=false;
 int draws=0,slot=-1,bonusCount=0;
 void refill(RNG&rng){for(int i=0;i<12;i++)deck[i]=i/4+1;for(int i=11;i>0;i--)std::swap(deck[i],deck[rng.index(i+1)]);left=12;}
 int normal(RNG&rng){if(!left)refill(rng);return deck[--left];}
 void draw(RNG&rng){
  int d=draws++;
  if(d>=1){int pos=(d-1)%21;if(pos==0)slot=rng.index(21);
   auto list=bonuses(high(b));
   if(pos==slot&&!list.empty()){double r=rng.next(),sum=0;next=list.back().first;for(auto&item:list){sum+=item.second;if(r<sum){next=item.first;break;}}bonusCount++;return;}}
  next={};next.size=1;next.cards[0]=normal(rng);next.p[0]=1;
 }
 void reset(RNG&rng){b={};turns=0;over=false;draws=0;slot=-1;bonusCount=0;refill(rng);for(int i=0;i<9;i++)b[i]=normal(rng);for(int i=15;i>0;i--)std::swap(b[i],b[rng.index(i+1)]);draw(rng);}
 double advance(const Projection&m,RNG&rng){
  if(high(m.b)>=FINAL_RANK){b=m.b;turns++;over=true;next={};return 0;}
  b=m.b;int pos=m.entries[rng.index(m.size)],card=next.cards[0];
  if(next.size>1){double r=rng.next(),sum=0;card=next.cards[next.size-1];for(int i=0;i<next.size;i++){sum+=next.p[i];if(r<sum){card=next.cards[i];break;}}}
  b[pos]=card;turns++;over=!legal(b);if(!over)draw(rng);else next={};return points(card);
 }
};
// State line: seed turns left deck[left] next.size bonus (card p)x3 counts[3] board[16] draws slot cyclePos cycleUsed cycleExcl
struct CSnap{uint32_t seed=0;CGame g;Counts c{};Cycle cy;};
inline void writeSnap(std::ostream&o,uint32_t seed,const CGame&g,Counts c,const Cycle&cy){
 o<<std::setprecision(17)<<seed<<' '<<g.turns<<' '<<g.left;for(int i=0;i<g.left;i++)o<<' '<<int(g.deck[i]);
 o<<' '<<g.next.size<<' '<<int(g.next.bonus);for(int i=0;i<3;i++)o<<' '<<int(g.next.cards[i])<<' '<<g.next.p[i];
 for(auto v:c)o<<' '<<v;for(auto r:g.b)o<<' '<<int(r);o<<' '<<g.draws<<' '<<g.slot<<' '<<cy.pos<<' '<<cy.used<<' '<<cy.excl<<'\n';o.flush();}
inline std::vector<CSnap> readSnaps(const std::string&path){std::ifstream f(path);if(!f)throw std::runtime_error("no pool "+path);std::vector<CSnap>out;std::string line;
 while(std::getline(f,line)){std::istringstream in(line);CSnap s;int n;in>>s.seed>>s.g.turns>>s.g.left;if(s.g.left<0||s.g.left>12)throw std::runtime_error("bad deck");
  for(int i=0;i<s.g.left;i++){in>>n;s.g.deck[i]=n;}in>>s.g.next.size>>n;s.g.next.bonus=n;for(int i=0;i<3;i++){in>>n>>s.g.next.p[i];s.g.next.cards[i]=n;}
  for(auto&v:s.c)in>>v;for(auto&r:s.g.b){in>>n;r=n;}in>>s.g.draws>>s.g.slot>>s.cy.pos>>s.cy.used>>s.cy.excl;if(!in)throw std::runtime_error("bad pool line");out.push_back(s);}return out;}
}
