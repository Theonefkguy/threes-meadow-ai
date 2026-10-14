// V35 game state under the bonus-cycle rule (see game.cpp).
#pragma once
#include "search5c.h"
namespace v35 {
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
}
