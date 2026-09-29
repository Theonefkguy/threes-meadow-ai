#pragma once
#include "ntuple-fast.h"
#include <chrono>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
namespace v2 {
using namespace threes;
using Counts=std::array<int,3>;
inline Counts observe(Counts c,const Preview&p){if(p.size&&!p.bonus){if(c[0]+c[1]+c[2]==0)c={4,4,4};c[p.cards[0]-1]--;}return c;}
inline Counts opening(const Game&g){Counts c{4,4,4};for(auto r:g.b)if(r>=1&&r<=3)c[r-1]--;return observe(c,g.next);}
struct Snapshot{Game game;Counts counts;};
struct Pool{
 std::vector<Snapshot> states;uint64_t seen=0;
 void add(const Game&g,Counts c,RNG&rng){
  if(g.over||high(g.b)!=13)return;seen++;
  if(states.size()<5000)states.push_back({g,c});else{uint64_t j=rng.next()*seen;if(j<states.size())states[j]={g,c};}
 }
 const Snapshot&sample(RNG&rng)const{return states[rng.index(states.size())];}
};
inline Pool loadPool(const std::string& path){
 Pool pool;std::ifstream f(path);std::string line;
 while(std::getline(f,line)){
  std::istringstream in(line);Game g;Counts c;int n;
  for(auto&v:g.b){in>>n;v=n;}in>>g.left;if(g.left<0||g.left>12)throw std::runtime_error("invalid deck size");for(int i=0;i<g.left;i++){in>>n;g.deck[i]=n;}
  in>>g.next.size;if(g.next.size<1||g.next.size>3)throw std::runtime_error("invalid preview size");for(int i=0;i<3&&i<g.next.size;i++){in>>n>>g.next.p[i];g.next.cards[i]=n;}
  in>>g.turns;for(auto&v:c)in>>v;g.next.bonus=g.next.cards[0]>=4;
  if(!in||high(g.b)!=13||!legal(g.b))throw std::runtime_error("invalid curriculum snapshot");
  // Old JSON snapshots rounded probabilities to six decimals. Reconstruct
  // the exact public posterior from its candidate window before resuming.
  if(g.next.bonus){bool found=false;for(auto item:bonuses(high(g.b)))if(item.first.cards==g.next.cards){g.next=item.first;found=true;break;}if(!found)throw std::runtime_error("invalid bonus window");}
  pool.states.push_back({g,c});
 }if(pool.states.empty())throw std::runtime_error("empty curriculum pool");pool.seen=pool.states.size();return pool;
}
inline void savePool(const Pool&pool,const std::string& path){
 std::ofstream out(path);out<<std::setprecision(17);
 for(auto&s:pool.states){const Game&g=s.game;for(auto r:g.b)out<<int(r)<<' ';out<<g.left<<' ';for(int i=0;i<g.left;i++)out<<int(g.deck[i])<<' ';
  out<<g.next.size<<' ';for(int i=0;i<g.next.size;i++)out<<int(g.next.cards[i])<<' '<<g.next.p[i]<<' ';out<<g.turns<<' ';for(auto v:s.counts)out<<v<<' ';out<<'\n';}
}
struct Leaf{Projection move;double scoreValue=0,logit=0;};
// A two-move policy, matching the afterstate cut used in browser depth=2.
inline Projection scorePolicy(const Game&g,const Model&model){
 double best=-1e300;Projection choice;
 for(int d=0;d<4;d++){
  auto m=move(g.b,d);if(!m.size)continue;double expected=0;
  for(int e=0;e<m.size;e++)for(int c=0;c<g.next.size;c++){
   Board b=m.b;int card=g.next.cards[c];b[m.entries[e]]=card;double value=-1e300;
   for(int d2=0;d2<4;d2++){auto next=move(b,d2);if(next.size)value=std::max(value,next.reward+model.value(next.b,stage(next.b)));}
   expected+=g.next.p[c]*(points(card)+(value==-1e300?0:value));
  }
  double q=m.reward+expected/m.size;if(q>best){best=q;choice=m;}
 }return choice;
}
struct ScoreRecord{Board board;double value=0,reward=0;bool boundary=false;};
// Five-step truncated lambda return, lambda=.5. A reached goal is a bootstrap
// boundary for the score model and an absorbing success for the probability model.
inline double lambdaTarget(const std::vector<ScoreRecord>& t,size_t i){
 constexpr double weights[5]={.5,.25,.125,.0625,.0625};
 double target=0,sum=0;size_t at=i;
 for(int n=0;n<5;n++){
  if(at<t.size()&&!t[at].boundary){sum+=t[at].reward;at++;}
  target+=weights[n]*(sum+(at<t.size()?t[at].value:0));
 }return target;
}
}
