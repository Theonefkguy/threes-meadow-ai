#pragma once
#include "../v3/common.h"
namespace v2 {
inline Pool loadEarlyPool(const std::string& path){
 Pool pool;std::ifstream f(path);std::string line;
 while(std::getline(f,line)){
  std::istringstream in(line);Game g;Counts c;int n;
  for(auto&v:g.b){in>>n;v=n;}in>>g.left;if(g.left<0||g.left>12)throw std::runtime_error("invalid deck size");for(int i=0;i<g.left;i++){in>>n;g.deck[i]=n;}
  in>>g.next.size;if(g.next.size<1||g.next.size>3)throw std::runtime_error("invalid preview size");for(int i=0;i<3&&i<g.next.size;i++){in>>n>>g.next.p[i];g.next.cards[i]=n;}
  in>>g.turns;for(auto&v:c)in>>v;g.next.bonus=g.next.cards[0]>=4;
  if(!in||high(g.b)>=12||!legal(g.b))throw std::runtime_error("invalid curriculum snapshot");
  // Old JSON snapshots rounded probabilities to six decimals. Reconstruct
  // the exact public posterior from its candidate window before resuming.
  if(g.next.bonus){bool found=false;for(auto item:bonuses(high(g.b)))if(item.first.cards==g.next.cards){g.next=item.first;found=true;break;}if(!found)throw std::runtime_error("invalid bonus window");}
  pool.states.push_back({g,c});
 }if(pool.states.empty())throw std::runtime_error("empty curriculum pool");pool.seen=pool.states.size();return pool;
}
inline void saveEarlyPool(const Pool&pool,const std::string& path){
 std::ofstream out(path);out<<std::setprecision(17);
 for(auto&s:pool.states){const Game&g=s.game;for(auto r:g.b)out<<int(r)<<' ';out<<g.left<<' ';for(int i=0;i<g.left;i++)out<<int(g.deck[i])<<' ';
  out<<g.next.size<<' ';for(int i=0;i<g.next.size;i++)out<<int(g.next.cards[i])<<' '<<g.next.p[i]<<' ';out<<g.turns<<' ';for(auto v:s.counts)out<<v<<' ';out<<'\n';}
}
}
