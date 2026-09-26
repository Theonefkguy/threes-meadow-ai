#include "ntuple.h"
#include <iostream>
#include <iomanip>
#include <string>
using namespace threes;
int main(){init();std::cout<<std::setprecision(16);std::string cmd;while(std::cin>>cmd){
 if(cmd=="move"){int d,n;std::cin>>d;Board b;for(auto&x:b){std::cin>>n;x=n;}auto m=move(b,d);std::cout<<m.size<<' '<<m.reward;for(auto x:m.b)std::cout<<' '<<int(x);for(int i=0;i<m.size;i++)std::cout<<' '<<m.entries[i];std::cout<<'\n';}
 else if(cmd=="game"){uint32_t seed;std::cin>>seed;RNG rng{seed};Game g;g.reset(rng);for(int i=0;i<120&&!g.over;i++){
  for(auto x:g.b)std::cout<<int(x)<<' ';std::cout<<g.left<<' ';for(int j=0;j<g.left;j++)std::cout<<int(g.deck[j])<<' ';
  std::cout<<g.next.size<<' ';for(int j=0;j<g.next.size;j++)std::cout<<int(g.next.cards[j])<<' '<<g.next.p[j]<<' ';std::cout<<'\n';
  for(int d=i%4,k=0;k<4;k++,d=(d+1)%4){auto m=move(g.b,d);if(m.size){g.advance(m,rng);break;}}
 }std::cout<<"END\n";}
 else if(cmd=="value"){std::string path;int s,n;std::cin>>path>>s;Board b;for(auto&x:b){std::cin>>n;x=n;}Model m;m.load(path);std::cout<<m.value(b,s)<<'\n';}
 else if(cmd=="bonus"){int hi;std::cin>>hi;auto list=bonuses(hi);std::cout<<list.size();for(const auto&[p,mass]:list){std::cout<<' '<<mass<<' '<<p.size;for(int i=0;i<p.size;i++)std::cout<<' '<<int(p.cards[i])<<' '<<p.p[i];}std::cout<<'\n';}
}}
