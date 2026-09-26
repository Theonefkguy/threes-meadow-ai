#include "search.h"
#include <cassert>
using namespace v2;
int main(){init();Learner m;m.load("dist/models/ntuple-v4.bin");auto later1=m.base.w[1],later2=m.base.w[2];m.initializeProbability();Board b{};b[0]=11;b[1]=11;b[5]=1;b[6]=2;assert(std::abs(m.prob(b)-.9)<1e-6);double before=m.prob(b);m.update(b,0,.1);assert(m.prob(b)<before);m.update(b,1,.1);assert(m.prob(b)>0&&m.prob(b)<1);assert(m.base.w[1]==later1&&m.base.w[2]==later2);
 for(int sym=0;sym<8;sym++){Board x{};for(int i=0;i<16;i++){int y=i/4,z=i%4;if(sym>=4)z=3-z;for(int k=0;k<sym%4;k++){int ny=z;z=3-y;y=ny;}x[y*4+z]=b[i];}assert(std::abs(m.prob(x)-m.prob(b))<1e-8);}
 Board goal=b;goal[0]=12;assert(m.prob(goal)==1);bool rejected=false;try{m.update(goal,1,.1);}catch(...){rejected=true;}assert(rejected);
 std::vector<ScoreRecord> t{{b,.2,0,false},{goal,1,0,true}};assert(lambdaTarget(t,0)==1);t={{b,.2,0,false}};assert(lambdaTarget(t,0)==0);t={{b,100,7,false},{goal,500,0,true}};assert(lambdaTarget(t,0)==507);
 Preview p;p.size=1;p.cards[0]=1;p.p[0]=1;Counts c{2,3,4};Search search(m);auto a=search.choose(b,p,c);assert(a.value==1&&a.direction>=0);auto projected=move(b,a.direction);assert(high(projected.b)>=12);
 Board dead={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};assert(search.choose(dead,p,c).direction==-1);assert(search.future(dead,c,2)==0);
 Learner base;base.load("dist/models/ntuple-v4.bin");Search reference(base);auto x=search.choose(goal,p,c),y=reference.choose(goal,p,c);assert(x.direction==y.direction&&x.value==y.value&&search.nodes==reference.nodes);
 RNG rng{97531};for(int i=0;i<100;i++){Game g;g.reset(rng);for(auto&r:g.b)if(r>3)r=3;auto z=search.choose(g.b,g.next,opening(g),2);assert(z.value>=0&&z.value<=1.00000001);}
 std::cout<<"PASS normalized probability updates, frozen stages, D4 symmetry, absorbing goal/death, lambda boundary, score handoff, probability-only backup\n";
}
