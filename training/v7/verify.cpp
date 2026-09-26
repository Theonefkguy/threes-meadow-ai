#include "search.h"
#include <cassert>
using namespace v2;
int main(){init();Learner m;m.load("training/v7/base.ntd");RNG rng{775511};Search search(m);for(int i=0;i<100;i++){Game g;g.reset(rng);Counts c=opening(g);int turns=i*3;for(int j=0;j<turns&&!g.over;j++){auto p=policyTwo(g,m);g.advance(p,rng);c=observe(c,g.next);}if(g.over)continue;
 auto a=search.choose(g.b,g.next,c,1);assert(search.rootValues[a.direction]==a.value);for(int d=0;d<4;d++){auto p=move(g.b,d);if(p.size)assert(std::abs(search.rootValues[d]-p.reward-m.value(p.b,stage(p.b)))<1e-7);}
 auto expected=search.rootValues;search.maxNodes=1;auto cut=search.choose(g.b,g.next,c,4);assert(search.completed==1&&search.rootValues==expected&&cut.direction==a.direction);search.maxNodes=96000;
 auto deep=search.choose(g.b,g.next,c,4);assert(search.rootValues[deep.direction]==deep.value);for(int d=0;d<4;d++){auto p=move(g.b,d);if(p.size)assert(std::isfinite(search.rootValues[d])&&search.rootValues[d]>-1e299);}
 auto two=search.choose(g.b,g.next,c,2);auto p=policyTwo(g,m);assert(move(g.b,two.direction).b==p.b);
 }
 auto late1=m.base.w[1],late2=m.base.w[2];Board b{};b[0]=11;b[1]=3;double value=m.value(b,0);m.update(b,value+100,.01);assert(m.value(b,0)>value);assert(m.base.w[1]==late1&&m.base.w[2]==late2);
 std::cout<<"PASS root action targets, reward subtraction, interrupted-iteration atomicity, two-ply policy agreement, finite teacher values, frozen later stages\n";
}
