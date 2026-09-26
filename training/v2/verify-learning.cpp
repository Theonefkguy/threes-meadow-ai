#include "goal.h"
#include <cassert>
using namespace v2;
int main(){
 init();Board b{};std::vector<ScoreRecord> t;
 t.push_back({b,10,2,false});t.push_back({b,20,3,false});t.push_back({b,30,4,false});
 assert(std::abs(lambdaTarget(t,0)-(.5*22+.25*35+.25*9))<1e-12);
 t[2].boundary=true;assert(std::abs(lambdaTarget(t,0)-(.5*22+.5*35))<1e-12);
 t[0].reward=t[1].reward=0;t[2].value=1;assert(std::abs(lambdaTarget(t,0)-(.5*20+.5))<1e-12);
 t={{b,.6,0,false}};assert(lambdaTarget(t,0)==0);
 Preview p;p.size=1;p.cards[0]=1;p.p[0]=1;Counts c{0,1,2};GoalModel model(true);double before=model.value(b,p,c);model.update(b,p,c,1,.2);assert(model.value(b,p,c)>before);
 b[0]=13;assert(model.value(b,p,c)==1);
 for(int empty=0;empty<=16;empty++)for(int ones=0;ones<=16-empty;ones++)for(int twos=0;twos<=16-empty-ones;twos++)for(int size=1;size<=3;size++)for(int first=1;first<=12;first++)for(int n=0;n<125;n++){
  p.size=size;p.cards[0]=first;c={n%5,n/5%5,n/25};auto ids=contextIndices({empty,ones,twos},p,c);assert(ids[0]>=0&&ids[0]<CONTEXT&&ids[1]>=0&&ids[1]<CONTEXT);
 }
 std::cout<<"Multistep returns, terminal/goal boundaries, logistic update and context bounds passed.\n";
}
