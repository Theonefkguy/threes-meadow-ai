#include "goal.h"
#include <cassert>
using namespace v2;
int main(){init();Board b{};b[0]=13;std::vector<ScoreRecord> won{{b,.2,0,false},{b,1,0,true}};assert(lambdaTarget(won,0)==1);std::vector<ScoreRecord> lost{{b,.2,0,false}};assert(lambdaTarget(lost,0)==0);GoalModel g;Preview p;Counts c{};double before=g.value(b,p,c);g.update(b,1,.1);assert(g.value(b,p,c)>before);b[0]=14;assert(g.value(b,p,c)==1);b[0]=12;assert(g.value(b,p,c)==0);std::cout<<"terminal success/failure and probability update verified\n";}
