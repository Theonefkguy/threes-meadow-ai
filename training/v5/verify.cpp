#include "model.h"
#include <cassert>
using namespace v2;
int main(){init();Board corner{},middle{};corner[0]=13;corner[5]=12;middle[5]=13;middle[0]=12;assert(potential(corner)==1&&potential(middle)==0);Board early=corner;early[0]=12;assert(!potential(early));
 for(double c:{0.,2048.,8192.,32768.}){
  assert(shapedReward(0,corner,&middle,c)==-c);assert(shapedReward(0,middle,&corner,c)==c);assert(shapedReward(50,corner,nullptr,c)==50-c);
  for(int length=1;length<10;length++){std::vector<ScoreRecord>original,shaped;for(int i=0;i<length;i++)original.push_back({i%2?middle:corner,1000.+i*11,30.+i,false});shaped=original;
   for(int i=0;i<length;i++){shaped[i].value-=c*potential(shaped[i].board);shaped[i].reward=shapedReward(original[i].reward,original[i].board,i+1<length?&original[i+1].board:nullptr,c);}
   double sum=0,raw=0;for(int i=0;i<length;i++){sum+=shaped[i].reward;raw+=original[i].reward;assert(std::abs(lambdaTarget(shaped,i)+c*potential(shaped[i].board)-lambdaTarget(original,i))<1e-8);}assert(std::abs(sum-raw+c*potential(shaped[0].board))<1e-8);
  }
 }
 Learner m;m.coefficient=8192;assert(m.value(corner,2)==8192&&m.value(corner,1)==0);m.update(corner,100,1);assert(std::abs(m.raw(corner,2)-100)<.001);assert(std::abs(m.value(corner,2)-8292)<.001);assert(m.raw(corner,0)==0&&m.raw(corner,1)==0);
 for(int sym=0;sym<8;sym++){Board b{};for(int i=0;i<16;i++){int y=i/4,x=i%4;if(sym>=4)x=3-x;for(int k=0;k<sym%4;k++){int ny=x;x=3-y;y=ny;}b[y*4+x]=corner[i];}assert(potential(b)==1);assert(std::abs(m.value(b,2)-m.value(corner,2))<.001);}
 std::cout<<"PASS terminal/cycle/telescoping, five-step lambda equivalence, corrected inference, stage isolation, D4 symmetry\n";
}
