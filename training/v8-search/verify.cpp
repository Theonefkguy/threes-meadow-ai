#include "selective.h"
#include "mcts.h"
#include "../v7/pool.h"
#include <cassert>
using namespace v2;
int main(){init();Learner m;m.load("training/v7/base.ntd");Reference r(m);Selective s(m);MCTS mc(m);auto pool=loadEarlyPool("training/v7/assessment-pool.txt");r.budget=s.budget=100;for(int i=0;i<40;i++){auto&p=pool.states[i];auto a=r.choose(p.game.b,p.game.next,p.counts,2);auto b=s.choose(p.game.b,p.game.next,p.counts,2);assert(a.direction==b.direction&&std::abs(a.value-b.value)<1e-8);s.adaptive=true;s.margin=1e300;a=r.choose(p.game.b,p.game.next,p.counts,3);b=s.choose(p.game.b,p.game.next,p.counts,3);assert(a.direction==b.direction&&std::abs(a.value-b.value)<1e-8);if(i<8){a=r.choose(p.game.b,p.game.next,p.counts,4);b=s.choose(p.game.b,p.game.next,p.counts,4);assert(a.direction==b.direction&&std::abs(a.value-b.value)<1e-8&&s.pruned==0);}mc.budget=.0001;auto c=mc.choose(p.game.b,p.game.next,p.counts,100+i);assert(c.direction>=0&&move(p.game.b,c.direction).size);for(auto&e:mc.nodes[0]->edges)if(e.move.size)assert(e.visits>=1);}
 // Empirical sampler check against exact public hint distribution.
 Board board{};board[0]=11;Counts counts{1,2,3};std::array<double,16>expected{},actual{};double bonusExpected=0;hints(board,counts,[&](const Preview&p,Counts c){if(p.bonus)bonusExpected+=0;return 0.;});int bonus=0;mc.rng.seed(202609228);for(int i=0;i<200000;i++){auto c=counts;auto p=mc.nextPreview(board,c);if(p.bonus){bonus++;assert(c==counts);}else{actual[p.cards[0]]++;assert(c[p.cards[0]-1]==counts[p.cards[0]-1]-1);}}assert(std::abs(double(bonus)/200000-1./21)<.002);for(int k=1;k<=3;k++)assert(std::abs(actual[k]/200000-(20./21)*counts[k-1]/6)<.004);
 std::cout<<"40 shallow parity and8 depth4 exact-reduction cases; all root actions explored; public normal/bonus sampling verified\n";
}
