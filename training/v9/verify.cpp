#include "search.h"
#include "../v8-search/mcts.h"
#include <cassert>
using namespace v9;
int main(){init();Learner base;base.load("training/v7/base.ntd");Net net;net.context=true;net.initialize(123);Game g;RNG rng{7654321};g.reset(rng);auto counts=opening(g);auto m=move(g.b,0);if(!m.size)m=move(g.b,1);auto f=features(m,g.next,counts,base.value(m.b,0),true);auto original=net.forward(f);assert(original[0]==0&&original[1]==0);StudentSearch student(base,net);MCTS reference(base);student.budget=reference.budget=100;student.maxSimulations=reference.maxSimulations=128;for(int i=0;i<20;i++){g.reset(rng);counts=opening(g);auto a=reference.choose(g.b,g.next,counts,i+1);auto b=student.choose(g.b,g.next,counts,i+1);assert(a.direction==b.direction);for(int d=0;d<4;d++)assert(reference.nodes[0]->edges[d].visits==student.nodes[0]->edges[d].visits);}
 for(int i=V;i<O;i++)net.w[i]=.02f*(i%7-3);std::array<float,H>h;net.forward(f,&h);std::array<float,N>grad{};net.backward(f,h,.31,-.22,grad);for(int i: {f[0].i*H,f[1].i*H+3,B+2,V+3,P+3,O,O+1}){float old=net.w[i],eps=.001;net.w[i]=old+eps;auto plus=net.forward(f);net.w[i]=old-eps;auto minus=net.forward(f);net.w[i]=old;double numeric=(.31*(plus[0]-minus[0])-.22*(plus[1]-minus[1]))/(2*eps);assert(std::abs(numeric-grad[i])<.0003);}
 for(int sym=0;sym<8;sym++){bool seen[16]{};for(int pos=0;pos<16;pos++){int j=transform(pos,sym);assert(j>=0&&j<16&&!seen[j]);seen[j]=true;}}
 std::cout<<"20 zero-residual MCTS parity cases; sparse-MLP gradients; allD4 maps verified\n";
}
