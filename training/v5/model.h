#pragma once
#include "../v3/goal.h"
#include <ctime>
namespace v2 {
// Potential on nonterminal AFTERSTATES. The absorbing terminal is represented
// separately in the trainer, with potential zero even if its board has a corner max.
inline int potential(const Board& b){int h=high(b);return h>=13&&(b[0]==h||b[3]==h||b[12]==h||b[15]==h);}
struct Learner {
 Model base;double coefficient=0;
 double raw(const Board& b,int s)const{return base.value(b,s);}
 double value(const Board& b,int s)const{return raw(b,s)+(s==2?coefficient*potential(b):0);}
 void update(const Board& b,double shapedTarget,double alpha){base.update(b,2,shapedTarget,alpha);}
 void load(const std::string& path,double c=0){if(!std::isfinite(c)||c<0)throw std::runtime_error("invalid coefficient");coefficient=c;base.load(path);}
 void save(const std::string& path)const{base.save(path+".ntd");}
};
inline double shapedReward(double reward,const Board& current,const Board* next,double coefficient){return reward+coefficient*((next?potential(*next):0)-potential(current));}
inline Projection policyTwo(const Game&g,const Learner&model){double best=-1e300;Projection chosen;for(int d=0;d<4;d++){auto m=move(g.b,d);if(!m.size)continue;double expected=0;for(int e=0;e<m.size;e++)for(int c=0;c<g.next.size;c++){Board b=m.b;int card=g.next.cards[c];b[m.entries[e]]=card;double value=-1e300;for(int d2=0;d2<4;d2++){auto n=move(b,d2);if(n.size)value=std::max(value,n.reward+model.value(n.b,stage(n.b)));}expected+=g.next.p[c]*(points(card)+(value==-1e300?0:value));}double q=m.reward+expected/m.size;if(q>best){best=q;chosen=m;}}return chosen;}
}
