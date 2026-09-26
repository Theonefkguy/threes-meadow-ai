#pragma once
#include "../v9/net.h"
using namespace v2;
struct ActorState {Board b;Preview p;Counts c;int action=-1,band=0;};
struct Distribution{std::array<v9::Features,4>f;std::array<std::array<float,v9::H>,4>h;std::array<double,4>p{};int greedy=-1;};
inline Distribution distribution(const ActorState&s,const Learner&base,const v9::Net&net,double temperature){Distribution o;double max=-1e300,sum=0;std::array<double,4>logit{};for(int d=0;d<4;d++){auto a=move(s.b,d);if(!a.size){logit[d]=-1e300;continue;}double v=base.value(a.b,stage(a.b));o.f[d]=v9::features(a,s.p,s.c,v,net.context);auto out=net.forward(o.f[d],&o.h[d]);logit[d]=((a.reward+v)/2000+out[1])/temperature;if(logit[d]>max){max=logit[d];o.greedy=d;}}if(o.greedy<0)throw std::runtime_error("no actions");for(int d=0;d<4;d++){o.p[d]=logit[d]<-1e299?0:std::exp(logit[d]-max);sum+=o.p[d];}for(auto&v:o.p)v/=sum;return o;}
inline int sampleAction(const Distribution&o,RNG&r){double u=r.next(),sum=0;for(int d=0;d<4;d++){sum+=o.p[d];if(u<sum)return d;}for(int d=3;d>=0;d--)if(o.p[d]>0)return d;throw std::runtime_error("sample");}
