#pragma once
#include "../v7/model.h"
#include <random>
namespace v9 {
using namespace v2;
constexpr int D=292,H=32,W=D*H,B=W,V=B+H,P=V+H,O=P+H,N=O+2;
struct Feature{int i;float x;};
using Features=std::vector<Feature>;
inline int transform(int pos,int sym){int y=pos/4,x=pos%4;if(sym>=4)x=3-x;for(int k=0;k<sym%4;k++){int t=x;x=3-y;y=t;}return y*4+x;}
inline Features features(const Projection&m,const Preview&p,Counts c,double base,bool context,int sym=0){Features f;f.reserve(32);for(int i=0;i<16;i++)f.push_back({transform(i,sym)*16+int(m.b[i]),1});if(context){for(int i=0;i<p.size;i++)f.push_back({256+p.cards[i],float(p.p[i])});for(int i=0;i<3;i++)if(c[i])f.push_back({272+i,c[i]/4.f});for(int i=0;i<m.size;i++)f.push_back({275+transform(m.entries[i],sym),1});}f.push_back({291,float(base/100000)});return f;}
struct Net{
 std::array<float,N>w{};bool context=false,policy=false;
 void initialize(uint32_t seed){std::mt19937 rng(seed);std::normal_distribution<float>normal(0,.08);for(int i=0;i<W;i++)w[i]=normal(rng);}
 std::array<float,2> forward(const Features&f,std::array<float,H>*saved=nullptr)const{std::array<float,H> h{};for(int j=0;j<H;j++)h[j]=w[B+j];for(auto a:f)for(int j=0;j<H;j++)h[j]+=w[a.i*H+j]*a.x;std::array<float,2> o{w[O],w[O+1]};for(int j=0;j<H;j++){h[j]=std::max(0.f,h[j]);o[0]+=w[V+j]*h[j];o[1]+=w[P+j]*h[j];}if(saved)*saved=h;return o;}
 void backward(const Features&f,const std::array<float,H>&h,float dv,float dp,std::array<float,N>&g)const{g[O]+=dv;g[O+1]+=dp;for(int j=0;j<H;j++){g[V+j]+=dv*h[j];g[P+j]+=dp*h[j];float dh=h[j]>0?w[V+j]*dv+w[P+j]*dp:0;g[B+j]+=dh;for(auto a:f)g[a.i*H+j]+=a.x*dh;}}
 void save(const std::string&path)const{std::ofstream f(path,std::ios::binary);uint32_t header[4]{0x394e4854,D,H,uint32_t(context)|(uint32_t(policy)<<1)};f.write((char*)header,16);f.write((char*)w.data(),N*4);if(!f)throw std::runtime_error("save");}
 void load(const std::string&path){std::ifstream f(path,std::ios::binary);uint32_t h[4];f.read((char*)h,16);if(!f||h[0]!=0x394e4854||h[1]!=D||h[2]!=H)throw std::runtime_error("net header");context=h[3]&1;policy=h[3]&2;f.read((char*)w.data(),N*4);if(!f)throw std::runtime_error("net data");for(float x:w)if(!std::isfinite(x))throw std::runtime_error("nonfinite net");}
};
}
