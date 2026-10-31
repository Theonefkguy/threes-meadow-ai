// V65: the V64 judges (training/v64-cascade/analyze.py, network B) in C++: P(the depth-d iteration already picks the
// depth-6 move) for d = 4, 5 from the iterations so far and the position; judge.txt = thr4 thr5, then per network:
// inputs, mu, sd, W1 b1 W2 b2 W3 b3 (out x in). Features as analyze.py feats(): for l = 1..d the four root values
// relative to the best (illegal -1, clipped to [-1, 0]), for l = 1..d the margin (best - second) / max(|best|, 1)
// capped at 1, for l < d whether depth l picked the depth-d move, legal count / 4, empty cells / 16, top / 15,
// board / 15, preview size / 3, first preview card / 15, bonus flag.
#pragma once
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace v65 {
struct Net{int n=0;std::vector<float>mu,sd,w1,b1,w2,b2,w3,b3;
 float prob(const std::vector<float>&x)const{
  std::array<float,128>h1,h2;
  for(int o=0;o<128;o++){float s=b1[o];for(int i=0;i<n;i++)s+=w1[o*n+i]*((x[i]-mu[i])/sd[i]);h1[o]=s>0?s:0;}
  for(int o=0;o<128;o++){float s=b2[o];for(int i=0;i<128;i++)s+=w2[o*128+i]*h1[i];h2[o]=s>0?s:0;}
  float s=b3[0];for(int i=0;i<128;i++)s+=w3[i]*h2[i];return 1/(1+std::exp(-s));}};
struct Judge{double thr[2]{2,2};Net net[2];
 void load(const std::string&path){std::ifstream f(path);if(!f)throw std::runtime_error("no judge "+path);f>>thr[0]>>thr[1];
  for(auto&q:net){f>>q.n;auto rd=[&](std::vector<float>&v,size_t k){v.resize(k);for(auto&x:v)f>>x;};
   rd(q.mu,q.n);rd(q.sd,q.n);rd(q.w1,size_t(128)*q.n);rd(q.b1,128);rd(q.w2,128*128);rd(q.b2,128);rd(q.w3,128);rd(q.b3,1);}
  if(!f)throw std::runtime_error("bad judge "+path);}
 // d = 4 or 5; dirs[l], vals[l] for depths l+1 = 1..d; board ranks; preview size, first card, bonus flag
 static std::vector<float>features(int d,const int*dirs,const std::array<double,4>*vals,const std::array<uint8_t,16>&board,int psize,int pcard,int pbonus){
  std::vector<float>x;bool legal[4];int nl=0;for(int i=0;i<4;i++){legal[i]=vals[d-1][i]>-1e299;nl+=legal[i];}
  for(int l=0;l<d;l++){double mx=-1e300;for(double v:vals[l])mx=std::max(mx,v);
   for(int i=0;i<4;i++){double r=legal[i]?(vals[l][i]-mx)/std::max(std::fabs(mx),1.0):-1;x.push_back(float(std::min(0.0,std::max(-1.0,r))));}}
  for(int l=0;l<d;l++){double best=-INFINITY,second=-INFINITY;for(int i=0;i<4;i++)if(legal[i]){double v=vals[l][i];if(v>best){second=best;best=v;}else if(v>second)second=v;}
   double m=std::isfinite(second)?(best-second)/std::max(std::fabs(best),1.0):INFINITY;x.push_back(float(std::min(m,1.0)));}
  for(int l=0;l<d-1;l++)x.push_back(dirs[l]==dirs[d-1]?1.f:0.f);
  int empty=0,top=0;for(int c:board){empty+=c==0;top=std::max(top,c);}
  x.push_back(float(nl/4.0));x.push_back(float(empty/16.0));x.push_back(float(top/15.0));
  for(int c:board)x.push_back(float(c/15.0));
  x.push_back(float(psize/3.0));x.push_back(float(pcard/15.0));x.push_back(float(pbonus));
  return x;}
 double prob(int d,const int*dirs,const std::array<double,4>*vals,const std::array<uint8_t,16>&board,int psize,int pcard,int pbonus)const{
  auto x=features(d,dirs,vals,board,psize,pcard,pbonus);const Net&q=net[d-4];if(int(x.size())!=q.n)throw std::runtime_error("feature size");return q.prob(x);}
};}
