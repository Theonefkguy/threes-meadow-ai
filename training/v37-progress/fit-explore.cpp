// V37 post-hoc diagnostic (not in the frozen protocol): fit.cpp plus KEY "const", a single global bias
// weight, to separate a global level shift from progress information.
// V37 progress-feature screening: training/v36-substage/fit.cpp with the shared P5 residual plus
// one bias weight per value of a "progress" key of the stage-3 afterstate (V33 records, same base,
// split, SGD and output as V33/V36).
//  fit KEY BASE_MODEL PATTERNS DATA_BIN [DATA_BIN...] -- OUT_JSON
// Ranks after removing one 6144, sorted with multiplicity: s2 >= s3 >= s4. Bucket b(r): r<=9 -> 0
// (192 or less), 10 -> 1, 11 -> 2, 12 -> 3, 13 -> 4, 14 -> 5 (a second 6144).
//   none   no bias (= V33 P5)
//   bias3  s2 < 1536 | 1536 | >= 3072 (= V36 bias3)
//   s2     b(s2)                     6 values
//   s23    b(s2), b(s3)              36 values
//   s234   b(s2), b(s3), b(s4)       216 values
//   chain  768, 1536, 3072 present (bits) and a second 6144 (bit)   16 values
//   mass   min(40, sum of values of cards >= 192 (one 6144 removed) / 192)   41 values
// OUT_JSON adds the key, record counts per key value, validation MSE per key value and the
// fitted bias weights.
// V36 note: SPLIT none/two/three/four/bias3 (separate residual tables per group, or bias3).
// Original V33 note:
// V33 pattern screening: fit an n-tuple residual (any cell count, 8 D4 symmetries) on top of a
// fixed four-cell stage-3 table to realised future scores (collect.cpp records).
//  fit BASE_MODEL PATTERNS DATA_BIN [DATA_BIN...] -- OUT_JSON
// PATTERNS: "0,1,2,3,4;4,5,6,7,8;..." (all patterns the same size, 4-6 cells), or "none".
// Split: records whose episode index % 5 == 0 are validation (all passes of a start state stay
// together). Target y = G - base(s), base = stage-3 four-cell value of BASE_MODEL. SGD with the
// multiplicity-normalised update of the TD trainers, epochs over shuffled training records with
// rates .003/.001/.0003/.0001, fixed seed. OUT_JSON: per-epoch train/validation MSE and, for the
// epoch with the lowest validation MSE, the per-validation-episode sum of squared errors.
#include "../v21-late-train/model4.h"
#include <map>
#include <random>
using namespace v21;
#pragma pack(push,1)
struct Record{uint8_t b[16];double G;uint32_t episode,move;};
#pragma pack(pop)
int main(int argc,char**argv){try{
 int sep=0;for(int i=1;i<argc;i++)if(std::string(argv[i])=="--")sep=i;
 if(argc<7||!sep||sep+2!=argc)throw std::runtime_error("fit SPLIT BASE_MODEL PATTERNS DATA_BIN... -- OUT_JSON");
 const std::string split=argv[1];
 const int G=split=="none"?1:split=="bias3"?3:split=="s2"?6:split=="s23"?36:split=="s234"?216:split=="chain"?16:split=="mass"?41:split=="const"?1:0;if(!G)throw std::runtime_error("bad key");
 const bool bias=split!="none";const int tableGroups=1;
 init();Model4 base;base.load(argv[2]);
 std::vector<std::vector<int>>patterns;std::string spec=argv[3];
 if(spec!="none"){std::stringstream ss(spec);std::string p;while(std::getline(ss,p,';')){std::vector<int>cells;std::stringstream ps(p);std::string c;while(std::getline(ps,c,','))cells.push_back(std::stoi(c));patterns.push_back(cells);}}
 const int K=patterns.empty()?0:int(patterns[0].size());for(auto&p:patterns)if(int(p.size())!=K||K<4||K>6)throw std::runtime_error("bad patterns");
 const size_t TABLE=size_t(1)<<(4*K);const int F=int(patterns.size())*8;
 std::vector<std::vector<int>>maps(F);
 for(size_t p=0;p<patterns.size();p++)for(int sym=0;sym<8;sym++){std::vector<int>m;for(int pos:patterns[p]){int y=pos/4,x=pos%4;if(sym>=4)x=3-x;for(int k=0;k<sym%4;k++){int ny=x;x=3-y;y=ny;}m.push_back(y*4+x);}maps[p*8+sym]=m;}
 std::vector<Record>train,val;
 for(int i=4;i<sep;i++){std::ifstream f(argv[i],std::ios::binary);Record r;while(f.read((char*)&r,sizeof r))(r.episode%5==0?val:train).push_back(r);}
 if(train.empty()||val.empty())throw std::runtime_error("no data");
 auto group=[&](const Record&r){int ranks[16],n=0;bool skipped=false;double mass=0;
  for(int i=0;i<16;i++){int x=r.b[i];if(x==14&&!skipped){skipped=true;continue;}ranks[n++]=x;if(x>=9)mass+=3.0*double(1<<(x-3));}
  std::sort(ranks,ranks+n,std::greater<int>());auto bk=[](int x){return x<=9?0:x-9;};
  int s2=ranks[0],s3=ranks[1],s4=ranks[2];bool has[16]={};for(int i=0;i<n;i++)has[ranks[i]]=true;
  if(split=="none"||split=="const")return 0;if(split=="bias3")return s2<12?0:s2==12?1:2;
  if(split=="s2")return bk(s2);if(split=="s23")return bk(s2)*6+bk(s3);if(split=="s234")return (bk(s2)*6+bk(s3))*6+bk(s4);
  if(split=="chain")return int(has[11])|int(has[12])<<1|int(has[13])<<2|int(has[14])<<3;
  return std::min(40,int(mass/192+1e-9));};
 const size_t BIAS=F?TABLE*patterns.size()*tableGroups:0;
 std::vector<float>w(BIAS+(bias?G:0),0.f);
 auto feats=[&](const Record&r,std::vector<size_t>&ids){ids.resize(F);size_t off=bias?0:size_t(group(r))*TABLE*patterns.size();
  for(int f=0;f<F;f++){size_t key=0;for(int j=0;j<K;j++)key|=size_t(r.b[maps[f][j]])<<(4*j);ids[f]=off+(f/8)*TABLE+key;}
  if(bias)ids.push_back(BIAS+group(r));};
 auto baseValue=[&](const Record&r){Board b;for(int i=0;i<16;i++)b[i]=r.b[i];return base.value(b,3);};
 std::vector<double>ytr(train.size()),yva(val.size());
 for(size_t i=0;i<train.size();i++)ytr[i]=train[i].G-baseValue(train[i]);
 for(size_t i=0;i<val.size();i++)yva[i]=val[i].G-baseValue(val[i]);
 std::vector<size_t>ids;
 auto predict=[&](const Record&r){if(!F&&!bias)return 0.0;feats(r,ids);double v=0;for(size_t i:ids)v+=w[i];return v;};
 std::vector<double>groupSSE(G);std::vector<long>groupN(G);
 auto evaluate=[&](const std::vector<Record>&d,const std::vector<double>&y,std::map<uint32_t,std::pair<double,int>>*per){double s=0;
  for(size_t i=0;i<d.size();i++){double e=y[i]-predict(d[i]);s+=e*e;if(per){auto&q=(*per)[d[i].episode];q.first+=e*e;q.second++;int g=group(d[i]);groupSSE[g]+=e*e;groupN[g]++;}}return s/d.size();};
 const double rates[4]={.003,.001,.0003,.0001};std::mt19937_64 rng(33);std::vector<size_t>order(train.size());for(size_t i=0;i<order.size();i++)order[i]=i;
 std::vector<long>trainN(G);for(auto&r:train)trainN[group(r)]++;
 std::ofstream out(argv[sep+1]);out<<std::setprecision(10)<<"{\"split\":\""<<split<<"\",\"patterns\":\""<<spec<<"\",\"train\":"<<train.size()<<",\"validation\":"<<val.size()<<",\"trainPerGroup\":[";
 for(int g=0;g<G;g++)out<<(g?",":"")<<trainN[g];out<<"],\"epochs\":[";
 double best=evaluate(val,yva,nullptr);int bestEpoch=0;std::vector<float>bestW=w;
 out<<"{\"epoch\":0,\"trainMSE\":"<<evaluate(train,ytr,nullptr)<<",\"validationMSE\":"<<best<<"}";
 for(int e=0;(F||bias)&&e<4;e++){std::shuffle(order.begin(),order.end(),rng);
  for(size_t i:order){feats(train[i],ids);double v=0;for(size_t k:ids)v+=w[k];
   auto sorted=ids;std::sort(sorted.begin(),sorted.end());int norm=0;for(size_t a=0;a<sorted.size();){size_t b=a+1;while(b<sorted.size()&&sorted[b]==sorted[a])b++;norm+=int((b-a)*(b-a));a=b;}
   float d=float(rates[e]*(ytr[i]-v)/norm);for(size_t k:ids)w[k]+=d;}
  double vm=evaluate(val,yva,nullptr);out<<",{\"epoch\":"<<e+1<<",\"rate\":"<<rates[e]<<",\"trainMSE\":"<<evaluate(train,ytr,nullptr)<<",\"validationMSE\":"<<vm<<"}";
  if(vm<best){best=vm;bestEpoch=e+1;bestW=w;}}
 w=bestW;std::map<uint32_t,std::pair<double,int>>per;evaluate(val,yva,&per);
 out<<"],\"bestEpoch\":"<<bestEpoch<<",\"bestValidationMSE\":"<<best<<",\"validationPerGroup\":[";
 for(int g=0;g<G;g++)out<<(g?",":"")<<"{\"n\":"<<groupN[g]<<",\"mse\":"<<(groupN[g]?groupSSE[g]/groupN[g]:0)<<"}";
 out<<"],\"biasWeights\":[";for(int g=0;bias&&g<G;g++)out<<(g?",":"")<<w[BIAS+g];
 out<<"],\"perEpisode\":{";
 bool firstOut=true;for(auto&[ep,q]:per){out<<(firstOut?"":",")<<"\""<<ep<<"\":["<<q.first<<","<<q.second<<"]";firstOut=false;}
 out<<"}}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
