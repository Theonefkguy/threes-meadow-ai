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
 if(argc<6||!sep||sep+2!=argc)throw std::runtime_error("fit BASE_MODEL PATTERNS DATA_BIN... -- OUT_JSON");
 init();Model4 base;base.load(argv[1]);
 std::vector<std::vector<int>>patterns;std::string spec=argv[2];
 if(spec!="none"){std::stringstream ss(spec);std::string p;while(std::getline(ss,p,';')){std::vector<int>cells;std::stringstream ps(p);std::string c;while(std::getline(ps,c,','))cells.push_back(std::stoi(c));patterns.push_back(cells);}}
 const int K=patterns.empty()?0:int(patterns[0].size());for(auto&p:patterns)if(int(p.size())!=K||K<4||K>6)throw std::runtime_error("bad patterns");
 const size_t TABLE=size_t(1)<<(4*K);const int F=int(patterns.size())*8;
 std::vector<std::vector<int>>maps(F);
 for(size_t p=0;p<patterns.size();p++)for(int sym=0;sym<8;sym++){std::vector<int>m;for(int pos:patterns[p]){int y=pos/4,x=pos%4;if(sym>=4)x=3-x;for(int k=0;k<sym%4;k++){int ny=x;x=3-y;y=ny;}m.push_back(y*4+x);}maps[p*8+sym]=m;}
 std::vector<Record>train,val;
 for(int i=3;i<sep;i++){std::ifstream f(argv[i],std::ios::binary);Record r;while(f.read((char*)&r,sizeof r))(r.episode%5==0?val:train).push_back(r);}
 if(train.empty()||val.empty())throw std::runtime_error("no data");
 std::vector<float>w(F?TABLE*patterns.size():0,0.f);
 auto feats=[&](const Record&r,std::vector<size_t>&ids){ids.resize(F);for(int f=0;f<F;f++){size_t key=0;for(int j=0;j<K;j++)key|=size_t(r.b[maps[f][j]])<<(4*j);ids[f]=(f/8)*TABLE+key;}};
 auto baseValue=[&](const Record&r){Board b;for(int i=0;i<16;i++)b[i]=r.b[i];return base.value(b,3);};
 std::vector<double>ytr(train.size()),yva(val.size());
 for(size_t i=0;i<train.size();i++)ytr[i]=train[i].G-baseValue(train[i]);
 for(size_t i=0;i<val.size();i++)yva[i]=val[i].G-baseValue(val[i]);
 std::vector<size_t>ids;
 auto predict=[&](const Record&r){if(!F)return 0.0;feats(r,ids);double v=0;for(size_t i:ids)v+=w[i];return v;};
 auto evaluate=[&](const std::vector<Record>&d,const std::vector<double>&y,std::map<uint32_t,std::pair<double,int>>*per){double s=0;
  for(size_t i=0;i<d.size();i++){double e=y[i]-predict(d[i]);s+=e*e;if(per){auto&q=(*per)[d[i].episode];q.first+=e*e;q.second++;}}return s/d.size();};
 const double rates[4]={.003,.001,.0003,.0001};std::mt19937_64 rng(33);std::vector<size_t>order(train.size());for(size_t i=0;i<order.size();i++)order[i]=i;
 std::ofstream out(argv[sep+1]);out<<std::setprecision(10)<<"{\"patterns\":\""<<spec<<"\",\"train\":"<<train.size()<<",\"validation\":"<<val.size()<<",\"epochs\":[";
 double best=evaluate(val,yva,nullptr);int bestEpoch=0;std::vector<float>bestW=w;
 out<<"{\"epoch\":0,\"trainMSE\":"<<evaluate(train,ytr,nullptr)<<",\"validationMSE\":"<<best<<"}";
 for(int e=0;F&&e<4;e++){std::shuffle(order.begin(),order.end(),rng);
  for(size_t i:order){feats(train[i],ids);double v=0;for(size_t k:ids)v+=w[k];
   auto sorted=ids;std::sort(sorted.begin(),sorted.end());int norm=0;for(size_t a=0;a<sorted.size();){size_t b=a+1;while(b<sorted.size()&&sorted[b]==sorted[a])b++;norm+=int((b-a)*(b-a));a=b;}
   float d=float(rates[e]*(ytr[i]-v)/norm);for(size_t k:ids)w[k]+=d;}
  double vm=evaluate(val,yva,nullptr);out<<",{\"epoch\":"<<e+1<<",\"rate\":"<<rates[e]<<",\"trainMSE\":"<<evaluate(train,ytr,nullptr)<<",\"validationMSE\":"<<vm<<"}";
  if(vm<best){best=vm;bestEpoch=e+1;bestW=w;}}
 w=bestW;std::map<uint32_t,std::pair<double,int>>per;evaluate(val,yva,&per);
 out<<"],\"bestEpoch\":"<<bestEpoch<<",\"bestValidationMSE\":"<<best<<",\"perEpisode\":{";
 bool firstOut=true;for(auto&[ep,q]:per){out<<(firstOut?"":",")<<"\""<<ep<<"\":["<<q.first<<","<<q.second<<"]";firstOut=false;}
 out<<"}}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
