#include "actor.h"
#include <ctime>
using namespace v9;
struct Episode{std::vector<ActorState>states;bool won;};
int main(int argc,char**argv){try{if(argc!=4)throw std::runtime_error("INITIAL SEED OUT");init();Learner base;base.load("training/v7/base.ntd");Net net,initial;net.load(argv[1]);initial=net;uint32_t seed=std::stoul(argv[2]);std::string out=argv[3];std::filesystem::create_directories(out);std::ofstream log(out+"/training.jsonl");RNG gameRng{seed},actions{seed^0x51abcdef},replay{seed^0x483b21};std::vector<ActorState>anchors;
 {std::ifstream f("training/v11/teacher.txt");uint32_t source;int won;while(f>>source>>won){ActorState s;for(auto&x:s.b){int n;f>>n;x=n;}f>>s.p.size;for(int i=0;i<3;i++){int n;f>>n>>s.p.p[i];s.p.cards[i]=n;}for(auto&v:s.c)f>>v;double q;for(int i=0;i<4;i++)f>>q;if(!f)throw std::runtime_error("anchor data");if((source-11410001)%5!=0)anchors.push_back(s);}}
 std::array<float,N>mom{},variance{},grad{};std::array<double,4>baseline{.5,.5,.5,.5};uint64_t steps=0;int totalWins=0;double begin=double(std::clock())/CLOCKS_PER_SEC;net.save(out+"/round-0.net");
 for(int batch=1;batch<=128;batch++){
  std::vector<Episode>episodes;std::array<double,4>bandSum{},bandCount{};int wins=0;grad.fill(0);
  for(int e=0;e<64;e++){Game g;g.reset(gameRng);Counts c=opening(g);Episode ep;while(!g.over&&high(g.b)<12&&g.turns<6000){int rank=high(g.b);ActorState s{g.b,g.next,c,-1,rank<7?0:rank<9?1:rank<11?2:3};auto o=distribution(s,base,net,.1);s.action=sampleAction(o,actions);ep.states.push_back(s);g.advance(move(g.b,s.action),gameRng);c=observe(c,g.next);steps++;}if(!g.over&&high(g.b)<12)throw std::runtime_error("truncated training");ep.won=high(g.b)>=12;wins+=ep.won;episodes.push_back(std::move(ep));}
  for(auto&ep:episodes)for(auto&s:ep.states){auto o=distribution(s,base,net,.1);double advantage=double(ep.won)-baseline[s.band];for(int d=0;d<4;d++)if(!o.f[d].empty())net.backward(o.f[d],o.h[d],0,float(-advantage*((d==s.action?1.:0)-o.p[d])/.1/64),grad);bandSum[s.band]+=ep.won;bandCount[s.band]++;}
  // Replay old training states with the frozen student's own probabilities.
  // This anchors old behavior; only fresh game outcomes supply improvement signals.
  for(int i=0;i<64;i++){auto&s=anchors[replay.index(anchors.size())];auto old=distribution(s,base,initial,1),cur=distribution(s,base,net,1);for(int d=0;d<4;d++)if(!cur.f[d].empty())net.backward(cur.f[d],cur.h[d],0,float(.02*(cur.p[d]-old.p[d])/64),grad);}
  double norm=0;for(float v:grad)norm+=double(v)*v;norm=std::sqrt(norm);double clip=norm>1?1/norm:1;double c1=1-std::pow(.9,batch),c2=1-std::pow(.999,batch);
  for(int j=0;j<N;j++){float g=grad[j]*clip;mom[j]=.9f*mom[j]+.1f*g;variance[j]=.999f*variance[j]+.001f*g*g;net.w[j]-=3e-5*(mom[j]/c1)/(std::sqrt(variance[j]/c2)+1e-8);if(!std::isfinite(net.w[j]))throw std::runtime_error("nonfinite");}
  for(int k=0;k<4;k++)if(bandCount[k])baseline[k]=.9*baseline[k]+.1*bandSum[k]/bandCount[k];totalWins+=wins;
  log<<"{\"batch\":"<<batch<<",\"episodes\":"<<batch*64<<",\"batchWins\":"<<wins<<",\"totalWins\":"<<totalWins<<",\"steps\":"<<steps<<",\"gradientNorm\":"<<norm<<",\"cpuSeconds\":"<<double(std::clock())/CLOCKS_PER_SEC-begin<<"}"<<std::endl;
  if(batch%32==0)net.save(out+"/round-"+std::to_string(batch/32)+".net");
 }
 for(int j=0;j<H;j++)if(net.w[V+j]!=0)throw std::runtime_error("value head update");
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
