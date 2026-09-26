#include "search.h"
#include "pool.h"
using namespace v2;
std::pair<double,double> wilson(int k,int n){double z=1.96,p=double(k)/n,d=1+z*z/n,c=(p+z*z/(2*n))/d,h=z*std::sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return {c-h,c+h};}
int main(int argc,char**argv){try{
 if(argc!=6)throw std::runtime_error("BASE POOL FIRST COUNT PREFIX");init();Learner m;m.load(argv[1]);Search search(m),teacher(m);teacher.maxNodes=96000;Pool p=loadEarlyPool(argv[2]);int first=std::stoi(argv[3]),count=std::stoi(argv[4]);std::string prefix=argv[5];std::ofstream out(prefix+".jsonl"),examples(prefix+".teacher.txt");out<<std::setprecision(17);examples<<std::setprecision(17);
 for(int i=first;i<first+count;i++){const auto&s=p.states.at(i);int wins[4]={-1,-1,-1,-1},n=8;auto original=search.choose(s.game.b,s.game.next,s.counts);std::vector<int>valid;for(int d=0;d<4;d++)if(move(s.game.b,d).size){valid.push_back(d);wins[d]=0;}
  auto trials=[&](int begin,int end){for(int d:valid)for(int k=begin;k<end;k++){RNG rng{uint32_t(2126100000u+i*256+d*48+k)};Game g=s.game;Counts c=s.counts;for(int j=g.left-1;j>0;j--)std::swap(g.deck[j],g.deck[rng.index(j+1)]);g.advance(move(g.b,d),rng);c=observe(c,g.next);int local=0;while(!g.over&&high(g.b)<12&&local<6000){auto a=search.choose(g.b,g.next,c);g.advance(move(g.b,a.direction),rng);c=observe(c,g.next);local++;}if(!g.over&&high(g.b)<12)throw std::runtime_error("assessment truncated");wins[d]+=high(g.b)>=12;}};
  trials(0,8);auto sorted=valid;std::sort(sorted.begin(),sorted.end(),[&](int a,int b){return wins[a]>wins[b];});if(sorted.size()>1&&wins[sorted[0]]-wins[sorted[1]]<2){trials(8,24);n=24;}
  int best=valid[0],worst=valid[0],total=0;for(int d:valid){if(wins[d]>wins[best])best=d;if(wins[d]<wins[worst])worst=d;total+=wins[d];}
  bool hard=wins[best]>=.25*n&&total>0&&total<int(valid.size())*n;bool clear=wilson(wins[best],n).first>wilson(wins[worst],n).second;auto a=teacher.choose(s.game.b,s.game.next,s.counts,4);int weight=clear?3:1;
  if(hard)for(int d:valid){auto moved=move(s.game.b,d);if(high(moved.b)>=12)continue;double target=teacher.rootValues[d]-moved.reward;if(!std::isfinite(target)||target<=-1e299)throw std::runtime_error("invalid teacher target");examples<<i<<' '<<d<<' '<<weight<<' '<<target;for(auto r:moved.b)examples<<' '<<int(r);examples<<'\n';}
  out<<"{\"index\":"<<i<<",\"trialsPerAction\":"<<n<<",\"wins\":["<<wins[0]<<','<<wins[1]<<','<<wins[2]<<','<<wins[3]<<"],\"best\":"<<best<<",\"worst\":"<<worst<<",\"baselineAction\":"<<original.direction<<",\"hard\":"<<(hard?"true":"false")<<",\"clearSpread\":"<<(clear?"true":"false")<<",\"teacherAction\":"<<a.direction<<",\"teacherDepth\":"<<teacher.completed<<",\"teacherNodes\":"<<teacher.nodes<<"}\n";out.flush();examples.flush();
 }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
