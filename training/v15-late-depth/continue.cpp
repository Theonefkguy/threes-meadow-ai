#include "../v7/search.h"
#include "../v7/pool.h"
#include <climits>
#include <ctime>
#include <cstdio>
using namespace v2;

struct ResumeState{
 Game game;Counts counts{};uint32_t rngState=0;int startTurns=0,local=0;
 bool reached1536=false;double cpu=0,maxMove=0,reward=0;uint64_t nodes=0;
};

template<class T>void put(std::ofstream&f,const T&v){f.write(reinterpret_cast<const char*>(&v),sizeof(v));}
template<class T>void get(std::ifstream&f,T&v){f.read(reinterpret_cast<char*>(&v),sizeof(v));}

void saveResume(const std::string&path,int index,int rep,int depth,const ResumeState&s){
 std::string tmp=path+".tmp";std::ofstream f(tmp,std::ios::binary|std::ios::trunc);uint32_t magic=0x35315654,version=1;
 put(f,magic);put(f,version);put(f,index);put(f,rep);put(f,depth);
 f.write(reinterpret_cast<const char*>(s.game.b.data()),s.game.b.size());
 f.write(reinterpret_cast<const char*>(s.game.deck.data()),s.game.deck.size());
 put(f,s.game.left);f.write(reinterpret_cast<const char*>(s.game.next.cards.data()),s.game.next.cards.size());
 f.write(reinterpret_cast<const char*>(s.game.next.p.data()),sizeof(double)*s.game.next.p.size());
 put(f,s.game.next.size);put(f,s.game.next.bonus);put(f,s.game.turns);put(f,s.game.over);
 for(auto v:s.counts)put(f,v);put(f,s.rngState);put(f,s.startTurns);put(f,s.local);put(f,s.reached1536);
 put(f,s.cpu);put(f,s.maxMove);put(f,s.reward);put(f,s.nodes);f.flush();if(!f)throw std::runtime_error("resume write failed");f.close();
 if(std::rename(tmp.c_str(),path.c_str()))throw std::runtime_error("resume rename failed");
}

bool loadResume(const std::string&path,int index,int rep,int depth,ResumeState&s){
 std::ifstream f(path,std::ios::binary);if(!f)return false;uint32_t magic=0,version=0;int i=0,r=0,d=0;
 get(f,magic);get(f,version);get(f,i);get(f,r);get(f,d);if(magic!=0x35315654||version!=1||i!=index||r!=rep||d!=depth)throw std::runtime_error("invalid resume checkpoint");
 f.read(reinterpret_cast<char*>(s.game.b.data()),s.game.b.size());f.read(reinterpret_cast<char*>(s.game.deck.data()),s.game.deck.size());
 get(f,s.game.left);f.read(reinterpret_cast<char*>(s.game.next.cards.data()),s.game.next.cards.size());
 f.read(reinterpret_cast<char*>(s.game.next.p.data()),sizeof(double)*s.game.next.p.size());
 get(f,s.game.next.size);get(f,s.game.next.bonus);get(f,s.game.turns);get(f,s.game.over);
 for(auto&v:s.counts)get(f,v);get(f,s.rngState);get(f,s.startTurns);get(f,s.local);get(f,s.reached1536);
 get(f,s.cpu);get(f,s.maxMove);get(f,s.reward);get(f,s.nodes);if(!f)throw std::runtime_error("truncated resume checkpoint");return true;
}

int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("POOL INDEX REPLICATE DEPTH OUT PROGRESS");
 init();Pool pool=loadEarlyPool(argv[1]);int index=std::stoi(argv[2]),rep=std::stoi(argv[3]),depth=std::stoi(argv[4]);
 if(index<0||index>=int(pool.states.size())||rep<0||rep>=4||(depth!=4&&depth!=5))throw std::runtime_error("arguments");
 Learner model;model.load("training/v7/base.ntd");Search search(model);search.maxNodes=INT_MAX;
 auto snapshot=pool.states[index];uint32_t rolloutSeed=12110001u+uint32_t(index*4+rep);std::string checkpoint=std::string(argv[5])+".ckpt";
 ResumeState state;bool resumed=loadResume(checkpoint,index,rep,depth,state);
 if(!resumed){state.game=snapshot.game;state.counts=snapshot.counts;RNG first{rolloutSeed};for(int i=state.game.left-1;i>0;i--)std::swap(state.game.deck[i],state.game.deck[first.index(i+1)]);state.rngState=first.state;state.startTurns=state.game.turns;state.reached1536=high(state.game.b)>=12;saveResume(checkpoint,index,rep,depth,state);}
 Game&g=state.game;Counts&counts=state.counts;RNG rng{state.rngState};int&startTurns=state.startTurns,&local=state.local;bool&reached1536=state.reached1536;double&cpu=state.cpu,&maxMove=state.maxMove,&reward=state.reward;uint64_t&nodes=state.nodes;
 std::ofstream progress(argv[6],std::ios::app);progress<<"{\"resume\":"<<(resumed?"true":"false")<<",\"moves\":"<<local<<"}"<<std::endl;
 while(!g.over&&high(g.b)<13&&local<6000){
  auto begin=std::clock();auto answer=search.choose(g.b,g.next,counts,depth);double elapsed=double(std::clock()-begin)/CLOCKS_PER_SEC;
  if(search.completed!=depth)throw std::runtime_error("incomplete search");cpu+=elapsed;maxMove=std::max(maxMove,elapsed);nodes+=search.nodes;
  auto moved=move(g.b,answer.direction);if(!moved.size)throw std::runtime_error("illegal action");reward+=moved.reward+g.advance(moved,rng);counts=observe(counts,g.next);local++;reached1536|=high(g.b)>=12;
  state.rngState=rng.state;if(local%10==0)saveResume(checkpoint,index,rep,depth,state);
  if(local%25==0)progress<<"{\"moves\":"<<local<<",\"rank\":"<<high(g.b)<<",\"reached1536\":"<<(reached1536?"true":"false")<<",\"searchCPU\":"<<cpu<<"}"<<std::endl;
 }
 if(!g.over&&high(g.b)<13)throw std::runtime_error("truncated continuation");
 std::ofstream out(argv[5]);out<<std::setprecision(17)<<"{\"index\":"<<index<<",\"replicate\":"<<rep<<",\"depth\":"<<depth<<",\"rolloutSeed\":"<<rolloutSeed<<",\"startTurn\":"<<startTurns<<",\"reached1536\":"<<(reached1536?"true":"false")<<",\"reached3072\":"<<(high(g.b)>=13?"true":"false")<<",\"moves\":"<<local<<",\"finalRank\":"<<high(g.b)<<",\"reward\":"<<reward<<",\"searchCPU\":"<<cpu<<",\"maxMoveCPU\":"<<maxMove<<",\"nodes\":"<<nodes<<"}"<<std::endl;out.close();if(!out)throw std::runtime_error("result write failed");std::remove(checkpoint.c_str());
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
