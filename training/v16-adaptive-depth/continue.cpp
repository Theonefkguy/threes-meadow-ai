#include "../v7/search.h"
#include "../v7/pool.h"
#include <climits>
#include <ctime>
#include <cstdio>
using namespace v2;

int emptyCells(const Board&b){int n=0;for(auto v:b)n+=v==0;return n;}
struct ResumeState{Game game;Counts counts{};uint32_t rngState=0;int startTurns=0,local=0;bool reached1536=false;double cpu=0,maxMove=0,reward=0;uint64_t nodes=0;std::array<int,6>used{};};
template<class T>void put(std::ofstream&f,const T&v){f.write(reinterpret_cast<const char*>(&v),sizeof(v));}
template<class T>void get(std::ifstream&f,T&v){f.read(reinterpret_cast<char*>(&v),sizeof(v));}
void saveResume(const std::string&path,int index,int rep,int mode,const ResumeState&s){std::string tmp=path+".tmp";std::ofstream f(tmp,std::ios::binary|std::ios::trunc);uint32_t magic=0x36315654,version=1;put(f,magic);put(f,version);put(f,index);put(f,rep);put(f,mode);f.write((char*)s.game.b.data(),s.game.b.size());f.write((char*)s.game.deck.data(),s.game.deck.size());put(f,s.game.left);f.write((char*)s.game.next.cards.data(),s.game.next.cards.size());f.write((char*)s.game.next.p.data(),sizeof(double)*s.game.next.p.size());put(f,s.game.next.size);put(f,s.game.next.bonus);put(f,s.game.turns);put(f,s.game.over);for(auto v:s.counts)put(f,v);put(f,s.rngState);put(f,s.startTurns);put(f,s.local);put(f,s.reached1536);put(f,s.cpu);put(f,s.maxMove);put(f,s.reward);put(f,s.nodes);for(auto v:s.used)put(f,v);f.flush();if(!f)throw std::runtime_error("resume write failed");f.close();if(std::rename(tmp.c_str(),path.c_str()))throw std::runtime_error("resume rename failed");}
bool loadResume(const std::string&path,int index,int rep,int mode,ResumeState&s){std::ifstream f(path,std::ios::binary);if(!f)return false;uint32_t magic=0,version=0;int i=0,r=0,m=0;get(f,magic);get(f,version);get(f,i);get(f,r);get(f,m);if(magic!=0x36315654||version!=1||i!=index||r!=rep||m!=mode)throw std::runtime_error("invalid resume checkpoint");f.read((char*)s.game.b.data(),s.game.b.size());f.read((char*)s.game.deck.data(),s.game.deck.size());get(f,s.game.left);f.read((char*)s.game.next.cards.data(),s.game.next.cards.size());f.read((char*)s.game.next.p.data(),sizeof(double)*s.game.next.p.size());get(f,s.game.next.size);get(f,s.game.next.bonus);get(f,s.game.turns);get(f,s.game.over);for(auto&v:s.counts)get(f,v);get(f,s.rngState);get(f,s.startTurns);get(f,s.local);get(f,s.reached1536);get(f,s.cpu);get(f,s.maxMove);get(f,s.reward);get(f,s.nodes);for(auto&v:s.used)get(f,v);if(!f)throw std::runtime_error("truncated resume checkpoint");return true;}

struct Choice{Answer answer;int depth=0,nodes=0;};
Choice chooseAdaptive(const Learner&model,const Game&g,Counts counts){
 Search s3(model);s3.maxNodes=INT_MAX;auto a3=s3.choose(g.b,g.next,counts,3);int empty=emptyCells(g.b);
 if(empty>=6)return {a3,3,s3.nodes};
 Search s4(model);s4.maxNodes=INT_MAX;auto a4=s4.choose(g.b,g.next,counts,4);
 if(empty>=4)return {a4,4,s3.nodes+s4.nodes};
 Search s5(model);s5.maxNodes=INT_MAX;auto a5=s5.choose(g.b,g.next,counts,5);return {a5,5,s3.nodes+s4.nodes+s5.nodes};
}

int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("POOL INDEX REPLICATE MODE OUT PROGRESS");init();Pool pool=loadEarlyPool(argv[1]);int index=std::stoi(argv[2]),rep=std::stoi(argv[3]),mode=std::stoi(argv[4]);if(index<0||index>=int(pool.states.size())||rep<0||rep>=2||(mode!=0&&mode!=5))throw std::runtime_error("arguments");Learner model;model.load("training/v7/base.ntd");
 auto snapshot=pool.states[index];uint32_t rolloutSeed=12110001u+uint32_t(index*4+rep);std::string checkpoint=std::string(argv[5])+".ckpt";ResumeState state;bool resumed=loadResume(checkpoint,index,rep,mode,state);if(!resumed){state.game=snapshot.game;state.counts=snapshot.counts;RNG first{rolloutSeed};for(int i=state.game.left-1;i>0;i--)std::swap(state.game.deck[i],state.game.deck[first.index(i+1)]);state.rngState=first.state;state.startTurns=state.game.turns;state.reached1536=high(state.game.b)>=12;saveResume(checkpoint,index,rep,mode,state);}
 Game&g=state.game;Counts&counts=state.counts;RNG rng{state.rngState};std::ofstream progress(argv[6],std::ios::app);progress<<"{\"resume\":"<<(resumed?"true":"false")<<",\"moves\":"<<state.local<<"}\n";
 while(!g.over&&high(g.b)<13&&state.local<6000){auto begin=std::clock();Choice c;if(mode==0)c=chooseAdaptive(model,g,counts);else{Search s(model);s.maxNodes=INT_MAX;c={s.choose(g.b,g.next,counts,5),5,s.nodes};if(s.completed!=5)throw std::runtime_error("incomplete search");}double elapsed=double(std::clock()-begin)/CLOCKS_PER_SEC;state.cpu+=elapsed;state.maxMove=std::max(state.maxMove,elapsed);state.nodes+=c.nodes;state.used[c.depth]++;auto moved=move(g.b,c.answer.direction);if(!moved.size)throw std::runtime_error("illegal action");state.reward+=moved.reward+g.advance(moved,rng);counts=observe(counts,g.next);state.local++;state.reached1536|=high(g.b)>=12;state.rngState=rng.state;if(state.local%10==0)saveResume(checkpoint,index,rep,mode,state);if(state.local%25==0)progress<<"{\"moves\":"<<state.local<<",\"rank\":"<<high(g.b)<<",\"d3\":"<<state.used[3]<<",\"d4\":"<<state.used[4]<<",\"d5\":"<<state.used[5]<<",\"searchCPU\":"<<state.cpu<<"}\n";}
 if(!g.over&&high(g.b)<13)throw std::runtime_error("truncated continuation");std::ofstream out(argv[5]);out<<std::setprecision(17)<<"{\"index\":"<<index<<",\"replicate\":"<<rep<<",\"mode\":"<<mode<<",\"rolloutSeed\":"<<rolloutSeed<<",\"startTurn\":"<<state.startTurns<<",\"reached1536\":"<<(state.reached1536?"true":"false")<<",\"reached3072\":"<<(high(g.b)>=13?"true":"false")<<",\"moves\":"<<state.local<<",\"finalRank\":"<<high(g.b)<<",\"reward\":"<<state.reward<<",\"searchCPU\":"<<state.cpu<<",\"maxMoveCPU\":"<<state.maxMove<<",\"nodes\":"<<state.nodes<<",\"depth3Moves\":"<<state.used[3]<<",\"depth4Moves\":"<<state.used[4]<<",\"depth5Moves\":"<<state.used[5]<<"}\n";out.close();if(!out)throw std::runtime_error("result write failed");std::remove(checkpoint.c_str());
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
