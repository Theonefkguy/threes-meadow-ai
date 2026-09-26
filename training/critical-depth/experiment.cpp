#include "../v7/search.h"
#include <filesystem>
#include <climits>
#include <ctime>
using namespace v2;
namespace fs=std::filesystem;
int blanks(const Board& b){return std::count(b.begin(),b.end(),0);}
void stateOut(std::ostream& o,const Game& g,Counts c){
 o<<std::setprecision(17);for(auto v:g.b)o<<int(v)<<' ';for(auto v:g.deck)o<<int(v)<<' ';
 o<<g.left<<' '<<g.next.size<<' '<<g.next.bonus<<' ';for(auto v:g.next.cards)o<<int(v)<<' ';for(auto v:g.next.p)o<<v<<' ';
 o<<g.turns<<' '<<g.over<<' ';for(auto v:c)o<<v<<' ';o<<'\n';
}
void stateIn(std::istream& in,Game& g,Counts& c){int n;for(auto&v:g.b){in>>n;v=n;}for(auto&v:g.deck){in>>n;v=n;}in>>g.left>>g.next.size>>g.next.bonus;for(auto&v:g.next.cards){in>>n;v=n;}for(auto&v:g.next.p)in>>v;in>>g.turns>>g.over;for(auto&v:c)in>>v;if(!in)throw std::runtime_error("invalid snapshot");}
void trace(std::ostream& o,const Game& g,Counts c,uint32_t rng,int action,int card,int pos,const Game& after){
 o<<"{\"turn\":"<<g.turns<<",\"boardRanks\":[";for(int i=0;i<16;i++)o<<(i?",":"")<<int(g.b[i]);
 o<<"],\"legalActions\":[";bool first=true;for(int d=0;d<4;d++)if(move(g.b,d).size){o<<(first?"":",")<<d;first=false;}
 o<<"],\"deckRanks\":[";for(int i=0;i<g.left;i++)o<<(i?",":"")<<int(g.deck[i]);o<<"],\"counts\":["<<c[0]<<','<<c[1]<<','<<c[2]<<"],\"previewRanks\":[";for(int i=0;i<g.next.size;i++)o<<(i?",":"")<<int(g.next.cards[i]);
 o<<"],\"previewProbabilities\":[";for(int i=0;i<g.next.size;i++)o<<(i?",":"")<<g.next.p[i];
 o<<"],\"rngBefore\":"<<rng<<",\"action\":"<<action<<",\"spawnRank\":"<<card<<",\"spawnPosition\":"<<pos<<",\"afterRanks\":[";for(int i=0;i<16;i++)o<<(i?",":"")<<int(after.b[i]);o<<"],\"over\":"<<(after.over?"true":"false")<<"}\n";
}
Answer choose(Search& s,const Game& g,Counts c,int depth){auto a=s.choose(g.b,g.next,c,depth);if(s.completed!=depth||a.direction<0)throw std::runtime_error("incomplete/illegal search");return a;}
struct Event{Preview hint;int card;double position;};
int resolve(const Preview&p,RNG&r){double u=r.next(),sum=0;for(int i=0;i<p.size;i++){sum+=p.p[i];if(u<sum)return p.cards[i];}return p.cards[p.size-1];}
int main(int argc,char**argv){try{
 if(argc<4)throw std::runtime_error("collect OUT ID SEED BACK GUARD | test OUT ID REP SEED HORIZON");
 init();Learner model;model.load("training/v7/base.ntd");Search search(model);search.maxNodes=INT_MAX;fs::path out=argv[2];fs::create_directories(out);std::string mode=argv[1];int id=std::stoi(argv[3]);
 if(mode=="collect"){
  if(argc!=7)throw std::runtime_error("collect arguments");uint32_t seed=std::stoul(argv[4]);int back=std::stoi(argv[5]),guard=std::stoi(argv[6]);if(back<20||back>30)throw std::runtime_error("back must be 20..30");
  RNG rng{seed};Game g;g.reset(rng);Counts c=opening(g);std::vector<Snapshot> history;std::vector<uint32_t> seeds;std::ofstream tr(out/("trajectory-"+std::to_string(id)+".jsonl"));tr<<std::setprecision(17);
  while(!g.over&&g.turns<guard){history.push_back({g,c});seeds.push_back(rng.state);Game before=g;Counts old=c;uint32_t r=rng.state;auto a=choose(search,g,c,4);auto m=move(g.b,a.direction);g.advance(m,rng);int pos=-1,card=0;for(int e=0;e<m.size;e++)if(g.b[m.entries[e]]!=m.b[m.entries[e]]){pos=m.entries[e];card=g.b[pos];}trace(tr,before,old,r,a.direction,card,pos,g);c=observe(c,g.next);}
  if(!g.over)throw std::runtime_error("collection truncated: no death endpoint");if(history.size()<size_t(back))throw std::runtime_error("trajectory too short");size_t at=history.size()-back;auto s=history[at];if(!legal(s.game.b))throw std::runtime_error("terminal cut");
  std::ofstream snap(out/("snapshot-"+std::to_string(id)+".txt"));stateOut(snap,s.game,s.counts);std::ofstream meta(out/("source-"+std::to_string(id)+".json"));meta<<"{\"id\":"<<id<<",\"seed\":"<<seed<<",\"rngAtCut\":"<<seeds[at]<<",\"deathTurn\":"<<g.turns<<",\"cutTurn\":"<<s.game.turns<<",\"back\":"<<back<<",\"empty\":"<<blanks(s.game.b)<<"}\n";
 }else if(mode=="test"){
  if(argc!=7)throw std::runtime_error("test arguments");int rep=std::stoi(argv[4]);uint32_t seed=std::stoul(argv[5]);int horizon=std::stoi(argv[6]);if(horizon<1)throw std::runtime_error("invalid horizon");
  Game start;Counts counts;std::ifstream input(out/("snapshot-"+std::to_string(id)+".txt"));stateIn(input,start,counts);RNG cards{seed},positions{seed^0x9e3779b9u};Game generator=start;
  for(int i=generator.left-1;i>0;i--)std::swap(generator.deck[i],generator.deck[cards.index(i+1)]);
  std::vector<Event> tape;for(int t=0;t<=horizon;t++){tape.push_back({generator.next,resolve(generator.next,cards),positions.next()});generator.draw(cards);}
  std::string key=std::to_string(id)+"-"+std::to_string(rep);std::ofstream tf(out/("tape-"+key+".txt"));tf<<std::setprecision(17);for(auto&e:tape){tf<<e.card<<' '<<e.position<<' '<<e.hint.size;for(int j=0;j<e.hint.size;j++)tf<<' '<<int(e.hint.cards[j])<<' '<<e.hint.p[j];tf<<'\n';}tf.close();
  for(int depth:{4,5}){Game g=start;Counts c=counts;int steps=0,maxrank=high(g.b),bestEmpty=blanks(g.b);double cpu=0;std::ofstream tr(out/("replay-"+key+"-d"+std::to_string(depth)+".jsonl"));tr<<std::setprecision(17);
   for(;steps<horizon&&!g.over;steps++){Game before=g;Counts old=c;g.next=tape[steps].hint;auto begin=std::clock();auto a=choose(search,g,c,depth);cpu+=double(std::clock()-begin)/CLOCKS_PER_SEC;auto m=move(g.b,a.direction);int pos=m.entries[std::min(m.size-1,int(tape[steps].position*m.size))];g.b=m.b;g.b[pos]=tape[steps].card;g.turns++;g.over=!legal(g.b);g.next=tape[steps+1].hint;c=observe(c,g.next);maxrank=std::max(maxrank,high(g.b));bestEmpty=std::max(bestEmpty,blanks(g.b));trace(tr,before,old,seed,a.direction,tape[steps].card,pos,g);}
   std::string name="result-"+key+"-d"+std::to_string(depth)+".json";std::ofstream f(out/(name+".tmp"));f<<std::setprecision(17)<<"{\"id\":"<<id<<",\"rep\":"<<rep<<",\"depth\":"<<depth<<",\"seed\":"<<seed<<",\"steps\":"<<steps<<",\"maxTile\":"<<(maxrank<3?maxrank:3*(1<<(maxrank-3)))<<",\"escaped\":"<<(!g.over&&steps==horizon?"true":"false")<<",\"censored\":"<<(!g.over?"true":"false")<<",\"bestEmpty\":"<<bestEmpty<<",\"cpuSeconds\":"<<cpu<<"}\n";f.close();fs::rename(out/(name+".tmp"),out/name);
  }
 }else throw std::runtime_error("unknown mode");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
