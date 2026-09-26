#include "ntuple.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
using namespace threes;
namespace fs=std::filesystem;
struct Reservoir{
 std::vector<Game> games;uint64_t seen=0;
 void add(const Game& g,RNG& rng){if(g.over)return;seen++;if(games.size()<5000)games.push_back(g);else{auto j=uint64_t(rng.next()*seen);if(j<games.size())games[j]=g;}}
};
int main(int argc,char**argv){
 try{
 init();std::string out=argc>1?argv[1]:"training/run-v1";int counts[3]={100000,40000,20000};
 for(int i=0;i<3;i++)if(argc>i+2)counts[i]=std::stoi(argv[i+2]);
 uint32_t seed=argc>5?std::stoul(argv[5]):20260921;fs::create_directories(out);
 int policyDepth=argc>7?std::stoi(argv[7]):1;if(policyDepth<1||policyDepth>2)throw std::runtime_error("policy depth must be 1 or 2");
 Model model;if(argc>6)model.load(argv[6]);RNG rng{seed},sampleRng{seed^0x9e3779b9};Reservoir pools[3];
 std::ofstream log(out+"/progress.jsonl"),snapshots(out+"/snapshots.jsonl");
 auto start=std::chrono::steady_clock::now();uint64_t totalSteps=0;int truncated=0;
 for(int s=0;s<3;s++){
  if(counts[s]==0){if(s)model.w[s]=model.w[s-1];model.save(out+"/stage-"+std::to_string(s)+".ntd");continue;}
  if(s){if(pools[s].games.empty())throw std::runtime_error("No real snapshots for stage "+std::to_string(s));model.w[s]=model.w[s-1];}
  double sum=0,steps=0;int reach[3]={0,0,0},best=0;
  for(int episode=1;episode<=counts[s];episode++){
   Game game;if(!s)game.reset(rng);else game=pools[s].games[sampleRng.index(pools[s].games.size())];
   if(game.over||!legal(game.b))throw std::runtime_error("curriculum start must be playable");
   int originalStage=stage(game.b),localSteps=0;Board previous{};bool hasPrevious=false;double spawnReward=0;
   double fraction=double(episode)/counts[s];double alpha=fraction<.6?.05:fraction<.85?.02:.008;
   while(!game.over&&localSteps<6000){
    double value=-1e100;Projection chosen;int dir=-1;
    // Optional two-ply policy improvement uses only the public current hint.
    // In both cases the TD target is the actually chosen next afterstate.
    for(int d=0;d<4;d++){
     auto m=move(game.b,d);if(!m.size)continue;double q=m.reward;
     if(policyDepth==1)q+=model.value(m.b,s);
     else{
      double future=0;
      for(int e=0;e<m.size;e++)for(int c=0;c<game.next.size;c++){
       Board spawned=m.b;int card=game.next.cards[c];spawned[m.entries[e]]=card;double best=-1e100;
       for(int d2=0;d2<4;d2++){auto next=move(spawned,d2);if(next.size)best=std::max(best,next.reward+model.value(next.b,s));}
       future+=game.next.p[c]*(points(card)+(best==-1e100?0:best));
      }
      q+=future/m.size;
     }
     if(q>value){value=q;chosen=m;dir=d;}
    }
    if(dir<0)throw std::runtime_error("nonterminal with no action");
    if(hasPrevious)model.update(previous,s,spawnReward+chosen.reward+model.value(chosen.b,s),alpha);
    previous=chosen.b;hasPrevious=true;spawnReward=game.advance(chosen,rng);localSteps++;totalSteps++;
    int nowStage=stage(game.b);
    if(nowStage>originalStage){
     for(int k=originalStage+1;k<=nowStage;k++)pools[k].add(game,sampleRng);
     originalStage=nowStage;
    }
   }
   if(game.over&&hasPrevious)model.update(previous,s,spawnReward,alpha);else if(!game.over)truncated++;
   sum+=score(game.b);steps+=localSteps;int hi=high(game.b);best=std::max(best,hi);for(int k=0;k<3;k++)reach[k]+=hi>=12+k;
   if(episode%5000==0||episode==counts[s]){
    int batch=episode%5000?episode%5000:5000;
    double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ostringstream line;line<<"{\"stage\":"<<s<<",\"episode\":"<<episode<<",\"batch\":"<<batch<<",\"meanFinalScore\":"<<sum/batch<<",\"meanSteps\":"<<steps/batch<<",\"reached1536\":"<<reach[0]<<",\"reached3072\":"<<reach[1]<<",\"reached6144\":"<<reach[2]<<",\"maxRank\":"<<best<<",\"pool1536\":"<<pools[1].games.size()<<",\"pool3072\":"<<pools[2].games.size()<<",\"totalSteps\":"<<totalSteps<<",\"seconds\":"<<secs<<",\"truncated\":"<<truncated<<"}";
    log<<line.str()<<std::endl;std::cout<<line.str()<<std::endl;sum=steps=0;best=0;std::fill(reach,reach+3,0);
    model.save(out+"/checkpoint.ntd");
   }
  }
  model.save(out+"/stage-"+std::to_string(s)+".ntd");
 }
 // Persist the actual curriculum states; hidden order is simulator-only.
 for(int s=1;s<3;s++)for(auto&g:pools[s].games){
  snapshots<<"{\"stage\":"<<s<<",\"ranks\":[";for(int i=0;i<16;i++){if(i)snapshots<<',';snapshots<<int(g.b[i]);}
  snapshots<<"],\"deck\":[";for(int i=0;i<g.left;i++){if(i)snapshots<<',';snapshots<<int(g.deck[i]);}
  snapshots<<"],\"cards\":[";for(int i=0;i<g.next.size;i++){if(i)snapshots<<',';snapshots<<int(g.next.cards[i]);}
  snapshots<<"],\"weights\":[";for(int i=0;i<g.next.size;i++){if(i)snapshots<<',';snapshots<<g.next.p[i];}snapshots<<"],\"turns\":"<<g.turns<<"}\n";
 }
 std::ofstream config(out+"/config.json");config<<"{\"seed\":"<<seed<<",\"episodes\":["<<counts[0]<<','<<counts[1]<<','<<counts[2]<<"],\"totalSteps\":"<<totalSteps<<",\"truncated\":"<<truncated<<",\"pool1536\":"<<pools[1].games.size()<<",\"pool3072\":"<<pools[2].games.size()<<",\"policyDepth\":"<<policyDepth<<",\"initial\":\""<<(argc>6?argv[6]:"zero")<<"\",\"method\":\"afterstate TD(0), gamma=1, alpha=.05/.02/.008\"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
