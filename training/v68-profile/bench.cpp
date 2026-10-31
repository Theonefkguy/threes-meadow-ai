// V68: website search (6-ply / 0.005, depth 6 only as V66) on the V62 positions with the original search and model
// (arm 0) and the exact faster versions of searcht.h / fastmodel.h (arm 1: compile-time maps; 2: plus float16
// residuals; 3: vector-built symmetric boards; 4: plus float16 residuals); arms 1-4 also pack the key with vector
// instructions. Arms are timed back to back in a rotating order, each with a fresh transposition table.
//  bench MODEL RES_LIST POSITIONS FIRST COUNT OUT
//  OUT: one line per position: id, then per arm: ms nodes dir v0 v1 v2 v3
#include "fastmodel.h"
#include <climits>
#include <cstdio>
#include <ctime>
using namespace v68;
template<class S> static Answer depth6(S&s,const Board&b,const Preview&p,Counts c,const Cycle&cy){
 s.stamp++;s.nodes=0;s.completed=0;s.cutoffs=0;typename S::Moves mv;for(int d=0;d<4;d++)mv.m[d]=move(b,d);
 auto a=s.decision(b,mv,p,c,cy,6,1.0,true);s.completed=6;return a;}
template<class S> static void setup(S&s){s.maxNodes=INT_MAX;s.clampLeaf=true;s.cycleAware=true;s.threshold=0.005;}
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("bench MODEL RES_LIST POSITIONS FIRST COUNT OUT");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");
 for(int f=0;f<threes::FEATURES;f++)for(int j=0;j<4;j++)if(M4.m[f][j]!=threes::maps[f][j])throw std::runtime_error("maps differ");
 for(int f=0;f<v29::FEATURES5;f++)for(int j=0;j<5;j++)if(M5.m[f][j]!=v29::maps5[f][j])throw std::runtime_error("maps5 differ");
 for(int f=0;f<threes::FEATURES;f++)for(int j=0;j<4;j++)if(PERMS.p[f&7][threes::patterns[f>>3][j]]!=threes::maps[f][j])throw std::runtime_error("perms differ");
 for(int f=0;f<v29::FEATURES5;f++)for(int j=0;j<5;j++)if(PERMS.p[f&7][v29::patterns5[f>>3][j]]!=v29::maps5[f][j])throw std::runtime_error("perms5 differ");
 FastModel<false> f32;f32.from(m);FastModel<true> f16;f16.from(m);NeonModel<false> n32;n32.from(m);NeonModel<true> n16;n16.from(m);
 for(int s=0;s<4;s++)fprintf(stderr,"stage %d residual %d scale %g\n",s,int(f16.has[s]),f16.scale[s]);
 std::ifstream fin(argv[3]);int first=std::stoi(argv[4]),count=std::stoi(argv[5]);FILE*out=fopen(argv[6],"w");
 Search s0(m);SearchT<FastModel<false>> s1(f32);SearchT<FastModel<true>> s2(f16);SearchT<NeonModel<false>> s3(n32);SearchT<NeonModel<true>> s4(n16);
 setup(s0);setup(s1);setup(s2);setup(s3);setup(s4);
 const int A=5;std::string line;int k=-1;
 while(std::getline(fin,line)){k++;if(k<first)continue;if(k>=first+count)break;
  std::istringstream l(line);int id,x;Board b{};Preview p;Counts c;Cycle cy;l>>id;
  for(int i=0;i<16;i++){l>>x;b[i]=x;}l>>p.size;for(int i=0;i<p.size;i++){l>>x;p.cards[i]=x;p.p[i]=1.0/p.size;}l>>x;p.bonus=x;
  for(auto&v:c)l>>v;l>>cy.pos>>cy.used>>cy.excl;if(!l)throw std::runtime_error("bad line: "+line);
  double ms[A];long nodes[A];int dir[A];std::array<double,4>vals[A];
  for(int r=0;r<A;r++){int j=(r+id)%A;
   auto run=[&](auto&s){auto t0=std::clock();Answer a=depth6(s,b,p,c,cy);ms[j]=1000.0*double(std::clock()-t0)/CLOCKS_PER_SEC;
    nodes[j]=s.nodes;vals[j]=s.rootValues;dir[j]=a.direction;};
   if(j==0)run(s0);else if(j==1)run(s1);else if(j==2)run(s2);else if(j==3)run(s3);else run(s4);}
  fprintf(out,"%d",id);
  for(int j=0;j<A;j++)fprintf(out," %.4f %ld %d %.17g %.17g %.17g %.17g",ms[j],nodes[j],dir[j],vals[j][0],vals[j][1],vals[j][2],vals[j][3]);
  fprintf(out,"\n");}
 fclose(out);
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
