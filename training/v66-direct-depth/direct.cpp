// V66: does the website search (6-ply / 0.005, cycle-aware) need its iterative deepening? Each position is
// searched with the iterations started at depth 1 (the current Search::choose), 3, 4, 5 or 6 (depth 6 only),
// always ending at depth 6 with a fresh transposition table; arms are timed back to back in a rotating order.
//  direct MODEL RES_LIST POSITIONS FIRST COUNT OUT
//  POSITIONS: as V61/V62 positions.txt
//  OUT: one line per position: id, then per start depth (1 3 4 5 6): ms nodes dir v0 v1 v2 v3
#include "../v38-progress/cgame.h"
#include <climits>
#include <cstdio>
#include <ctime>
using namespace v38;
static Answer from(Search&s,const Board&b,const Preview&p,Counts c,const Cycle&cy,int start){
 s.stamp++;s.nodes=0;s.completed=0;s.cutoffs=0;Answer a;
 Search::Moves mv;for(int d=0;d<4;d++)mv.m[d]=move(b,d);
 for(int depth=start;depth<=6;depth++){a=s.decision(b,mv,p,c,cy,depth,1.0,true);s.completed=depth;}
 return a;}
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("direct MODEL RES_LIST POSITIONS FIRST COUNT OUT");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");
 std::ifstream fin(argv[3]);int first=std::stoi(argv[4]),count=std::stoi(argv[5]);FILE*out=fopen(argv[6],"w");
 Search s(m);s.maxNodes=INT_MAX;s.clampLeaf=true;s.cycleAware=true;s.threshold=0.005;
 const int starts[5]={1,3,4,5,6};
 std::string line;int k=-1;
 while(std::getline(fin,line)){k++;if(k<first)continue;if(k>=first+count)break;
  std::istringstream l(line);int id,x;Board b{};Preview p;Counts c;Cycle cy;l>>id;
  for(int i=0;i<16;i++){l>>x;b[i]=x;}l>>p.size;for(int i=0;i<p.size;i++){l>>x;p.cards[i]=x;p.p[i]=1.0/p.size;}l>>x;p.bonus=x;
  for(auto&v:c)l>>v;l>>cy.pos>>cy.used>>cy.excl;if(!l)throw std::runtime_error("bad line: "+line);
  double ms[5];long nodes[5];int dir[5];std::array<double,4>vals[5];
  for(int r=0;r<5;r++){int j=(r+id)%5;
   auto t0=std::clock();auto a=from(s,b,p,c,cy,starts[j]);ms[j]=1000.0*double(std::clock()-t0)/CLOCKS_PER_SEC;
   nodes[j]=s.nodes;dir[j]=a.direction;vals[j]=s.rootValues;}
  fprintf(out,"%d",id);
  for(int j=0;j<5;j++)fprintf(out," %.4f %ld %d %.17g %.17g %.17g %.17g",ms[j],nodes[j],dir[j],vals[j][0],vals[j][1],vals[j][2],vals[j][3]);
  fprintf(out,"\n");}
 fclose(out);
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
