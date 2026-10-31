// V64: the website search (6-ply / 0.005, cycle-aware, iterative deepening) on sampled positions, recording every
// iteration: after depth d = 1..6, the chosen direction, the four root values and the CPU time so far. The loop is
// Search::choose written out (same stamp, same calls), so depth 6 is exactly the website move.
//  cascade MODEL RES_LIST POSITIONS FIRST COUNT OUT
//  POSITIONS: as V61 positions.txt (id board[16] ncards cards[ncards] bonus counts[3] cyclePos cycleUsed cycleExcl)
//  OUT: one line per position: id, then for d = 1..6: dir ms v0 v1 v2 v3 (illegal directions -1e300)
#include "../v38-progress/cgame.h"
#include <climits>
#include <ctime>
using namespace v38;
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("cascade MODEL RES_LIST POSITIONS FIRST COUNT OUT");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");
 std::ifstream fin(argv[3]);int first=std::stoi(argv[4]),count=std::stoi(argv[5]);std::ofstream out(argv[6]);out<<std::setprecision(10);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.005;s.clampLeaf=true;s.cycleAware=true;
 std::string line;int k=-1;
 while(std::getline(fin,line)){k++;if(k<first)continue;if(k>=first+count)break;
  std::istringstream l(line);int id,x;Board b{};Preview p;Counts c;Cycle cy;l>>id;
  for(int i=0;i<16;i++){l>>x;b[i]=x;}l>>p.size;for(int i=0;i<p.size;i++){l>>x;p.cards[i]=x;p.p[i]=1.0/p.size;}l>>x;p.bonus=x;
  for(auto&v:c)l>>v;l>>cy.pos>>cy.used>>cy.excl;if(!l)throw std::runtime_error("bad line: "+line);
  auto t0=std::clock();
  s.stamp++;s.nodes=0;s.completed=0;s.cutoffs=0;
  Search::Moves mv;for(int d=0;d<4;d++)mv.m[d]=move(b,d);
  out<<id;
  for(int depth=1;depth<=6;depth++){auto a=s.decision(b,mv,p,c,cy,depth,1.0,true);s.completed=depth;
   double ms=1000.0*double(std::clock()-t0)/CLOCKS_PER_SEC;
   out<<' '<<a.direction<<' '<<ms;for(double v:s.rootValues)out<<' '<<v;}
  out<<'\n';}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
