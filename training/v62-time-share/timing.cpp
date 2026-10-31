// V62: CPU time of one move's search on sampled website-AI positions, for the website setting (6-ply / 0.005)
// and the cheap setting (5-ply / 0.01), measured back to back on the same position.
//  timing MODEL RES_LIST POSITIONS FIRST COUNT OUT
//  POSITIONS: as V61 positions.txt (id board[16] ncards cards[ncards] bonus counts[3] cyclePos cycleUsed cycleExcl)
//  OUT: one line per position: id ms6 ms5 dir6 dir5 nodes6 nodes5
#include "../v38-progress/cgame.h"
#include <climits>
#include <ctime>
using namespace v38;
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("timing MODEL RES_LIST POSITIONS FIRST COUNT OUT");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");
 std::ifstream fin(argv[3]);int first=std::stoi(argv[4]),count=std::stoi(argv[5]);std::ofstream out(argv[6]);
 Search s(m);s.maxNodes=INT_MAX;s.clampLeaf=true;s.cycleAware=true;
 std::string line;int k=-1;
 while(std::getline(fin,line)){k++;if(k<first)continue;if(k>=first+count)break;
  std::istringstream l(line);int id,x;Board b{};Preview p;Counts c;Cycle cy;l>>id;
  for(int i=0;i<16;i++){l>>x;b[i]=x;}l>>p.size;for(int i=0;i<p.size;i++){l>>x;p.cards[i]=x;p.p[i]=1.0/p.size;}l>>x;p.bonus=x;
  for(auto&v:c)l>>v;l>>cy.pos>>cy.used>>cy.excl;if(!l)throw std::runtime_error("bad line: "+line);
  double ms[2];int dir[2];long nodes[2];
  for(int j=0;j<2;j++){int depth=j?5:6;s.threshold=j?0.01:0.005;
   auto t0=std::clock();auto a=s.choose(b,p,c,cy,depth);ms[j]=1000.0*double(std::clock()-t0)/CLOCKS_PER_SEC;
   if(s.completed!=depth)throw std::runtime_error("incomplete");dir[j]=a.direction;nodes[j]=s.nodes;}
  out<<id<<' '<<ms[0]<<' '<<ms[1]<<' '<<dir[0]<<' '<<dir[1]<<' '<<nodes[0]<<' '<<nodes[1]<<'\n';}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
