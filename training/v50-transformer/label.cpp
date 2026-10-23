// V50: the website AI's root values for arbitrary positions (same search as V46 margins.cpp:
// 6-ply / 0.005, cycle-aware, clamp, float16 residual weights). One input line per position:
//  board[16] ranks, ncards, cards[ncards], bonus(0/1), counts[3], cyclePos, cycleUsed, cycleExcl
// One output line per position: four values ("nan" for an illegal direction).
//  label MODEL RES_LIST IN OUT   (IN "-" = stdin, line by line)
#include "../v38-progress/cgame.h"
#include <climits>
using namespace v38;
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("label MODEL RES_LIST IN OUT");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");
 std::ifstream fin;if(std::string(argv[3])!="-")fin.open(argv[3]);std::istream&in=fin.is_open()?fin:std::cin;
 std::ofstream out(argv[4]);out<<std::setprecision(10);  // IN "-": std::cin, which returns each line as it arrives (ifstream on /dev/stdin waits for a full buffer)
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.005;s.clampLeaf=true;s.cycleAware=true;
 std::string line;
 while(std::getline(in,line)){std::istringstream l(line);Board b{};int x;
  for(int i=0;i<16;i++){l>>x;b[i]=x;}
  Preview p;l>>p.size;for(int i=0;i<p.size;i++){l>>x;p.cards[i]=x;p.p[i]=1.0/p.size;}l>>x;p.bonus=x;
  Counts c;for(auto&v:c)l>>v;Cycle cy;l>>cy.pos>>cy.used>>cy.excl;if(!l)throw std::runtime_error("bad line: "+line);
  s.rootValues.fill(0);s.choose(b,p,c,cy,6);if(s.completed!=6)throw std::runtime_error("incomplete depth");
  for(int d=0;d<4;d++){out<<(d?" ":"");if(move(b,d).size)out<<s.rootValues[d];else out<<"nan";}out<<std::endl;}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
