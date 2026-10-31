// V65 check: the C++ judges on the V64 records (positions + every iteration) -> one line per position: id p4 p5,
// to compare with the Python probabilities (judge-probs.npz).
//  verify JUDGE POSITIONS RUN OUT     (RUN: one V64 runs/c-*.txt chunk; ids index POSITIONS lines)
#include "judge.h"
#include <iostream>
#include <sstream>
#include <iomanip>
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("verify JUDGE POSITIONS RUN OUT");
 v65::Judge J;J.load(argv[1]);
 std::vector<std::string>pos;{std::ifstream f(argv[2]);std::string l;while(std::getline(f,l))pos.push_back(l);}
 std::ifstream run(argv[3]);std::ofstream out(argv[4]);out<<std::setprecision(9);std::string line;
 while(std::getline(run,line)){std::istringstream r(line);int id;r>>id;int dirs[6];std::array<double,4>vals[6];
  for(int d=0;d<6;d++){double ms;r>>dirs[d]>>ms;for(auto&v:vals[d])r>>v;}
  std::istringstream p(pos.at(id));int pid,x;std::array<uint8_t,16>b{};p>>pid;for(auto&c:b){p>>x;c=x;}int n,first=0;p>>n;for(int i=0;i<n;i++){p>>x;if(!i)first=x;}int bonus;p>>bonus;
  out<<id<<' '<<J.prob(4,dirs,vals,b,n,first,bonus)<<' '<<J.prob(5,dirs,vals,b,n,first,bonus)<<'\n';}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
