// V45 counterfactual rollouts: from a logged state, play a forced first move, then continue with the
// website search (6-ply / 0.005, cycle-aware, clamp) to 12288 or death. Rollout r of a position uses
// RNG(SEED_BASE + 1000 * position + r) for the reshuffled unseen deck, the forced move's entry lane and
// everything after, so the two arms of a position see the same random stream.
//  rollout MODEL RES_LIST JOBS_JSONL FIRST COUNT ROLLOUTS SEED_BASE OUT
// JOBS_JSONL lines: {"position":P,"state":"<cgame line>","arms":{"ai":D,"calm":D2}}
#include "../v38-progress/cgame.h"
#include <climits>
using namespace v38;
static std::string field(const std::string&l,const std::string&k){auto p=l.find("\""+k+"\":");if(p==std::string::npos)throw std::runtime_error("missing "+k);p+=k.size()+3;
 if(l[p]=='"'){auto q=l.find('"',p+1);return l.substr(p+1,q-p-1);}auto q=l.find_first_of(",}",p);return l.substr(p,q-p);}
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("rollout MODEL RES_LIST JOBS_JSONL FIRST COUNT ROLLOUTS SEED_BASE OUT");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");std::ifstream jf(argv[3]);std::vector<std::string>jobs;std::string l;while(std::getline(jf,l))if(!l.empty())jobs.push_back(l);
 int first=std::stoi(argv[4]),count=std::stoi(argv[5]),R=std::stoi(argv[6]);uint32_t base=std::stoul(argv[7]);std::ofstream out(argv[8],std::ios::app);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.005;s.clampLeaf=true;s.cycleAware=true;
 for(int j=first;j<first+count&&j<int(jobs.size());j++){
  const std::string&J=jobs[j];int pos=std::stoi(field(J,"position"));std::string st=field(J,"state");
  std::string tmp="/dev/null";std::istringstream dummy;
  // parse the state line through readSnaps' format
  std::string fn=std::string(argv[8])+".state";{std::ofstream f(fn);f<<st<<"\n";}auto sn=readSnaps(fn)[0];std::remove(fn.c_str());
  for(const char*arm:{"ai","calm"}){int dir=std::stoi(field(J.substr(J.find("\"arms\"")),arm));
   for(int r=0;r<R;r++){CGame g=sn.g;Counts c=sn.c;Cycle cy=sn.cy;RNG rng{base+uint32_t(1000*pos+r)};
    for(int i=g.left-1;i>0;i--)std::swap(g.deck[i],g.deck[rng.index(i+1)]);
    double start=score(g.b);int moves=0;
    auto p=move(g.b,dir);if(!p.size)throw std::runtime_error("forced move illegal");g.advance(p,rng);moves++;
    if(!g.over){cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);}
    while(!g.over&&moves<30000){auto a=s.choose(g.b,g.next,c,cy,6);if(s.completed!=6)throw std::runtime_error("incomplete depth");
     auto q=move(g.b,a.direction);g.advance(q,rng);moves++;if(!g.over){cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);}}
    out<<"{\"position\":"<<pos<<",\"arm\":\""<<arm<<"\",\"rollout\":"<<r<<",\"success\":"<<(high(g.b)>=FINAL_RANK?"true":"false")<<",\"moves\":"<<moves<<",\"scoreGained\":"<<score(g.b)-start<<"}"<<std::endl;}}}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
