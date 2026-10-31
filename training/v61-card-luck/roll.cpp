// V61 step 2: replay positions many times with the hidden information resampled (same public state: board,
// preview, cards left in this deck, bonus-cycle progress). Two streams: CARDS (normal-card order, bonus timing
// and the actual bonus value) and LAND (landing cell), so a replay's card sequence does not depend on the play;
// the first SCRIPT cards after the preview are recorded even when the game ends earlier.
//  roll MODEL RES_LIST POSITIONS FIRST COUNT REPLAYS HORIZON OUT
//  POSITIONS: one line per position: id board[16] ncards cards[ncards] bonus counts[3] cyclePos cycleUsed cycleExcl
#include "../v38-progress/cgame.h"
#include <climits>
using namespace v38;
constexpr int SCRIPT=20;
struct Sim{
 Board b{};Preview next{};int actual=0;std::array<uint8_t,12>deck{};int left=0,w=0,slot=-1;RNG cr{0};
 void refill(){for(int i=0;i<12;i++)deck[i]=i/4+1;for(int i=11;i>0;i--)std::swap(deck[i],deck[cr.index(i+1)]);left=12;}
 int normal(){if(!left)refill();return deck[--left];}
 void draw(){                                              // the website rule (CGame::draw) with the window position w
  int pos=w;w=(w+1)%21;if(pos==0)slot=cr.index(21);
  auto list=bonuses(high(b));
  if(pos==slot&&!list.empty()){double r=cr.next(),sum=0;next=list.back().first;for(auto&item:list){sum+=item.second;if(r<sum){next=item.first;break;}}
   double q=cr.next(),s=0;actual=next.cards[next.size-1];for(int i=0;i<next.size;i++){s+=next.p[i];if(q<s){actual=next.cards[i];break;}}return;}
  next={};next.size=1;next.cards[0]=normal();next.p[0]=1;actual=next.cards[0];}
};
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("roll MODEL RES_LIST POSITIONS FIRST COUNT REPLAYS HORIZON OUT");
 v31::init();ModelP m;m.load(argv[1],argv[2],"-");
 std::ifstream fin(argv[3]);int first=std::stoi(argv[4]),count=std::stoi(argv[5]),R=std::stoi(argv[6]),H=std::stoi(argv[7]);
 std::ofstream out(argv[8],std::ios::app);
 Search s(m);s.maxNodes=INT_MAX;s.threshold=0.005;s.clampLeaf=true;s.cycleAware=true;
 std::string line;int k=-1;
 while(std::getline(fin,line)){k++;if(k<first)continue;if(k>=first+count)break;
  std::istringstream l(line);int id,x;Board b0{};Preview p0;Counts c0;Cycle cy0;l>>id;
  for(int i=0;i<16;i++){l>>x;b0[i]=x;}l>>p0.size;for(int i=0;i<p0.size;i++){l>>x;p0.cards[i]=x;p0.p[i]=1.0/p0.size;}l>>x;p0.bonus=x;
  for(auto&v:c0)l>>v;l>>cy0.pos>>cy0.used>>cy0.excl;if(!l)throw std::runtime_error("bad line: "+line);
  for(int r=0;r<R;r++){
   Sim g;g.cr=RNG{uint32_t(610000000u+uint32_t(id)*1000u+uint32_t(r))};RNG land{uint32_t(910000000u+uint32_t(id)*1000u+uint32_t(r))};
   g.b=b0;g.next=p0;Counts c=c0;Cycle cy=cy0;
   int n=0;for(int t=0;t<3;t++){for(int j=0;j<c0[t];j++)g.deck[n++]=t+1;}g.left=n;
   for(int i=n-1;i>0;i--)std::swap(g.deck[i],g.deck[g.cr.index(i+1)]);
   g.w=cy0.pos;
   if(cy0.pos>0){if(cy0.used)g.slot=-1;else{int fut=21-cy0.pos,imp=cy0.pos-cy0.excl,q=g.cr.index(fut+imp);g.slot=q<fut?cy0.pos+q:-1;}}
   if(p0.size>1){double q=g.cr.next(),sum=0;g.actual=p0.cards[p0.size-1];for(int i=0;i<p0.size;i++){sum+=p0.p[i];if(q<sum){g.actual=p0.cards[i];break;}}}else g.actual=p0.cards[0];
   std::vector<int>script;bool dead=false,won=false;int moves=0;
   while(moves<H){
    auto a=s.choose(g.b,g.next,c,cy,6);if(s.completed!=6)throw std::runtime_error("incomplete");
    auto pr=move(g.b,a.direction);if(!pr.size)throw std::runtime_error("illegal");moves++;
    if(high(pr.b)>=FINAL_RANK){won=true;break;}
    g.b=pr.b;g.b[pr.entries[land.index(pr.size)]]=g.actual;
    if(!legal(g.b)){dead=true;break;}
    g.draw();script.push_back(g.actual);
    cy=cy.after(g.next.bonus,!bonuses(high(g.b)).empty());c=observe(c,g.next);
    {Counts d{};for(int i=0;i<g.left;i++)d[g.deck[i]-1]++;                    // the public state the search sees matches the hidden one
     if(g.w!=cy.pos||(d!=c&&!(g.left==0&&c==Counts{0,0,0}))||(g.next.bonus&&g.slot!=(g.w+20)%21))throw std::runtime_error("state mismatch");}}
   while(int(script.size())<SCRIPT){g.draw();script.push_back(g.actual);}  // the rest of the card sequence (independent of the play)
   out<<"{\"pos\":"<<id<<",\"r\":"<<r<<",\"dead\":"<<(dead?"true":"false")<<",\"won\":"<<(won?"true":"false")<<",\"moves\":"<<moves<<",\"script\":[";
   for(int i=0;i<SCRIPT;i++)out<<(i?",":"")<<script[i];out<<"]}\n";}
  out.flush();}
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
