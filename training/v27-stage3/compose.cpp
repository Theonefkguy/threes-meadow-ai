// compose BASE STAGE3_SOURCE OUT : BASE (4-stage website model) with stage 3 replaced by
// stage 3 of STAGE3_SOURCE; checks that stages 0-2 of the source were not changed.
#include "../v21-late-train/model4.h"
using namespace v21;
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("compose BASE STAGE3_SOURCE OUT");
 init();Model4 b,s;b.load(argv[1]);s.load(argv[2]);
 for(int i=0;i<3;i++)if(s.w[i]!=b.w[i])throw std::runtime_error("stage 0-2 changed");
 b.w[3]=s.w[3];b.save(argv[3]);std::printf("%s written\n",argv[3]);
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
