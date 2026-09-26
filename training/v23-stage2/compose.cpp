// compose OUT STAGE2_SOURCE : base stages 0,1 + stage 2 from STAGE2_SOURCE (4-stage file) + V21 S3 stage 3.
#include "../v21-late-train/model4.h"
using namespace v21;
int main(int argc,char**argv){init();Model4 b,s2,s3;b.load("training/v7/base.ntd");s3.load("training/v21-late-train/models/S3.bin");
 if(std::string(argv[2])!="base"){s2.load(argv[2]);if(s2.w[0]!=b.w[0]||s2.w[1]!=b.w[1])throw std::runtime_error("stage 0/1 changed");b.w[2]=s2.w[2];}
 b.w[3]=s3.w[3];b.save(argv[1]);std::printf("%s written\n",argv[1]);}
