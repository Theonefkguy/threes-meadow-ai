// NEW = stage 0/1 from base, stage 2 from S2, stage 3 from S3 (protocol.json).
#include "model4.h"
using namespace v21;
int main(){init();Model4 b,s2,s3,n;b.load("training/v7/base.ntd");s2.load("training/v21-late-train/models/S2.bin");s3.load("training/v21-late-train/models/S3.bin");
 // S2 must leave stages 0,1,3 untouched; S3 must leave 0,1,2 untouched.
 if(s2.w[0]!=b.w[0]||s2.w[1]!=b.w[1]||s2.w[3]!=b.w[3])throw std::runtime_error("S2 changed other stages");
 if(s3.w[0]!=b.w[0]||s3.w[1]!=b.w[1]||s3.w[2]!=b.w[2])throw std::runtime_error("S3 changed other stages");
 n.w={b.w[0],b.w[1],s2.w[2],s3.w[3]};n.save("training/v21-late-train/models/NEW.bin");
 double d2=0,d3=0;for(int i=0;i<WEIGHTS;i++){d2+=std::fabs(n.w[2][i]-b.w[2][i]);d3+=std::fabs(n.w[3][i]-b.w[2][i]);}
 std::printf("NEW written; mean |dw| stage2 %.3f stage3 %.3f\n",d2/WEIGHTS,d3/WEIGHTS);}
