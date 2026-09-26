// Website model V23: V4 stages 0-1, V23 arm A stage 2, V21 S3 stage 3 (4-stage file).
#include "../v21-late-train/model4.h"
using namespace v21;
int main(){init();Model4 v4,a,s3;v4.load("dist/models/ntuple-v4.bin");a.load("training/v23-stage2/models/A-train.bin");s3.load("training/v21-late-train/models/S3.bin");
 v4.w[2]=a.w[2];v4.w[3]=s3.w[3];v4.save("dist/models/ntuple-v23.bin");std::puts("dist/models/ntuple-v23.bin written");}
