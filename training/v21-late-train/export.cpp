// Website model: V4 stages 0-2 (unchanged) + V21 stage 3 (S3). Stage 2 retraining was
// not adopted (E1 not significant), per protocol.json.
#include "model4.h"
using namespace v21;
int main(){init();Model4 v4,s3;v4.load("dist/models/ntuple-v4.bin");s3.load("training/v21-late-train/models/S3.bin");
 v4.w[3]=s3.w[3];v4.save("dist/models/ntuple-v21.bin");std::puts("dist/models/ntuple-v21.bin written");}
