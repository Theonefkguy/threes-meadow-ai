#pragma once
#include <time.h>
inline double cpuNow(){timespec t;clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return double(t.tv_sec)+t.tv_nsec*1e-9;}
