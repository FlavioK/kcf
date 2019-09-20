#ifndef UTIL_FUNC_H
#define UTIL_FUNC_H
#include <stdint.h>
#include <cuda.h>
#include <stdint.h>

namespace Util {
    __device__ __inline__ uint64_t getTimeGPU(void){
        uint64_t time;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
        return time;
    }

    __device__ __inline__ unsigned int get_smid(void)
    {
        unsigned int ret;
        asm("mov.u32 %0, %%smid;":"=r"(ret) );
        return ret;
    }

    __device__ __inline__ void spinUntil(const uint64_t endTime){
        if( threadIdx.x == 0){
            while(Util::getTimeGPU() < endTime);
        }
    }

    __device__ __inline__ void logBlockStart(uint64_t *targetTimes){

        uint64_t start_time = Util::getTimeGPU();
        if(threadIdx.x == 0){
            targetTimes[blockIdx.x*2] = start_time;
        }
    }

   // __device__ __inline__ void logBlockStart(uint64_t *targetTimes, unsigned int *smid){

   //     uint64_t start_time = Util::getTimeGPU();
   //     if(threadIdx.x == 0){
   //         targetTimes[blockIdx.x*2] = start_time;
   //         smid[blockIdx.x] = Util::get_smid();
   //     }
   // }

    __device__ __inline__ void logBlockEnd(uint64_t *targetTimes){
        if(threadIdx.x == 0){
            targetTimes[blockIdx.x*2+1] = Util::getTimeGPU();
        }
    }
}

#endif
