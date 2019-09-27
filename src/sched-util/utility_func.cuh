#ifndef UTIL_FUNC_H
#define UTIL_FUNC_H
#include <stdint.h>
#include <cuda.h>
#include <stdint.h>

namespace Util {
    __device__ __inline__ uint64_t getTimeGPU(void){
        // Due to a bug in CUDA's 64-bit globaltimer, the lower 32 bits can wrap
        // around after the upper bits have already been read. Work around this by
        // reading the high bits a second time. Use the second value to detect a
        // rollover, and set the lower bits of the 64-bit "timer reading" to 0, which
        // would be valid, it's passed over during the duration of the reading. If no
        // rollover occurred, just return the initial reading.
        volatile uint64_t first_reading;
        volatile uint32_t second_reading;
        uint32_t high_bits_first;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
        high_bits_first = first_reading >> 32;
        asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
        if (high_bits_first == second_reading) {
          return first_reading;
        }
        // Return the value with the updated high bits, but the low bits set to 0.
        return ((uint64_t) second_reading) << 32;
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

#ifdef USE_KERNEL_SCHED
    __device__ __inline__ void logBlockStart(uint64_t *targetTimes, uint64_t *startTime, uint64_t *offsets){
        // TODO: Use spinUntil to spin until this block is allowed to start processing
        uint64_t start_time = *startTime+offsets[blockIdx.x];
        Util::spinUntil(start_time);
        start_time = Util::getTimeGPU();
        if(threadIdx.x == 0){
            targetTimes[blockIdx.x*2] = start_time;
        }
    }
#endif

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
