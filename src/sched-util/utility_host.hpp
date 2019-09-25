#ifndef UTIL_HOST_H
#define UTIL_HOST_H
#include <stdint.h>
#include <time.h>
#include <iostream>

namespace Util {
    void getTimeGPU(uint64_t *d_targetStartTime);
    uint64_t getCpuTimeNs(void);
    int getHostDeviceTimeOffset(uint64_t *device_ns, uint64_t *host_ns);
    int getGpuTimeScale(double* scale);

    inline uint64_t getCpuTimeNs(void){
        struct timespec time;
        if(clock_gettime(CLOCK_MONOTONIC_RAW, &time)){
            std::cerr << "Error getting CPU time." << std::endl;
            return 0;
        }
        return time.tv_sec * 1000000000 + time.tv_nsec;
    }
}

#endif /* UTIL_HOST */
