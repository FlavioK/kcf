#ifndef UTIL_HOST_H
#define UTIL_HOST_H
#include <stdint.h>

namespace Util {
    void getStartTimeGPU(uint64_t *d_targetStartTime, uint64_t startTimeOffsetN);
    uint64_t getCpuTimeNs(void);
    int getHostDeviceTimeOffset(int deviceId, uint64_t *device_ns, double *host_ns);
    int getGpuTimeScale(int deviceId, double* scale);
}

#endif /* UTIL_HOST */
