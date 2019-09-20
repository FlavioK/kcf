#include "utility_func.cuh"
#include "utility_host.hpp"
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

// Prints a message and returns zero if the given value is not cudaSuccess
#define CheckCUDAError(val) (InternalCheckCUDAError((val), #val, __FILE__, __LINE__))

// Called internally by CheckCUDAError
static inline int InternalCheckCUDAError(cudaError_t result, const char *fn,
        const char *file, int line) {
    if (result == cudaSuccess) return 0;
    printf("CUDA error %d in %s, line %d (%s): %s\n", (int) result, file, line,
            fn, cudaGetErrorString(result));
    return -1;
}

static __global__ void getTimeInternal(uint64_t *targetTime, uint64_t startTimeOffsetNs) {
    if(threadIdx.x == 0){
        *targetTime = Util::getTimeGPU() + startTimeOffsetNs;
    }
    __syncthreads();
}

void Util::getStartTimeGPU(uint64_t *d_targetStartTime, uint64_t startTimeOffsetNs){
        getTimeInternal<<<1,1>>>(d_targetStartTime, startTimeOffsetNs);

        if (CheckCUDAError(cudaDeviceSynchronize())) perror("Could not synchronize device\n");
}


int Util::getHostDeviceTimeOffset(int deviceId, uint64_t *device_ns, double *host_ns){
    uint64_t *time_d;

    if (CheckCUDAError(cudaSetDevice(deviceId))) return -1;
    if (CheckCUDAError(cudaMalloc((void**)&time_d, sizeof(*time_d)))) return -1;

    // Warm-up
    getTimeInternal<<<1,1>>>(time_d, 0.0);
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    // Do Measurement
    getTimeInternal<<<1,1>>>(time_d, 0.0);
    *host_ns = Util::getCpuTimeNs();
    if (CheckCUDAError(cudaMemcpy(device_ns, time_d, sizeof(*device_ns), cudaMemcpyDeviceToHost))) {
        cudaFree(time_d);
        return -1;
    }
    cudaFree(time_d);
    return 0;
}

#define SPIN_DURATION (1000000000)

__global__ void spinKernel(uint64_t spin_duration) {
    uint64_t start_time = Util::getTimeGPU();
    while ((Util::getTimeGPU() - start_time) < spin_duration) {
        continue;
    }
}

int Util::getGpuTimeScale(int deviceId, double* scale){
    double cpuStart, cpuStop;
    if (CheckCUDAError(cudaSetDevice(deviceId))) return -1;

    // Warm-up
    spinKernel<<<1,1>>>(1000);
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;
    cpuStart = Util::getCpuTimeNs();
    spinKernel<<<1, 1>>>(SPIN_DURATION);
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;
    cpuStop = Util::getCpuTimeNs();
    *scale = (double)(cpuStop-cpuStart)/(double)SPIN_DURATION;
    return 0;
}
