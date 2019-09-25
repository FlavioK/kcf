#ifndef PROFCUDA_H
#define PROFCUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_error_check.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#ifdef OPENMP
#include <omp.h>
#endif

class ProfCUDA
{
public:
    // Must be continuous otherwise this class fails
    enum Kernel : uint
    {
       KER_XF_SQR_NORM=0,
       KER_YF_SQR_NORM,
       KER_YF_CONJ,
       KER_XF_MUL,
       KER_XYF_SUM,
       KER_CORR,
       KER_NOF
    };
    const std::string kernel_str[KER_NOF] = {
            "KER_XF_SQR_NORM",
            "KER_YF_SQR_NORM",
            "KER_YF_CONJ",
            "KER_XF_MUL",
            "KER_XYF_SUM",
            "KER_CORR"
    };

    ProfCUDA();
    ~ProfCUDA();
    void init(int ctxId);
    void setThreadId(void);
    uint64_t * getProfDevicePointer(Kernel ker);
    void logHostStart(void);
    void logHostEnd(void);
    void storeFrameData(void);
    void printData(std::string fileName);
    static void syncCpuGpuTimer(void);

private:
    // Profiling stuff
    static const uint nofBlocks = 2;     // Generally allocate space for two blocks per kernel
    static const uint timesPerBlock = 2; // Start and end timestamp
    static const uint numElem = nofBlocks * KER_NOF * timesPerBlock;
    int thread_id;
    int ctxId;
    static uint64_t startingCpuClock;
    static uint64_t startingGpuClock;
    static double gpuCpuScale;

    // Indicates the current kcf frame number
    struct frameData{
        uint64_t h_targetTimes[numElem];
        uint64_t hostStart;
        uint64_t hostEnd;
    };

    std::vector<frameData> profData;
    frameData hostData;
    uint64_t *d_targetTimes;
    uint64_t convertGpuToCpu(uint64_t gpuTime);
    void copyToHost();
    void copyToDev();
};

#endif // PROFCUDA_H
