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


#ifdef USE_KERNEL_SCHED
#define SCHED_PARAMS(param) , param
#else
#define SCHED_PARAMS(param)
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
#ifdef USE_KERNEL_SCHED
    static void getStartTime(void);
    static uint64_t* d_targetStartTime;
    uint64_t * getSchedDevicePointer(Kernel ker);
#endif

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
#ifdef USE_KERNEL_SCHED
    uint64_t *d_targetOffsets;
    // TODO: Maybe we could pass this table through the init function.
    // So it could be prepared for each thread ctx individually. With
    // that we could get rid of the static stream offset...
    //                                             Block 0,      Block 1
    uint64_t h_targetOffsets[KER_NOF*nofBlocks] = {0,            0, // KER_XF_SQR_NORM
                                                   300000,       0, // KER_YF_SQR_NORM
                                                   550000,       1000000, // KER_YF_CONJ,
                                                   1700000,      2500000, // KER_XF_MUL,
                                                   3650000,      3700000, // KER_XYF_SUM,
                                                   3900000,      4000000}; // KER_CORR,

   static const uint64_t streamOffset = 125000;
#endif

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
