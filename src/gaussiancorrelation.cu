#include "kcf.h"
#include "threadctx.hpp"
#include "debug.h"
#include "sched-util/utility_func.cuh"
#ifdef OPENMP
#include <omp.h>
#endif // OPENMP
#ifdef CUFFT

__inline__ __device__ void correlation(float *ifft_res,
                                       size_t size,
                                       size_t sizeScale,
                                       const float *xf_sqr_norm,
                                       const float *yf_sqr_norm,
                                       const double sigma,
                                       const double normFactor) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += blockDim.x * gridDim.x)
    {
        double elem = ifft_res[i];
        double xf_norm = xf_sqr_norm[i/sizeScale];
        double yf_norm = yf_sqr_norm[0];
        elem = exp((-1.0 / (sigma * sigma)) * fmax(((xf_norm + yf_norm) - (2 * elem)) / normFactor, 0.0));
        ifft_res[i] = __double2float_ru(elem);
    }
}

#ifdef PROFILE_GAUSSIAN
__global__ void kernel_correlation(float *ifft_res,
                                   size_t size,
                                   size_t sizeScale,
                                   const float *xf_sqr_norm,
                                   const float *yf_sqr_norm,
                                   const double sigma,
                                   const double normFactor,
                                   uint64_t *targetTimes,
                                   uint64_t *startTime,
                                   uint64_t *offsets)
{
#ifdef USE_KERNEL_SCHED
    Util::logBlockStart(targetTimes, startTime, offsets);
#else
    Util::logBlockStart(targetTimes);
#endif
    correlation(ifft_res, size, sizeScale, xf_sqr_norm, yf_sqr_norm, sigma, normFactor);
    Util::logBlockEnd(targetTimes);
}
#else
__global__ void kernel_correlation(float *ifft_res, size_t size, size_t sizeScale, const float *xf_sqr_norm, const float *yf_sqr_norm, const double sigma, const double normFactor)
{
    correlation(ifft_res, size, sizeScale, xf_sqr_norm, yf_sqr_norm, sigma, normFactor);
}
#endif

void KCF_Tracker::GaussianCorrelation::operator()(ComplexMat &result, const ComplexMat &xf, const ComplexMat &yf,
                                                  double sigma, bool auto_correlation, ThreadCtx &ctx)
{
    TRACE("");
#ifdef PROFILE_GAUSSIAN
    ProfCUDA &pd = ctx.profData;
#endif
    if(!auto_correlation){
#ifdef USE_KERNEL_SCHED
#ifdef OPENMP
        if(omp_get_thread_num() == 0){
            // Only one thread should get the scheduling start time
            ProfCUDA::getStartTime();
        }
#else
        ProfCUDA::getStartTime();
#endif
#endif
        pthread_barrier_wait(&ctx.barrier);
#ifdef PROFILE_GAUSSIAN
        ctx.profData.logHostStart();
#endif /* PROFILE_GAUSSIAN */
    }
    DEBUG_PRINTM(xf);
    DEBUG_PRINT(xf_sqr_norm.num_elem);
#ifdef PROFILE_GAUSSIAN
#ifdef USE_KERNEL_SCHED
    xf.sqr_norm(xf_sqr_norm, pd.getProfDevicePointer(ProfCUDA::KER_XF_SQR_NORM), ProfCUDA::d_targetStartTime, pd.getSchedDevicePointer(ProfCUDA::KER_XF_SQR_NORM));
#else
    xf.sqr_norm(xf_sqr_norm, pd.getProfDevicePointer(ProfCUDA::KER_XF_SQR_NORM), NULL, NULL);
#endif
#else
    xf.sqr_norm(xf_sqr_norm);
#endif
    if (auto_correlation) {
        yf_sqr_norm = xf_sqr_norm;
    } else {
        DEBUG_PRINTM(yf);
#ifdef PROFILE_GAUSSIAN
#ifdef USE_KERNEL_SCHED
        yf.sqr_norm(yf_sqr_norm, pd.getProfDevicePointer(ProfCUDA::KER_YF_SQR_NORM), ProfCUDA::d_targetStartTime, pd.getSchedDevicePointer(ProfCUDA::KER_YF_SQR_NORM));
#else
        yf.sqr_norm(yf_sqr_norm, pd.getProfDevicePointer(ProfCUDA::KER_YF_SQR_NORM), NULL, NULL);
#endif
#else
        yf.sqr_norm(yf_sqr_norm);
#endif
    }
#ifdef PROFILE_GAUSSIAN
#ifdef USE_KERNEL_SCHED
    xyf = auto_correlation ? xf.sqr_mag() : xf.mulProf(
                                                       yf.conj(pd.getProfDevicePointer(ProfCUDA::KER_YF_CONJ),
                                                               ProfCUDA::d_targetStartTime,
                                                               pd.getSchedDevicePointer(ProfCUDA::KER_YF_CONJ)),
                                                       pd.getProfDevicePointer(ProfCUDA::KER_XF_MUL),
                                                       ProfCUDA::d_targetStartTime,
                                                       pd.getSchedDevicePointer(ProfCUDA::KER_XF_MUL)
                                                       ); // xf.muln(yf.conj());
#else
    xyf = auto_correlation ? xf.sqr_mag() : xf.mulProf(
                                                       yf.conj(pd.getProfDevicePointer(ProfCUDA::KER_YF_CONJ),
                                                               NULL,
                                                               NULL,
                                                       pd.getProfDevicePointer(ProfCUDA::KER_XF_MUL),
                                                       NULL,
                                                       NULL
                                                       ); // xf.muln(yf.conj());
#endif
#else
    xyf = auto_correlation ? xf.sqr_mag() : xf * yf.conj(); // xf.muln(yf.conj());
#endif
    DEBUG_PRINTM(xyf);

    // ifft2 and sum over 3rd dimension, we dont care about individual channels
#ifdef PROFILE_GAUSSIAN
#ifdef USE_KERNEL_SCHED
    ComplexMat xyf_sum = xyf.sum_over_channels(pd.getProfDevicePointer(ProfCUDA::KER_XYF_SUM), ProfCUDA::d_targetStartTime, pd.getSchedDevicePointer(ProfCUDA::KER_XYF_SUM));
#else
    ComplexMat xyf_sum = xyf.sum_over_channels(pd.getProfDevicePointer(ProfCUDA::KER_XYF_SUM), NULL, NULL);
#endif
#else
    ComplexMat xyf_sum = xyf.sum_over_channels();
#endif
    DEBUG_PRINTM(xyf_sum);
    ctx.fft.inverse(xyf_sum, ifft_res);
    DEBUG_PRINTM(ifft_res);

    double numel_xf = (xf.cols * xf.rows * (xf.channels() / xf.n_scales));
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((ifft_res.num_elem + threads.x - 1) / threads.x);

#ifdef PROFILE_GAUSSIAN
#ifdef USE_KERNEL_SCHED
    kernel_correlation<<<blocks, threads>>>(ifft_res.deviceMem(),
                                            ifft_res.num_elem,
                                            ifft_res.num_elem/xf.n_scales,
                                            xf_sqr_norm.deviceMem(),
                                            yf_sqr_norm.deviceMem(),
                                            sigma,
                                            numel_xf,
                                            ctx.profData.getProfDevicePointer(ProfCUDA::KER_CORR),
                                            ProfCUDA::d_targetStartTime,
                                            pd.getSchedDevicePointer(ProfCUDA::KER_CORR));
#else
    kernel_correlation<<<blocks, threads>>>(ifft_res.deviceMem(),
                                            ifft_res.num_elem,
                                            ifft_res.num_elem/xf.n_scales,
                                            xf_sqr_norm.deviceMem(),
                                            yf_sqr_norm.deviceMem(),
                                            sigma,
                                            numel_xf,
                                            pd.getProfDevicePointer(ProfCUDA::KER_CORR), NULL, NULL);
#endif
#else
    kernel_correlation<<<blocks, threads>>>(ifft_res.deviceMem(),
                                            ifft_res.num_elem,
                                            ifft_res.num_elem/xf.n_scales,
                                            xf_sqr_norm.deviceMem(),
                                            yf_sqr_norm.deviceMem(),
                                            sigma,
                                            numel_xf);
#endif
#ifdef PROFILE_GAUSSIAN
    CudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
    if(!auto_correlation){
        ctx.profData.logHostEnd();
        pthread_barrier_wait(&ctx.barrier);
        ctx.profData.storeFrameData();
    }
#endif /* PROFILE_GAUSSIAN */
    ctx.fft.forward(ifft_res, result);
}
#endif
