#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#ifdef OPENMP
#include <omp.h>
#endif
#include "profCUDA.hpp"
#include "cuda_error_check.hpp"
#include "utility_host.hpp"

uint64_t ProfCUDA::startingCpuClock = 0;
uint64_t ProfCUDA::startingGpuClock = 0;
double ProfCUDA::gpuCpuScale = 0;

ProfCUDA::ProfCUDA(){
    // Init CUDA data
    CudaSafeCall(cudaMalloc(&this->d_targetTimes, sizeof(this->hostData.h_targetTimes)));
}

ProfCUDA::~ProfCUDA(){
    // Deinit CUDA data
    CudaSafeCall(cudaFree(this->d_targetTimes));
}

void ProfCUDA::init(int ctxId){
    // Memset to 0 jsut to be sure
    memset(this->hostData.h_targetTimes, 0, sizeof(this->hostData.h_targetTimes));
    this->copyToDev();
    this->ctxId = ctxId;
}

void ProfCUDA::setThreadId(void){
#ifdef OPENMP
    this->thread_id = omp_get_thread_num();
#else
    this->thread_id = 0;
#endif
}

uint64_t *ProfCUDA::getProfDevicePointer(Kernel ker){
    return &this->d_targetTimes[ker * this->nofBlocks * this->timesPerBlock];
}

void ProfCUDA::logHostStart(void){
    this->hostData.hostStart = Util::getCpuTimeNs();
}

void ProfCUDA::logHostEnd(void){
    this->hostData.hostEnd = Util::getCpuTimeNs();
}

void ProfCUDA::storeFrameData(){
    this->copyToHost();
    //Store current frame data to profData
    this->profData.push_back(this->hostData);
    // Safe thread id
    this->setThreadId();
}

void ProfCUDA::printData(std::string fileName){
    // Open the file to write the json data
    std::ofstream file;
    file.open (fileName);

    // Write json header
    file << "{\n";
    //file << "\"gpu_cpu_scale\": " << ProfCUDA::gpuCpuScale << ", \n";
    file << "\"cpu_start_ns\": " << ProfCUDA::startingCpuClock << ", \n";
    file << "\"gpu_start_ns\": " << ProfCUDA::startingGpuClock << ", \n";
    file << "\"nof_blocks\": " << this->nofBlocks << ", \n";
    file << "\"times_per_block\": " << this->timesPerBlock << ", \n";
    file << "\"kernel_names\": [ ";
    for(uint ker = KER_XF_SQR_NORM ; ker < KER_NOF ; ++ker){
        if(ker > 0)
            file << ", ";
            file << "\"" << this->kernel_str[ker] << "\"";
    }
    file << "],\n";
    file << "\"thread_id\": " << this->thread_id << ", \n";
    file << "\"ctx_id\": " << this->ctxId << ", \n";
    file << "\"frame\": [\n";

    // Iterate throught profData and print it out
    bool first = true;
    for(auto &frame:this->profData){
        // Open frame object
        if(!first)
            file << ",\n";
        file << "{\n";
        first = false;

        // Print host times
        file << "\"host_start\": " << frame.hostStart - ProfCUDA::startingCpuClock << ", \n";
        file << "\"host_end\": " << frame.hostEnd  - ProfCUDA::startingCpuClock << ", \n";

        // Print target kerneltimes
        for(uint ker = KER_XF_SQR_NORM ; ker < KER_NOF ; ++ker){
            if(ker > 0)
                file << ",\n";
            file << "\"" << this->kernel_str[ker] << "\" :[";
            uint64_t *data = &frame.h_targetTimes[ker * this->nofBlocks * this->timesPerBlock];
            for(uint i = 0; i < this->nofBlocks * this->timesPerBlock ; ++i){
                if(i > 0)
                    file << ", ";
                file << this->convertGpuToCpu(data[i]);
            }
            file << "]";
        }

        // Close frame object
        file << "\n}";
    }

    // Close frame list
    file << "\n]\n";

    // Close json file
    file << "}\n";
    file.close();
}

void ProfCUDA::syncCpuGpuTimer(void){
    std::cout << "Retrieving GPU-CPU timer offset" << std::endl;
    (void)Util::getHostDeviceTimeOffset(&ProfCUDA::startingGpuClock, &ProfCUDA::startingCpuClock);
    //std::cout << "Retrieving GPU-CPU timer scale (takes 1s)" << std::endl;
    //(void)Util::getGpuTimeScale(&ProfCUDA::gpuCpuScale);
    std::cout << "End CPU-GPU sync" << std::endl;
}

uint64_t ProfCUDA::convertGpuToCpu(uint64_t gpuTime){
        if(gpuTime == 0) return 0;
        return (gpuTime-ProfCUDA::startingGpuClock);// * ProfCUDA::gpuCpuScale;
}

void ProfCUDA::copyToDev(){
    CudaSafeCall(cudaMemcpyAsync(this->d_targetTimes, this->hostData.h_targetTimes, sizeof(this->hostData.h_targetTimes), cudaMemcpyHostToDevice, cudaStreamPerThread));
    CudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
}

void ProfCUDA::copyToHost(){
    CudaSafeCall(cudaMemcpyAsync(this->hostData.h_targetTimes, this->d_targetTimes, sizeof(this->hostData.h_targetTimes), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
}
