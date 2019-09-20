#include "utility_host.hpp"
#include <time.h>
#include <iostream>

uint64_t Util::getCpuTimeNs(void){
    struct timespec time;
    if(clock_gettime(CLOCK_MONOTONIC, &time)){
        std::cerr << "Error getting CPU time." << std::endl;
        return 0;
    }
    return time.tv_sec * 1000000000 + time.tv_nsec;
}
