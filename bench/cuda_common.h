/**
* cuda_common.h: This file is part of the gpuroofperf-bench benchmark suite.
*
* Contact: Elias Konstantinidis <ekondis@gmail.com>
**/

#pragma once

#include <iostream>

#define CUDA_SAFE_CALL(call) {                              \
    cudaError err = call;                                   \
    if( cudaSuccess != err) {                               \
        std::cerr << "Cuda error 'in file '" << __FILE__    \
		          << "' in line " << __LINE__ << " : "      \
				  << cudaGetErrorString(err) << ".\n";      \
        exit(EXIT_FAILURE);                                 \
    } }

// Beginning of GPU Architecture definitions
static inline int _ConvertSMVer2Cores(int major, int minor){
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
		{ 0x10,   8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11,   8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12,   8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13,   8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20,  32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21,  48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
		{ 0x53, 128 }, // Maxwell Generation (SM 5.3) GM20x class
		{ 0x60,  64 }, // Pascal Generation (SM 6.0) GP100 class
		{ 0x61, 128 }, // Pascal Generation (SM 6.1) GP10x class
		{ 0x62, 128 }, // Pascal Generation (SM 6.2) GP10x class
		{ 0x70,  64 }, // Volta Generation (SM 7.0) GV100 class
		{ 0x75,  64 }, // Turing Generation (SM 7.5) TU10x class
		{   -1,  -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    return nGpuArchCoresPerSM[index].Cores;
}

static inline void GetDevicePeakInfo(double *aGIPS, double *aGBPS, cudaDeviceProp *aDeviceProp = NULL){
	cudaDeviceProp deviceProp;
	int current_device;
	if( aDeviceProp )
		deviceProp = *aDeviceProp;
	else{
		CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
		CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
	}
	const int TotalSPs = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)*deviceProp.multiProcessorCount;
	*aGIPS = 1000.0 * deviceProp.clockRate * TotalSPs / (1000.0 * 1000.0 * 1000.0);  // Giga instructions/sec
	*aGBPS = 2.0 * (double)deviceProp.memoryClockRate * 1000.0 * (double)deviceProp.memoryBusWidth / 8.0;
}
