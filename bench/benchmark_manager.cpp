/**
* benchmark_manager.cpp: This file is part of the gpuroofperf-bench benchmark suite.
*
* Contact: Elias Konstantinidis <ekondis@gmail.com>
**/

#include <iostream>
#include <cuda_runtime_api.h>
#include "benchmark_manager.h"
#include "cuda_common.h"
#include "kernels_common.h"
#include "computation_kernels.h"
#include "shmem_kernels.h"
#include "dram_kernels.h"
#include "cache_kernels.h"

benchmark_manager::benchmark_manager(int idevice) {
	CUDA_SAFE_CALL( cudaSetDevice(idevice) );
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, idevice) );
	device_name = deviceProp.name;
}

benchmark_manager::~benchmark_manager() {
	CUDA_SAFE_CALL( cudaDeviceReset() );
}

// Output basic device information
void benchmark_manager::get_device_info(std::ostream& out) {
	cudaDeviceProp deviceProp;
	int current_device, driver_version;
	CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
	CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
	CUDA_SAFE_CALL( cudaDriverGetVersion(&driver_version) );
	out << "------------------------ Device specifications ------------------------" << std::endl;
	out << "Device:              " << deviceProp.name << std::endl;
	out << "CUDA driver version: " << driver_version/1000 << '.' << driver_version%1000 << std::endl;
	out << "GPU clock rate:      " << deviceProp.clockRate/1000 << " MHz" << std::endl;
	out << "Memory clock rate:   " << deviceProp.memoryClockRate/1000/2 << " MHz" << std::endl;
	out << "Memory bus width:    " << deviceProp.memoryBusWidth << " bits" << std::endl;
	out << "WarpSize:            " << deviceProp.warpSize << std::endl;
	out << "L2 cache size:       " << deviceProp.l2CacheSize/1024 << " KB" << std::endl;
	out << "Total global mem:    " << (int)(deviceProp.totalGlobalMem/1024/1024) << " MB" << std::endl;
	out << "ECC enabled:         " << (deviceProp.ECCEnabled?"Yes":"No") << std::endl;
	out << "Compute Capability:  " << deviceProp.major << '.' << deviceProp.minor << std::endl;
	const int TotalSPs = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)*deviceProp.multiProcessorCount;
	out << "Total SPs:           " << TotalSPs << " (" << deviceProp.multiProcessorCount << " MPs x " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << " SPs/MP)" << std::endl;
	double InstrThroughput, MemBandwidth;
	GetDevicePeakInfo(&InstrThroughput, &MemBandwidth, &deviceProp);
	out << "Compute throughput:  " << 2.0*InstrThroughput << " GFlops (theoretical single precision FMAs)" << std::endl;
	out << "Memory bandwidth:    " << MemBandwidth/(1000.0*1000.0*1000.0) << " GB/sec" << std::endl;
	out << "-----------------------------------------------------------------------" << std::endl;
}

benchmark_manager::device_list benchmark_manager::query_devices() {
	benchmark_manager::device_list devices;
	int device_count;
	CUDA_SAFE_CALL( cudaGetDeviceCount(&device_count) );
	for (int i = 0; i < device_count; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		devices.push_back( device_list_element(i, prop.name) );
	}
	return devices;
}

void benchmark_manager::run_computations(void) {
	auto res = benchmark_kernels::run_computation_benchmark();
	benchmark_measurements current;
	current.tp_gflops_sp = res["GFLOPS_SP"];
	current.tp_gflops_dp = res["GFLOPS_DP"];
	current.tp_giops = res["GIOPS_MAD"];
	current.tp_iadd = res["GIOPS_ADD"];
	measurements = benchmark_measurements::get_max( {measurements, current} );
}

void benchmark_manager::run_shmem_ops(void) {
	auto res = benchmark_kernels::run_ldst_benchmark();
	benchmark_measurements current;
	current.tp_ldst = res["GIOPS_LDST"];
	measurements = benchmark_measurements::get_max( {measurements, current} );
}

void benchmark_manager::run_memory_ops(void) {
	auto res = benchmark_kernels::run_dram_benchmark();
	benchmark_measurements current;
	current.bw_dram = res["BW_DRAM"];
	measurements = benchmark_measurements::get_max( {measurements, current} );
}

void benchmark_manager::run_cache_ops(void) {
	auto res = benchmark_kernels::run_cache_benchmark();
	benchmark_measurements current;
	current.bw_l2 = res["BW_L2"];
	measurements = benchmark_measurements::get_max( {measurements, current} );
}
