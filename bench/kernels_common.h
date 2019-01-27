/**
* kernels_common.h: This file is part of the gpuroofperf-bench benchmark suite.
*
* Contact: Elias Konstantinidis <ekondis@gmail.com>
**/

#pragma once

#include <map>
#include "cuda_common.h"

namespace benchmark_kernels {
	typedef std::map<std::string, double> generic_benchmark_result_list;

	inline void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop){
		CUDA_SAFE_CALL( cudaEventCreate(start) );
		CUDA_SAFE_CALL( cudaEventCreate(stop) );
		CUDA_SAFE_CALL( cudaEventRecord(*start, 0) );
	}

	inline float finalizeEvents(cudaEvent_t start, cudaEvent_t stop){
		CUDA_SAFE_CALL( cudaGetLastError() );
		CUDA_SAFE_CALL( cudaEventRecord(stop, 0) );
		CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
		float kernel_time;
		CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
		CUDA_SAFE_CALL( cudaEventDestroy(start) );
		CUDA_SAFE_CALL( cudaEventDestroy(stop) );
		return kernel_time;
	}

}
