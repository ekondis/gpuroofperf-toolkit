/**
* shmem_kernels.cu: This file is part of the gpuroofperf-bench benchmark suite.
*
* Contact: Elias Konstantinidis <ekondis@gmail.com>
**/

#include <algorithm>
#include "shmem_kernels.h"

namespace benchmark_kernels {

	#define TOTAL_ITERATIONS (1024)

	// shared memory swap operation (2 floats read + 2 floats write)
	template <class T>
	__device__ void shmem_swap(T *v1, T *v2){
		T tmp = *v2;
		*v2 = *v1;
		*v1 = tmp;
	}

	template <class T>
	__device__ T init_val(int i){
		return static_cast<T>(i);
	}

	template <>
	__device__ float2 init_val(int i){
		return make_float2(i, i+11);
	}

	template <>
	__device__ float4 init_val(int i){
		return make_float4(i, i+11, i+19, i+23);
	}

	template <class T>
	__device__ T reduce_vector(T v1, T v2, T v3, T v4, T v5, T v6){
		return static_cast<T>(v1 + v2 + v3 + v4 + v5 + v6);
	}

	template <>
	__device__ float2 reduce_vector(float2 v1, float2 v2, float2 v3, float2 v4, float2 v5, float2 v6){
		return make_float2(v1.x + v2.x + v3.x + v4.x + v5.x + v6.x, v1.y + v2.y + v3.y + v4.y + v5.y + v6.y);
	}

	template <>
	__device__ float4 reduce_vector(float4 v1, float4 v2, float4 v3, float4 v4, float4 v5, float4 v6){
		return make_float4(v1.x + v2.x + v3.x + v4.x + v5.x + v6.x, v1.y + v2.y + v3.y + v4.y + v5.y + v6.y, v1.z + v2.z + v3.z + v4.z + v5.z + v6.z, v1.w + v2.w + v3.w + v4.w + v5.w + v6.w);
	}

	template <class T>
	__device__ void set_vector(T *target, int offset, T v){
		target[offset] = v;
	}

	template <>
	__device__ void set_vector(float2 *target, int offset, float2 v){
		target[offset].x = v.x;
		target[offset].y = v.y;
	}

	template <>
	__device__ void set_vector(float4 *target, int offset, float4 v){
		target[offset].x = v.x;
		target[offset].y = v.y;
		target[offset].z = v.z;
		target[offset].w = v.w;
	}

	extern __shared__ float shm_buffer_ptr[];

	template <class T>
	__global__ void benchmark_shmem(T *g_data){
		T *shm_buffer = (T*)shm_buffer_ptr;
		int tid = threadIdx.x;
		int globaltid = blockIdx.x*blockDim.x + tid;
		set_vector(shm_buffer, tid+0*blockDim.x, init_val<T>(tid));
		set_vector(shm_buffer, tid+1*blockDim.x, init_val<T>(tid+1));
		set_vector(shm_buffer, tid+2*blockDim.x, init_val<T>(tid+3));
		set_vector(shm_buffer, tid+3*blockDim.x, init_val<T>(tid+7));
		set_vector(shm_buffer, tid+4*blockDim.x, init_val<T>(tid+13));
		set_vector(shm_buffer, tid+5*blockDim.x, init_val<T>(tid+17));
		__threadfence_block();
	#pragma unroll 32
		for(int j=0; j<TOTAL_ITERATIONS; j++){
			shmem_swap(shm_buffer+tid+0*blockDim.x, shm_buffer+tid+1*blockDim.x);
			shmem_swap(shm_buffer+tid+2*blockDim.x, shm_buffer+tid+3*blockDim.x);
			shmem_swap(shm_buffer+tid+4*blockDim.x, shm_buffer+tid+5*blockDim.x);
			//__threadfence_block();
			shmem_swap(shm_buffer+tid+1*blockDim.x, shm_buffer+tid+2*blockDim.x);
			shmem_swap(shm_buffer+tid+3*blockDim.x, shm_buffer+tid+4*blockDim.x);
			//__threadfence_block();
		}
		g_data[globaltid] = reduce_vector<T>(shm_buffer[tid+0*blockDim.x], shm_buffer[tid+1*blockDim.x], shm_buffer[tid+2*blockDim.x], shm_buffer[tid+3*blockDim.x], shm_buffer[tid+4*blockDim.x], shm_buffer[tid+5*blockDim.x]);
	}

	generic_benchmark_result_list run_ldst_benchmark(void) {
		constexpr unsigned int VECTOR_SIZE = 8*1024*1024;
		constexpr int BLOCK_SIZE = 256;
		constexpr int TOTAL_BLOCKS = VECTOR_SIZE/(BLOCK_SIZE);
		double *cd;

		//CUDA_SAFE_CALL( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );

		CUDA_SAFE_CALL( cudaMalloc((void**)&cd, VECTOR_SIZE*sizeof(double)) );

		// Copy data to device memory
		CUDA_SAFE_CALL( cudaMemset(cd, 0, VECTOR_SIZE*sizeof(double)) );  // initialize to zeros

		// Synchronize in order to wait for memory operations to finish
		CUDA_SAFE_CALL( cudaThreadSynchronize() );

	    dim3 dimBlock(BLOCK_SIZE, 1, 1);
	    dim3 dimGrid_f1(TOTAL_BLOCKS, 1, 1);
	    dim3 dimGrid_f2(TOTAL_BLOCKS/2, 1, 1);
	    dim3 dimGrid_f4(TOTAL_BLOCKS/4, 1, 1);
		int shared_mem_per_block = BLOCK_SIZE*sizeof(float)*6;
		cudaEvent_t start, stop;

		// warm up
		benchmark_shmem<float><<< dimGrid_f4, dimBlock, shared_mem_per_block >>>((float*)cd);
		CUDA_SAFE_CALL( cudaGetLastError() );
		CUDA_SAFE_CALL( cudaThreadSynchronize() );

		initializeEvents(&start, &stop);
		benchmark_shmem<float><<< dimGrid_f1, dimBlock, shared_mem_per_block >>>((float*)cd);
		float krn_time_shmem_32b = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		benchmark_shmem<float2><<< dimGrid_f2, dimBlock, shared_mem_per_block*2 >>>((float2*)cd);
		float krn_time_shmem_64b = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		benchmark_shmem<float4><<< dimGrid_f4, dimBlock, shared_mem_per_block*4 >>>((float4*)cd);
		float krn_time_shmem_128b = finalizeEvents(start, stop);

		// Copy results back to host memory
		//CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaFree(cd) );

		const long long operations_32bit  = (6LL+4*5*TOTAL_ITERATIONS+6)*VECTOR_SIZE;
		const long long operations_64bit  = (6LL+4*5*TOTAL_ITERATIONS+6)*VECTOR_SIZE/2;
		const long long operations_128bit = (6LL+4*5*TOTAL_ITERATIONS+6)*VECTOR_SIZE/4;

		double ldst_ops_tput_32b  = ((double)operations_32bit)/ krn_time_shmem_32b*1000./(double)(1000*1000*1000);
		double ldst_ops_tput_64b  = ((double)operations_64bit)/ krn_time_shmem_64b*1000./(double)(1000*1000*1000);
		double ldst_ops_tput_128b = ((double)operations_128bit)/krn_time_shmem_128b*1000./(double)(1000*1000*1000);

		generic_benchmark_result_list results;
		results["GIOPS_LDST"] = std::max( {ldst_ops_tput_32b, ldst_ops_tput_64b, ldst_ops_tput_128b} );
		return results;
	}

}
