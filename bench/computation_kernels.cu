/**
* computation_kernels.cu: This file is part of the gpuroofperf-bench benchmark suite.
*
* Contact: Elias Konstantinidis <ekondis@gmail.com>
**/

#include <algorithm>
#include "cuda_common.h"
#include "kernels_common.h"

#define COMP_ITERATIONS (1024)
#define REGBLOCK_SIZE (16)

namespace benchmark_kernels {

	template<class T>
	class dev_fun_mad{
	public:
		__device__ T operator()(T v1, T v2, T v3){
			return v1*v2+v3;
		}
	};

	template<class T>
	class dev_fun_plus{
	public:
		__device__ T operator()(T v1, T v2, T v3){
			return v1+v3;
		}
	};

	template<class T>
	class dev_fun_mult{
	public:
		__device__ T operator()(T v1, T v2, T v3){
			return v1*v3;
		}
	};

	template<class T>
	class dev_fun_shift{
	public:
		__device__ T operator()(T v1, T v2, T v3){
			return v1 << v3;
		}
	};

	template <class T, class F, bool ManyRegs, bool TemperateUnroll>
	__global__ void benchmark_func(T *g_data){
		F func;
		int tid = threadIdx.x;
		T r0 = g_data[blockIdx.x*blockDim.x + threadIdx.x],
		  r1 = r0+(T)(31),
		  r2 = r0+(T)(37),
		  r3 = r0+(T)(41),
		  r4 = r0+(T)(43),
		  r5 = r0+(T)(47),
		  r6 = r0+(T)(53),
		  r7 = r0+(T)(59),
		  r8 = r0+(T)(61),
		  r9 = r0+(T)(67),
		  rA = r0+(T)(71),
		  rB = r0+(T)(73),
		  rC = r0+(T)(79),
		  rD = r0+(T)(83),
		  rE = r0+(T)(89),
		  rF = r0+(T)(97);

	#pragma unroll  TemperateUnroll ? 2 : 32
		for(int j=0; j<COMP_ITERATIONS; j++){
			if( ManyRegs ){
				r2 = func(r0, r1, r2);
				r5 = func(r3, r4, r5);
				r8 = func(r6, r7, r8);
				rB = func(r9, rA, rB);
				rE = func(rC, rD, rE);
				r1 = func(rF, r0, r1);
				r4 = func(r2, r3, r4);
				r7 = func(r5, r6, r7);
				rA = func(r8, r9, rA);
				rD = func(rB, rC, rD);
				r0 = func(rE, rF, r0);
				r3 = func(r1, r2, r3);
				r6 = func(r4, r5, r6);
				r9 = func(r7, r8, r9);
				rC = func(rA, rB, rC);
				rF = func(rD, rE, rF);
			} else {
				r0 = func(r6, r6, r7);
				r1 = func(r7, r7, r8);
				r2 = func(r8, r8, r9);
				r3 = func(r9, r9, rA);
				r4 = func(rA, rA, rB);
				r5 = func(rB, rB, rC);
				r6 = func(rC, rC, rD);
				r7 = func(rD, rD, rE);
				r8 = func(rE, rE, rF);
				r9 = func(rF, rF, r0);
				rA = func(r0, r0, r1);
				rB = func(r1, r1, r2);
				rC = func(r2, r2, r3);
				rD = func(r3, r3, r4);
				rE = func(r4, r4, r5);
				rF = func(r5, r5, r6);
			}
		}
		if(r0==(T)-123456789.123456789){ // extremely unlikely just to avoid code elimination
			g_data[tid+0*warpSize] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+rA+rB+rC+rD+rE+rF;
		}
	}

	generic_benchmark_result_list run_computation_benchmark(void) {
		constexpr unsigned int VECTOR_SIZE = 8*1024*1024;
		constexpr unsigned int datasize = VECTOR_SIZE*sizeof(double);

		// initialize host data
		double *c = (double*)malloc(datasize);
		memset(c, 0, sizeof(int)*VECTOR_SIZE);

		// GPU execution configuration
		const int BLOCK_SIZE = 256;//128;//256;
		const int TOTAL_BLOCKS = VECTOR_SIZE/(BLOCK_SIZE*REGBLOCK_SIZE);
		double *cd = nullptr;

		// Allocate space % copy data to device memory
		CUDA_SAFE_CALL( cudaMalloc((void**)&cd, VECTOR_SIZE*sizeof(double)) );
		CUDA_SAFE_CALL( cudaMemset(cd, 0, VECTOR_SIZE*sizeof(double)) );  // initialize to zeros

		// Synchronize in order to wait for memory operations to finish
		CUDA_SAFE_CALL( cudaThreadSynchronize() );

		dim3 dimBlock(BLOCK_SIZE, 1, 1);
		dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
		dim3 dimGridSP(2*TOTAL_BLOCKS, 1, 1);
		cudaEvent_t start, stop;

		// warm up
		benchmark_func< double, dev_fun_mad<double>, false, true ><<< dimGrid, dimBlock >>>(cd);
		CUDA_SAFE_CALL( cudaGetLastError() );
		CUDA_SAFE_CALL( cudaThreadSynchronize() );

		initializeEvents(&start, &stop);
		benchmark_func< float, dev_fun_mad<float>, false, false ><<< dimGridSP, dimBlock >>>((float*)cd);
		float kernel_time_mad_sp = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		benchmark_func< float, dev_fun_mad<float>, true, false ><<< dimGridSP, dimBlock >>>((float*)cd);
		float kernel_time_mad_sp_manyregs = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		benchmark_func< int, dev_fun_mad<int>, false, true ><<< dimGridSP, dimBlock >>>((int*)cd);
		float kernel_time_mad_int = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		benchmark_func< int, dev_fun_plus<int>, false, false ><<< dimGridSP, dimBlock >>>((int*)cd);
		float kernel_time_plus_int = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		benchmark_func< double, dev_fun_mad<double>, true, false ><<< dimGrid, dimBlock >>>(cd);
		float kernel_time_mad_dp = finalizeEvents(start, stop);

		// Copy results back to host memory
		CUDA_SAFE_CALL( cudaMemcpy(c, cd, VECTOR_SIZE*sizeof(double), cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaFree(cd) );

		const double kernel_perf_mad_sp = (2*(double)COMP_ITERATIONS*2*VECTOR_SIZE)/std::min(kernel_time_mad_sp, kernel_time_mad_sp_manyregs)*1000./(double)(1000*1000*1000);
		const double kernel_perf_mad_dp = (2*(double)COMP_ITERATIONS*VECTOR_SIZE)/kernel_time_mad_dp*1000./(double)(1000*1000*1000);
		const double kernel_perf_mad_int = (2*(double)COMP_ITERATIONS*2*VECTOR_SIZE)/kernel_time_mad_int*1000./(double)(1000*1000*1000);
		const double kernel_perf_plus_int = ((double)COMP_ITERATIONS*2*VECTOR_SIZE)/kernel_time_plus_int*1000./(double)(1000*1000*1000);

		free(c);

		generic_benchmark_result_list results;
		results["GFLOPS_SP"] = kernel_perf_mad_sp;
		results["GFLOPS_DP"] = kernel_perf_mad_dp;
		results["GIOPS_MAD"] = kernel_perf_mad_int;
		results["GIOPS_ADD"] = kernel_perf_plus_int;

		return results;
	}

}
