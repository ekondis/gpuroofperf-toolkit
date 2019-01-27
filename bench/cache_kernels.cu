/**
* cache_kernels.cu: This file is part of the gpuroofperf-bench benchmark suite.
*
* Contact: Elias Konstantinidis <ekondis@gmail.com>
**/

#include <algorithm>
#include "cache_kernels.h"

namespace benchmark_kernels {

	#define TOTAL_ITERATIONS  (8192)
	#define UNROLL_ITERATIONS (64)

	#define UNROLL_ITERATIONS_MEM (UNROLL_ITERATIONS/2)

	const int BLOCK_SIZE = 256;

	texture< int, 1, cudaReadModeElementType> texdataI1;
	texture<int2, 1, cudaReadModeElementType> texdataI2;
	texture<int4, 1, cudaReadModeElementType> texdataI4;

	template<class T>
	class dev_fun{
	public:
		// Pointer displacement operation
		__device__ unsigned int operator()(T v1, unsigned int v2);
		// Compute operation (#1)
		__device__ T operator()(const T &v1, const T &v2);
		// Compute operation (#2)
		__device__ T comp_per_element(const T &v1, const T &v2);
		// Value initialization
		__device__ T init(int v);
		// Element loading
		__device__ T load(volatile const T* p, unsigned int offset);
		// Element storing
		__device__ void store(volatile T* p, unsigned int offset, const T &value);
		// Get first element
		__device__ int first_element(const T &v);
		// Reduce elements (XOR operation)
		__device__ int reduce(const T &v);
	};


	template<>
	__device__ unsigned int dev_fun<int>::operator()(int v1, unsigned int v2){
		return v2+static_cast<unsigned int>(v1) ;
	}
	template<>
	__device__ int dev_fun<int>::operator()(const int &v1, const int &v2){
	  return v1 + v2;
	}
	template<>
	__device__ int dev_fun<int>::comp_per_element(const int &v1, const int &v2){
	  return v1 - v2;
	}
	template<>
	__device__ int dev_fun<int>::init(int v){
		return v;
	}
	template<>
	__device__ int dev_fun<int>::load(volatile const int* p, unsigned int offset){
		int retval;
		p += offset;
		// Global level caching (.cg Cache at global level (cache in L2 and below, not L1).)
		asm volatile ("ld.cg.u32 %0, [%1];" : "=r"(retval) : "l"(p));
		return retval;
	}
	template<>
	__device__ void dev_fun<int>::store(volatile int* p, unsigned int offset, const int &value){
		p += offset;
		// Streaming store (.cs Cache streaming, likely to be accessed once.)
		asm volatile ("st.cs.global.u32 [%0], %1;" :: "l"(p), "r"(value));
	}
	template<>
	__device__ int dev_fun<int>::first_element(const int &v){
		return v;
	}
	template<>
	__device__ int dev_fun<int>::reduce(const int &v){
		return v;
	}


	template<>
	__device__ unsigned int dev_fun<int2>::operator()(int2 v1, unsigned int v2){
		return v2+(unsigned int)(v1.x+v1.y) ;
	}
	template<>
	__device__ int2 dev_fun<int2>::operator()(const int2 &v1, const int2 &v2){
		return make_int2(v1.x + v2.x, v1.y + v2.y);
	}
	template<>
	__device__ int2 dev_fun<int2>::comp_per_element(const int2 &v1, const int2 &v2){
		return make_int2(v1.x - v2.x, v1.y - v2.y);
	}
	template<>
	__device__ int2 dev_fun<int2>::init(int v){
		return make_int2(v, v);
	}
	template<>
	__device__ int2 dev_fun<int2>::load(volatile const int2* p, unsigned int offset){
		union{
			unsigned long long ll;
			int2 i2;
		} retval;
		p += offset;
		// Global level caching
		asm volatile ("ld.cg.u64 %0, [%1];" : "=l"(retval.ll) : "l"(p));
		return retval.i2;
	}
	template<>
	__device__ void dev_fun<int2>::store(volatile int2* p, unsigned int offset, const int2 &value){
		union{
			unsigned long long ll;
			int2 i2;
		} retval;
		retval.i2 = value;
		p += offset;
		// Streaming store
		asm volatile ("st.cs.global.u64 [%0], %1;" :: "l"(p), "l"(retval.ll));
	}
	template<>
	__device__ int dev_fun<int2>::first_element(const int2 &v){
		return v.x;
	}
	template<>
	__device__ int dev_fun<int2>::reduce(const int2 &v){
		return v.x ^ v.y;
	}


	template<>
	__device__ unsigned int dev_fun<int4>::operator()(int4 v1, unsigned int v2){
		return v2+(unsigned int)(v1.x+v1.y+v1.z+v1.w) ;
	}
	template<>
	__device__ int4 dev_fun<int4>::operator()(const int4 &v1, const int4 &v2){
		return make_int4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
	}
	template<>
	__device__ int4 dev_fun<int4>::comp_per_element(const int4 &v1, const int4 &v2){
		return make_int4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
	}
	template<>
	__device__ int4 dev_fun<int4>::init(int v){
		return make_int4(v, v, v, v);
	}
	template<>
	__device__ int4 dev_fun<int4>::load(volatile const int4* p, unsigned int offset){
		int4 retval;
		p += offset;
		// Global level caching
		asm volatile ("ld.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(retval.x), "=r"(retval.y), "=r"(retval.z), "=r"(retval.w) : "l"(p));
		return retval;
	}
	template<>
	__device__ void dev_fun<int4>::store(volatile int4* p, unsigned int offset, const int4 &value){
		p += offset;
		// Streaming store
		asm volatile ("st.cs.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) );
	}
	template<>
	__device__ int dev_fun<int4>::first_element(const int4 &v){
		return v.x;
	}
	template<>
	__device__ int dev_fun<int4>::reduce(const int4 &v){
		return v.x ^ v.y ^ v.z ^ v.w;
	}


	template <class T, int blockdim, int stepwidth, int index_clamping>
	__global__ void benchmark_func(T * const g_data){
		dev_fun<T> func;

		// Thread block-wise striding
		int index = stepwidth*blockIdx.x*blockdim + threadIdx.x;
		index = index_clamping==0 ? index : index % index_clamping;
		const int stride = blockdim;

		unsigned int offset = index;
		T temp = func.init(0);
		for(int j=0; j<TOTAL_ITERATIONS; j+=UNROLL_ITERATIONS){
			// Pretend updating of offset in order to force repetitive loads
			offset = func(temp, offset);
	#ifndef TEX_LOADS
			union {
				const T *ptr;
				int2 i;
			} g_data_load_ptr = { g_data+offset };
	#endif

	#pragma unroll
			for(int i=0; i<UNROLL_ITERATIONS; i++){
				const unsigned int iteration_offset = i % stepwidth;
				const T v = func.load(g_data_load_ptr.ptr, iteration_offset*stride);
				// Pretend update of data pointer in order to force reloads
				g_data_load_ptr.i.x ^= func.reduce(v);
				temp = v;
			}
		}
		offset = func(temp, offset);
		if( offset != index ) // Does not actually occur
			*g_data = func.init(offset);
	}

	template<class datatype>
	void runbench_warmup(datatype *cd, long size){
		const long reduced_grid_size = size/(UNROLL_ITERATIONS_MEM)/32;
		const int TOTAL_REDUCED_BLOCKS = reduced_grid_size/BLOCK_SIZE;

		dim3 dimBlock(BLOCK_SIZE, 1, 1);
		dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

		benchmark_func< datatype, BLOCK_SIZE, 1, 256 ><<< dimReducedGrid, dimBlock >>>(cd);
		CUDA_SAFE_CALL( cudaGetLastError() );
		CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	}

	template<class datatype, int stepwidth, int index_clamping>
	double runbench(int total_blocks, datatype *cd, long size){
		const long compute_grid_size = total_blocks*BLOCK_SIZE;
		const long data_size = ((index_clamping==0) ? compute_grid_size : min((int)compute_grid_size, (int)index_clamping))*stepwidth;//*(2-readonly);

		const long long total_iterations = (long long)(TOTAL_ITERATIONS)*compute_grid_size;
		const long long memoryoperations = total_iterations;

		// Set device memory
		CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(datatype)) );  // initialize to zeros

		dim3 dimBlock(BLOCK_SIZE, 1, 1);
		dim3 dimGrid(total_blocks, 1, 1);

		cudaEvent_t start, stop;

		initializeEvents(&start, &stop);
		benchmark_func< datatype, BLOCK_SIZE, stepwidth, index_clamping ><<< dimGrid, dimBlock >>>(cd);
		float kernel_time = finalizeEvents(start, stop);
		double bandwidth = (static_cast<double>(memoryoperations)*sizeof(datatype))/kernel_time*1000./(1000.*1000.*1000.);

		int current_device;
		cudaDeviceProp deviceProp;
		CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
		CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );

		return bandwidth;
	}

	template<class datatype>
	double cachebenchGPU(long size){
		// Construct grid size
		cudaDeviceProp deviceProp;
		int current_device;
		CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
		CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
		const int SM_count = deviceProp.multiProcessorCount;
		const int Threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
		const int BLOCKS_PER_SM = Threads_per_SM/BLOCK_SIZE;
		const int TOTAL_BLOCKS = BLOCKS_PER_SM * SM_count;

		datatype *cd;

		CUDA_SAFE_CALL( cudaMalloc((void**)&cd, size*sizeof(datatype)) );

		// Set device memory
		CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(datatype)) );  // initialize to zeros

		// Bind textures to buffer
		cudaBindTexture(0, texdataI1, cd, size*sizeof(datatype));
		cudaBindTexture(0, texdataI2, cd, size*sizeof(datatype));
		cudaBindTexture(0, texdataI4, cd, size*sizeof(datatype));

		// Synchronize in order to wait for memory operations to finish
		CUDA_SAFE_CALL( cudaDeviceSynchronize() );

		runbench_warmup(cd, size);

		double peak_bw = 0.0;
		peak_bw = max( peak_bw, runbench<datatype, 1,  512>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 1, 1024>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 1, 2048>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 1, 4096>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 1, 8192>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 1,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 2,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 3,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 4,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 5,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 6,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 7,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 8,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 9,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 10,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 11,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 12,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 13,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 14,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 15,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 16,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 18,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 20,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 22,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 24,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 28,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 32,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 40,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 48,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 56,    0>(TOTAL_BLOCKS, cd, size) );
		peak_bw = max( peak_bw, runbench<datatype, 64,    0>(TOTAL_BLOCKS, cd, size) );

		// Copy results back to host memory (not needed)
		//CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(datatype), cudaMemcpyDeviceToHost) );

		// Unbind textures
		cudaUnbindTexture(texdataI1);
		cudaUnbindTexture(texdataI2);
		cudaUnbindTexture(texdataI4);

		CUDA_SAFE_CALL( cudaFree(cd) );
		return peak_bw;
	}

	generic_benchmark_result_list run_cache_benchmark(void) {
		constexpr unsigned int VECTOR_SIZE = 8*1024*1024;

		double peak_bw_ro_int1 = cachebenchGPU<int>(VECTOR_SIZE);
		double peak_bw_ro_int2 = cachebenchGPU<int2>(VECTOR_SIZE);
		double peak_bw_ro_int4 = cachebenchGPU<int4>(VECTOR_SIZE);

		generic_benchmark_result_list results;
		results["BW_L2"] = std::max( {peak_bw_ro_int1, peak_bw_ro_int2, peak_bw_ro_int4} );
		return results;
	}
}
