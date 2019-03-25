/**
* dram_kernels.cu: This file is part of the gpuroofperf-bench benchmark suite.
*
* Contact: Elias Konstantinidis <ekondis@gmail.com>
**/

#include <algorithm>
#include "dram_kernels.h"

namespace benchmark_kernels {

	const double constVal  = 13.1;
	const float  constValf = static_cast<float>(constVal);
	const int4 constVali4  = make_int4(3,7,13,17);

	typedef texture<float, 1, cudaReadModeElementType> texture_float;
	typedef texture<int2, 1, cudaReadModeElementType> texture_double;
	typedef texture<int4, 1, cudaReadModeElementType> texture_int4;
	typedef texture<float, 2, cudaReadModeElementType> texture_float_2d;
	typedef texture<int2, 2, cudaReadModeElementType> texture_int2_2d;

	texture_float texdataF;
	texture_double texdataD;
	texture_int4 texdataI4;
	texture_float_2d texdataF_2D;
	texture_int2_2d texdataD_2D;

	inline __device__ int4 operator+(int4 a, int4 b) {
		return make_int4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
	}

	inline __device__ bool operator==(int4 a, int4 b) {
		return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w;
	}

	template <class T>
	__device__ __forceinline__ T textureFetch(unsigned int Aindex){
		return T();
	}

	template <>
	__device__ __forceinline__ double textureFetch<double>(unsigned int Aindex){
		int2 v = tex1Dfetch(texdataD, Aindex);
		return __hiloint2double(v.y, v.x);
	}

	template <>
	__device__ __forceinline__ float textureFetch<float>(unsigned int Aindex){
		return tex1Dfetch(texdataF, Aindex);
	}

	template <>
	__device__ __forceinline__ int4 textureFetch<int4>(unsigned int Aindex){
		return tex1Dfetch(texdataI4, Aindex);
	}

	template <class T>
	__device__ __forceinline__ T texture2DFetch(unsigned int Ax, unsigned int Ay){
	}

	template <>
	__device__ __forceinline__ double texture2DFetch<double>(unsigned int Ax, unsigned int Ay){
		union {
			int2 i;
			float2 f;
		} v;
		v.i = tex2D(texdataD_2D, Ax, Ay);
		return __hiloint2double(v.i.y, v.i.x);
	}

	template <>
	__device__ __forceinline__ float texture2DFetch<float>(unsigned int Ax, unsigned int Ay){
		return tex2D(texdataF_2D, Ax, Ay);
	}

	template <class T, int granularity, bool doRead, bool doWrite, bool useTexture>
	__global__ void kmemaccess(const T value, const T * __restrict g_adata, T * __restrict g_cdata){
		const unsigned int blockSize = blockDim.x;
		const int stride = blockSize;
		int i = blockIdx.x*blockSize*granularity + threadIdx.x;
		T tmps[granularity];
		#pragma unroll
		for(int j=0; j<granularity; j++)
			tmps[j] = doRead ? (useTexture ? textureFetch<T>(i+j*stride) : g_adata[i+j*stride]) : value;
		T sum = tmps[0];
		#pragma unroll
		for(int j=1; j<granularity; j++)
			sum = sum + tmps[j];
		if( doWrite || value==sum ){
			#pragma unroll
			for(int j=0; j<granularity; j++)
				g_cdata[i+j*stride] = tmps[j];
		}
	}

	template <class T>
	__global__ void kmemaccess2D(const T value, T * __restrict g_cdata){
		const int granularity = 8;
		const int stride = blockDim.y;
		int ix = blockIdx.x*blockDim.x + threadIdx.x;
		int iy = blockIdx.y*blockDim.y*granularity + threadIdx.y;
		if( granularity==8 ){
			// specialization of code with manual unrolling
			const T tmp0 = texture2DFetch<T>(ix, iy+0*stride);
			const T tmp1 = texture2DFetch<T>(ix, iy+1*stride);
			const T tmp2 = texture2DFetch<T>(ix, iy+2*stride);
			const T tmp3 = texture2DFetch<T>(ix, iy+3*stride);
			const T tmp4 = texture2DFetch<T>(ix, iy+4*stride);
			const T tmp5 = texture2DFetch<T>(ix, iy+5*stride);
			const T tmp6 = texture2DFetch<T>(ix, iy+6*stride);
			const T tmp7 = texture2DFetch<T>(ix, iy+7*stride);
			if( value==(tmp0+tmp1+tmp2+tmp3+tmp4+tmp5+tmp6+tmp7) ){
				const int wx = blockDim.x * gridDim.x;
				g_cdata[ix+iy*wx+0*stride] = tmp0;
				g_cdata[ix+iy*wx+1*stride] = tmp1;
				g_cdata[ix+iy*wx+2*stride] = tmp2;
				g_cdata[ix+iy*wx+3*stride] = tmp3;
				g_cdata[ix+iy*wx+4*stride] = tmp4;
				g_cdata[ix+iy*wx+5*stride] = tmp5;
				g_cdata[ix+iy*wx+6*stride] = tmp6;
				g_cdata[ix+iy*wx+7*stride] = tmp7;
			}
		}
	}

	generic_benchmark_result_list run_dram_benchmark(void) {
		constexpr unsigned int VECTOR_SIZE = 8*1024*1024;
		const int GRANULARITY = 8;
		const int BLOCK_SIZE = 256;
		const int TOTAL_BLOCKS = VECTOR_SIZE/(BLOCK_SIZE*GRANULARITY);

		double *ad, *cd;

		CUDA_SAFE_CALL( cudaMalloc((void**)&ad, VECTOR_SIZE*sizeof(double)) );
		CUDA_SAFE_CALL( cudaMalloc((void**)&cd, VECTOR_SIZE*sizeof(double)) );

		// Copy data to device memory
		CUDA_SAFE_CALL( cudaMemset(ad, 1, VECTOR_SIZE*sizeof(double)) );  // initialize to zeros
		CUDA_SAFE_CALL( cudaMemset(cd, 0, VECTOR_SIZE*sizeof(double)) );  // initialize to zeros

		// Synchronize in order to wait for memory operations to finish
		CUDA_SAFE_CALL( cudaThreadSynchronize() );

		if( BLOCK_SIZE % 32 != 0 || TOTAL_BLOCKS % (8192/32) != 0 ){
			exit(1);
		}
		dim3 dimBlock(BLOCK_SIZE, 1, 1), dimBlock2D(32, BLOCK_SIZE / 32, 1);
		dim3 dimGrid_double(TOTAL_BLOCKS, 1, 1), dimGrid2D(8192/32, TOTAL_BLOCKS/(8192/32), 1);
		const int TOTAL_BLOCKS_FLOAT = TOTAL_BLOCKS/sizeof(float)*sizeof(double), TOTAL_BLOCKS_INT4 = TOTAL_BLOCKS/sizeof(int4)*sizeof(double);
		const int TOTAL_BLOCKS_CHAR = TOTAL_BLOCKS_FLOAT, TOTAL_BLOCKS_SHORT = TOTAL_BLOCKS_FLOAT;
		dim3 dimGrid_float(TOTAL_BLOCKS_FLOAT, 1, 1), dimGrid2D_float(8192/32, TOTAL_BLOCKS_FLOAT/(8192/32), 1);
		dim3 dimGrid_i4(TOTAL_BLOCKS_INT4, 1, 1), dimGrid2D_i4(8192/32, TOTAL_BLOCKS_INT4/(8192/32), 1);
		dim3 dimGrid_char(TOTAL_BLOCKS_CHAR, 1, 1);
		dim3 dimGrid_short(TOTAL_BLOCKS_SHORT, 1, 1);
		cudaEvent_t start, stop;

		// Bind textures to buffer
		cudaBindTexture(0, texdataF,  ad, VECTOR_SIZE*sizeof(double));
		cudaBindTexture(0, texdataD,  ad, VECTOR_SIZE*sizeof(double));
		cudaBindTexture(0, texdataI4, ad, VECTOR_SIZE*sizeof(double));

		initializeEvents(&start, &stop);
		kmemaccess<char, GRANULARITY, true, true, false><<< dimGrid_char, dimBlock >>>(static_cast<char>(constVal), reinterpret_cast<char*>(ad), reinterpret_cast<char*>(cd));
		float kernel_time_vadd_ch = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<char, GRANULARITY, true, false, false><<< dimGrid_char, dimBlock >>>(static_cast<char>(constVal), reinterpret_cast<char*>(ad), reinterpret_cast<char*>(cd));
		float kernel_time_vget_ch = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<char, GRANULARITY, false, true, false><<< dimGrid_char, dimBlock >>>(static_cast<char>(constVal), reinterpret_cast<char*>(ad), reinterpret_cast<char*>(cd));
		float kernel_time_vset_ch = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<short, GRANULARITY, true, true, false><<< dimGrid_short, dimBlock >>>(static_cast<short>(constVal), reinterpret_cast<short*>(ad), reinterpret_cast<short*>(cd));
		float kernel_time_vadd_sh = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<short, GRANULARITY, true, false, false><<< dimGrid_short, dimBlock >>>(static_cast<short>(constVal), reinterpret_cast<short*>(ad), reinterpret_cast<short*>(cd));
		float kernel_time_vget_sh = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<short, GRANULARITY, false, true, false><<< dimGrid_short, dimBlock >>>(static_cast<short>(constVal), reinterpret_cast<short*>(ad), reinterpret_cast<short*>(cd));
		float kernel_time_vset_sh = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<double, GRANULARITY, true, true, false><<< dimGrid_double, dimBlock >>>(constVal, ad, cd);
		float kernel_time_vadd_dp = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<double, GRANULARITY, true, false, false><<< dimGrid_double, dimBlock >>>(constVal, ad, cd);
		float kernel_time_vget_dp = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<double, GRANULARITY, false, true, false><<< dimGrid_double, dimBlock >>>(constVal, ad, cd);
		float kernel_time_vset_dp = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<double, GRANULARITY, true, true, true><<< dimGrid_double, dimBlock >>>(constVal, ad, cd);
		float kernel_time_vcpytex_dp = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<double, GRANULARITY, true, false, true><<< dimGrid_double, dimBlock >>>(constVal, ad, cd);
		float kernel_time_vgettex_dp = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<float, GRANULARITY, true, true, false><<< dimGrid_float, dimBlock >>>(constValf, reinterpret_cast<float*>(ad), reinterpret_cast<float*>(cd));
		float kernel_time_vadd_sp = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<float, GRANULARITY, true, false, false><<< dimGrid_float, dimBlock >>>(constValf, reinterpret_cast<float*>(ad), reinterpret_cast<float*>(cd));
		float kernel_time_vget_sp = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<float, GRANULARITY, false, true, false><<< dimGrid_float, dimBlock >>>(constValf, reinterpret_cast<float*>(ad), reinterpret_cast<float*>(cd));
		float kernel_time_vset_sp = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<float, GRANULARITY, true, true, true><<< dimGrid_float, dimBlock >>>(constValf, reinterpret_cast<float*>(ad), reinterpret_cast<float*>(cd));
		float kernel_time_vcpytex_sp = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<float, GRANULARITY, true, false, true><<< dimGrid_float, dimBlock >>>(constValf, reinterpret_cast<float*>(ad), reinterpret_cast<float*>(cd));
		float kernel_time_vgettex_sp = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<int4, GRANULARITY, true, true, false><<< dimGrid_i4, dimBlock >>>(constVali4, reinterpret_cast<int4*>(ad), reinterpret_cast<int4*>(cd));
		float kernel_time_vadd_i4 = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<int4, GRANULARITY, true, false, false><<< dimGrid_i4, dimBlock >>>(constVali4, reinterpret_cast<int4*>(ad), reinterpret_cast<int4*>(cd));
		float kernel_time_vget_i4 = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<int4, GRANULARITY, false, true, false><<< dimGrid_i4, dimBlock >>>(constVali4, reinterpret_cast<int4*>(ad), reinterpret_cast<int4*>(cd));
		float kernel_time_vset_i4 = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<int4, GRANULARITY, true, true, true><<< dimGrid_i4, dimBlock >>>(constVali4, reinterpret_cast<int4*>(ad), reinterpret_cast<int4*>(cd));
		float kernel_time_vcpytex_i4 = finalizeEvents(start, stop);

		initializeEvents(&start, &stop);
		kmemaccess<int4, GRANULARITY, true, false, true><<< dimGrid_i4, dimBlock >>>(constVali4, reinterpret_cast<int4*>(ad), reinterpret_cast<int4*>(cd));
		float kernel_time_vgettex_i4 = finalizeEvents(start, stop);

		// Unbind textures
		cudaUnbindTexture(texdataI4);
		cudaUnbindTexture(texdataF);
		cudaUnbindTexture(texdataD);

		// Copy results back to host memory (not needed)
		//CUDA_SAFE_CALL( cudaMemcpy(c, cd, VECTOR_SIZE*sizeof(double), cudaMemcpyDeviceToHost) );

		cudaArray* cudaArrayA;
		CUDA_SAFE_CALL( cudaMallocArray(&cudaArrayA, &texdataF_2D.channelDesc, dimGrid2D_float.x * dimBlock2D.x, dimGrid2D_float.y * dimBlock2D.y * GRANULARITY) );
		CUDA_SAFE_CALL( cudaMemcpyToArray(cudaArrayA, 0, 0, ad, VECTOR_SIZE*sizeof(float), cudaMemcpyDeviceToDevice) );

		texdataF_2D.addressMode[0] = cudaAddressModeBorder;
		texdataF_2D.addressMode[1] = cudaAddressModeBorder;
		texdataF_2D.filterMode = cudaFilterModePoint;
		texdataF_2D.normalized = false;

		CUDA_SAFE_CALL( cudaBindTextureToArray(texdataF_2D, cudaArrayA, texdataF_2D.channelDesc) );

		initializeEvents(&start, &stop);
		kmemaccess2D<float><<< dimGrid2D_float, dimBlock2D >>>(constValf, reinterpret_cast<float*>(cd));
		float kernel_time_2d_tex_f = finalizeEvents(start, stop);

		CUDA_SAFE_CALL( cudaUnbindTexture( texdataF_2D ) );
		CUDA_SAFE_CALL( cudaFreeArray(cudaArrayA) );

		CUDA_SAFE_CALL( cudaMallocArray(&cudaArrayA, &texdataD_2D.channelDesc, dimGrid2D.x * dimBlock2D.x, dimGrid2D.y * dimBlock2D.y * GRANULARITY) );
		CUDA_SAFE_CALL( cudaMemcpyToArray(cudaArrayA, 0, 0, ad, VECTOR_SIZE*sizeof(double), cudaMemcpyDeviceToDevice) );

		texdataD_2D.addressMode[0] = cudaAddressModeBorder;
		texdataD_2D.addressMode[1] = cudaAddressModeBorder;
		texdataD_2D.filterMode = cudaFilterModePoint;
		texdataD_2D.normalized = false;

		CUDA_SAFE_CALL( cudaBindTextureToArray(texdataD_2D, cudaArrayA, texdataD_2D.channelDesc) );

		initializeEvents(&start, &stop);
		kmemaccess2D<double><<< dimGrid2D, dimBlock2D >>>(constVal, reinterpret_cast<double*>(cd));
		float kernel_time_2d_tex_d = finalizeEvents(start, stop);

		CUDA_SAFE_CALL( cudaUnbindTexture( texdataD_2D ) );
		CUDA_SAFE_CALL( cudaFreeArray(cudaArrayA) );


		CUDA_SAFE_CALL( cudaFree(ad) );
		CUDA_SAFE_CALL( cudaFree(cd) );

		generic_benchmark_result_list results;
		results["BW_DRAM"] = std::max( {
				// double precision vector addition, read & write
				((2.0*VECTOR_SIZE*sizeof(double))/kernel_time_vadd_dp*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vget_dp*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vset_dp*1000./(1000.0*1000*1000))/3.0,
				// double precision vector texture copy, texture read & write
				((2.0*VECTOR_SIZE*sizeof(double))/kernel_time_vcpytex_dp*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vgettex_dp*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vset_dp*1000./(1000.0*1000*1000))/3.0,
				// single precision vector addition, read & write
				((2.0*VECTOR_SIZE*sizeof(double))/kernel_time_vadd_sp*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vget_sp*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vset_sp*1000./(1000.0*1000*1000))/3.0,
				// single precision vector texture copy, texture read & write
				((2.0*VECTOR_SIZE*sizeof(double))/kernel_time_vcpytex_sp*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vgettex_sp*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vset_sp*1000./(1000.0*1000*1000))/3.0,
				// int4 vector addition, read & write
				((2.0*VECTOR_SIZE*sizeof(double))/kernel_time_vadd_i4*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vget_i4*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vset_i4*1000./(1000.0*1000*1000))/3.0,
				// int4 vector texture copy, texture read & write
				((2.0*VECTOR_SIZE*sizeof(double))/kernel_time_vcpytex_i4*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vgettex_i4*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vset_i4*1000./(1000.0*1000*1000))/3.0,
				// short int vector addition, read & write
				((2.0*VECTOR_SIZE*sizeof(double))/kernel_time_vadd_sh*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vget_sh*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vset_sh*1000./(1000.0*1000*1000))/(3.0*2.0),
				// char vector addition, read & write
				((2.0*VECTOR_SIZE*sizeof(double))/kernel_time_vadd_ch*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vget_ch*1000./(1000.0*1000*1000) +
				 (1.0*VECTOR_SIZE*sizeof(double))/kernel_time_vset_ch*1000./(1000.0*1000*1000))/(3.0*4.0)
			} );
		return results;
	}

}
