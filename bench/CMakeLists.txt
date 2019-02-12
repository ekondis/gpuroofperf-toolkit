cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpuroofperf-bench LANGUAGES CXX CUDA)

# Include CUDA header directory in cpp files
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets")
string(APPEND CMAKE_CUDA_FLAGS " --cudart=static")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_30,code=compute_30")

add_executable(gpuroofperf-bench main.cpp benchmark_manager.cpp benchmark_manager.h computation_kernels.cu computation_kernels.h shmem_kernels.cu shmem_kernels.h dram_kernels.cu dram_kernels.h cache_kernels.cu cache_kernels.h cuda_common.h kernels_common.h)
target_compile_features(gpuroofperf-bench PUBLIC cxx_std_11)

if(APPLE)
	# We need to add the default path to the driver (libcuda.dylib) as an rpath,
	# so that the static cuda runtime can find it at runtime.
	set_property(TARGET gpuroofperf-bench PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
