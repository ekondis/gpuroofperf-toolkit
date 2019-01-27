/**
* benchmark_manager.h: This file is part of the gpuroofperf-bench benchmark suite.
*
* Contact: Elias Konstantinidis <ekondis@gmail.com>
**/

#pragma once

#include <algorithm>
#include <string>
#include <vector>

class benchmark_manager {
public:
	typedef struct _benchmark_measurements {
		double tp_gflops_sp;
		double tp_gflops_dp;
		double tp_giops;
		double bw_dram;
		double bw_l2;
		double tp_ldst;
		double tp_iadd;
		_benchmark_measurements(void): tp_gflops_sp{-1.0}, tp_gflops_dp{-1.0}, tp_giops{-1.0}, bw_dram{-1.0}, bw_l2{-1.0}, tp_ldst{-1.0}, tp_iadd{-1.0} {
		}
		static struct _benchmark_measurements get_max(std::initializer_list<struct _benchmark_measurements> b) {
			benchmark_measurements res;
			for(auto& el:b){
				res.tp_gflops_sp = std::max(res.tp_gflops_sp, el.tp_gflops_sp);
				res.tp_gflops_dp = std::max(res.tp_gflops_dp, el.tp_gflops_dp);
				res.tp_giops     = std::max(res.tp_giops, el.tp_giops);
				res.bw_dram      = std::max(res.bw_dram, el.bw_dram);
				res.bw_l2        = std::max(res.bw_l2, el.bw_l2);
				res.tp_ldst      = std::max(res.tp_ldst, el.tp_ldst);
				res.tp_iadd      = std::max(res.tp_iadd, el.tp_iadd);
			}
			return res;
		}
	} benchmark_measurements;
private:
	benchmark_measurements measurements;
	std::string device_name;
public:
	typedef std::pair<int, std::string> device_list_element;
	typedef std::vector< device_list_element > device_list;
	benchmark_manager(int idevice);
	~benchmark_manager();
	// get string stream of device information
	void get_device_info(std::ostream& out);
	std::string get_device_name(){
		return device_name;
	}
	// returns a list of valid devices
	static device_list query_devices();
	// compute operations benchmarks
	void run_computations(void);
	// load/store operations benchmarks
	void run_shmem_ops(void);
	// memory operations benchmarks
	void run_memory_ops(void);
	// L2 cache operations benchmarks
	void run_cache_ops(void);
	// benchmark results
	const benchmark_measurements& get_results(void){
		return measurements;
	}
};
