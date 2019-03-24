/**
* main.cpp: This file is part of the gpuroofperf-bench benchmark suite.
*
* Contact: Elias Konstantinidis <ekondis@gmail.com>
**/

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include "benchmark_manager.h"

using namespace std;

typedef decltype(benchmark_manager::query_devices()) device_list;
using benchmark_results = benchmark_manager::benchmark_measurements;

// print help information
void print_help(const device_list &devices) {
	cout <<
		"Syntax: gpuroofbench device_index [options]" << endl << endl <<
		"Command line options:" << endl <<
		"-h or --help:            Show this help" << endl <<
		"-o or --output=filename: Append CSV output to file" << endl <<
		endl;
	cout << endl << "Valid device indexes:" << endl;
	for (const auto& dev : devices) {
		cout << '[' << dev.first+1 << "] " << dev.second << endl;
	}
}

// parse command line arguments
int parse_command_line_args(const char **begin, const char **end, int &device_id, string &fname, bool &show_help){
	// iterate over arguments
	for(auto iter=begin; iter!=end; iter++){
		string argument{*iter};
		int argument_as_int;
		stringstream ss;
		ss << argument;
		ss >> argument_as_int;
		if(!ss.fail()) {
			// argument is integer, thus defining device id
			device_id = argument_as_int-1;
			continue;
		}
		if( argument=="-h" || argument=="--help"  ){
			show_help = true;
			continue;
		}
		// keep filename for output
		if( argument=="-o" ){
			iter++;
			if(iter==end){
				cerr << "ERROR: No filename supplied" << endl;
				return 1;
			}
			fname = *iter;
			continue;
		}
		if( argument.rfind("--output=", 0) == 0 ){
			argument = argument.substr(9);
			if(argument==""){
				cerr << "ERROR: No filename supplied" << endl;
				return 1;
			}
			fname = argument;
			continue;
		}
		cerr << "ERROR: Invalid argument: '" << argument << '\'' << endl;
		return 1;
	}
	return 0;
}

// check device validity
bool isDeviceValid(const int i_device, const device_list& devices){
	if (i_device==-1)
		return true; // no device selected is considered as valid
	// check device index validity
	bool valid_index = false;
	//for_each(devices.begin(), devices.end(), [i_device, &valid_index](const benchmark_manager::device_list_element& v) {if (v.first == i_device) valid_index = true; });
	for(const auto& v:devices)
		if (v.first == i_device) valid_index = true;
	return valid_index;
}

// print results
void print_results(const string &dev_name, const benchmark_results& res){
	// print output measurements
	cout << endl << "Execution results for '" << dev_name << "':" << endl;
	cout << fixed << setprecision(3) <<
		"GFLOPS - SP:  " << setw(10) << res.tp_gflops_sp << endl <<
		"GFLOPS - DP:  " << setw(10) << res.tp_gflops_dp << endl <<
		"GIOPS - MAD:  " << setw(10) << res.tp_giops << endl <<
		"GBSEC - DRAM: " << setw(10) << res.bw_dram << endl <<
		"GBSEC - L2:   " << setw(10) << res.bw_l2 << endl <<
		"GOPS - SHMEM: " << setw(10) << res.tp_ldst << endl <<
		"GIOPS - ADD:  " << setw(10) << res.tp_iadd << endl << endl;
}

// save results to CSV output file stream
int save_results(const string &file_name, const string &dev_name, const benchmark_results& res){
	constexpr char HEADER[] = {"\"GPU\", \"GFLOPS - SP\", \"GFLOPS - DP\", \"GIOPS - MAD\", \"GBSEC - DRAM\", \"GBSEC - L2\", \"GOPS - SHMEM\", \"GIOPS - ADD\""};
	ifstream reader{file_name};
	bool append_header = !reader.good();
	if (!append_header) {
		string line;
		getline(reader, line);
		if (line.compare(HEADER)!=0)
			cerr << "WARNING: First line of '" << file_name << "' seems not to be an appropriate header line." << endl;
	}

	// append benchmark results
	ofstream writer{file_name, ios::app};
	if( !writer ){
		cerr << "ERROR: Cannot open '" << file_name << "'";
		return 1;
	}
	if (append_header)
		writer << HEADER << endl;
	writer << '"' << dev_name << "\", " << setprecision(4) << fixed <<
		res.tp_gflops_sp << ", " <<
		res.tp_gflops_dp << ", " <<
		res.tp_giops << ", " <<
		res.bw_dram << ", " <<
		res.bw_l2 << ", " <<
		res.tp_ldst << ", " <<
		res.tp_iadd << endl;
	return 0;
}

int main(int argc, const char *argv[]) {
	const string VERSION = "v.0.9.0";
	cout << "gpuroofperf-bench (CUDA) microbenchmarking tool " << VERSION << endl << endl;

	bool show_help = false;
	int i_device = -1;
	string fn_output{""};
	int retcode = parse_command_line_args(argv+1, argv+argc, i_device, fn_output, show_help);
	if( retcode )
		return retcode;

	auto devices = benchmark_manager::query_devices();
	if (!isDeviceValid(i_device, devices)) {
		cerr << "ERROR: Invalid device index (" << i_device << ")" << endl;
		print_help(devices);
		return 1;
	}

	// output help information
	if (show_help || i_device==-1) {
		print_help(devices);
		return 1;
	}

	// proceed to benchmarks
	benchmark_manager manager{i_device};

	// print CUDA device information
	manager.get_device_info(cout);

	// launch benchmarks
	cout << std::endl;
	cout << "Running compute intensive benchmarks...    " << std::flush;
	manager.run_computations();
	manager.run_computations();
	manager.run_computations();
	cout << "Done" << std::endl;
	cout << "Running load/store intensive benchmarks... " << std::flush;
	manager.run_shmem_ops();
	manager.run_shmem_ops();
	manager.run_shmem_ops();
	cout << "Done" << std::endl;
	cout << "Running memory intensive benchmarks...     " << std::flush;
	manager.run_memory_ops();
	manager.run_memory_ops();
	manager.run_memory_ops();
	cout << "Done" << std::endl;
	cout << "Running L2 cache intensive benchmarks...   " << std::flush;
	manager.run_cache_ops();
	manager.run_cache_ops();
	manager.run_cache_ops();
	cout << "Done" << std::endl;

	// show and write output results
	const auto& res = manager.get_results();
	print_results(manager.get_device_name(), res);
	if (fn_output!="") {
		int retcode = save_results(fn_output, manager.get_device_name(), res);
		if (retcode)
			return retcode;
		cout << "Results appended to file '" << fn_output << "'" << std::endl;
	}

	return 0;
}
