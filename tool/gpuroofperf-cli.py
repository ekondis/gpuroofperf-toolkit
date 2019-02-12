#!/usr/bin/python3

"""
gpuroofperf-tool (CLI, version 0.9.0)

A command line tool implementing the proposed performance model.
"""

import sys, getopt
import kernel_inspector as ki
import profiling_executor
import performance_predictor
import gpu_parameter_loader
from kernel_parameters import KernelParameters

class gpuroofperftool_CLI:
	VERSION = "0.9.0"

	def __init__(self, args):
		""" Default parameters """
		self.cmdargs = args
		self.path_nvprof = None
		self.gpu_params = None
		self.output_params = None
		self.input_params = None
		self.output_prediction = None
		self.use_csv = False
		self.nvprof_check = False
		return

	def showHelp(self):
		""" Displays command line options """
		print("usage: gpuroofperf-cli.py [options...] [invocation]")
		print()
		print("where:")
		print("\t-h, --help              Show this information")
		print("\t-o, --output=FILE       Save extracted kernel parameters to output FILE (kernel inspection mode)")
		print("\t-g, --gpuspecs=FILE     Use GPU parameters as input from FILE")
		print("\t-i, --input=FILE        Kernel parameters in FILE will be used as input (enables performance estimation mode)")
		print("\t-p, --profpath=FILE     nvprof full path is set to FILE")
		print("\t-s, --prediction=FILE   Save prediction data to FILE (JSON default format)")
		print("\t-c, --csv               Use CSV format for saving data instead of JSON")
		print("\t-v, --nvprof-check      Check if both nvprof tool and CUDA device function properly")
		print()
		print("The tool works in two different modes (kernel inspection/performance estimation). It can work in either "\
			"one or both depending on the provided arguments:\n"\
			"1) Providing an invocation argument enables the kernel inspection mode.\n"\
			"2) Providing GPU parameters through a kernel parameter file (-g. --gpuspecs argument) "\
			"enables performance estimation mode.\n")
		print("Examples:")
		print("\tgpuroofperf-cli.py --output=\"myprof.json\" ./my_cuda_app")
		print("\tRuns profiling on the execution of my_cuda_app and saves extracted kernel parameters (\"myprof.json\")")
		print()
		print("\tgpuroofperf-cli.py --input=\"myprof.json\" --gpuspecs=\"myspecs.csv\" --prediction=\"pred.csv\" -c")
		print("\tPerforms performance estimation based on input kernel parameters (\"myprof.json\") and GPU parameters (\"myspecs.csv\") and saves prediction data in CSV format (\"pred.csv\")")
		return

	def check_nvprof(self):
		"""Check if NVidia profiler is accessible"""
		try:
			extractor = ki.KernelParamExtractor(self.invocation, self.path_nvprof)
			gpumetrics = extractor.retrieveGPUInfo()
		except ki.NoneGPUsException as e:
			print('Error:',str(e), file=sys.stderr)
		except Exception as e:
			print('Error:',str(e), file=sys.stderr)
			print('Is nvprof tool actually installed?', file=sys.stderr)
			sys.exit(2)
		if len(gpumetrics)==0:
			print('Error: CUDA GPU not found! Though, nvprof tool seems functional.', file=sys.stderr)
			sys.exit(2)
		print("Good news! nvprof seems to be functioning correctly and CUDA devices are also found (%s)." % (', '.join(['"'+g['device']+'"' for g in gpumetrics])))
		return

	def check_for_warning_info(self, params):
		"""Checks for various warnings undermining the prediction effectiveness"""
		high_l2_kernels = []
		high_num_calls_kernels = []
		low_utilization_kernels = []
		for k,v in params.data.items():
			warnings = []
			if v['l2_bytes'] > 2*v['dram_bytes']:
				warnings.append( 'Warning: L2 traffic observed to be significantly higher than DRAM traffic ({} MB vs {} MB). '\
					'In this case the actual performance could be L2 cache bound '\
					'instead of DRAM bound in case the memory subsystem is a performance '\
					'limitation factor.'.format(v['l2_bytes']/1000000., v['dram_bytes']/1000000.) )
			if v['reference_time']/v['count'] < 0.1:
				warnings.append( 'Warning: Significantly low execution time per kernel invocation ({} msecs for {} invocations). '\
					'Unpredictable overheads be non negligible in situations where the '\
					'execution time of a kernel invocation gets very short.'.format(v['reference_time'], v['count']) )
			max_utilization = max( v['utilizations'].values() )
			if max_utilization < 5:
				max_utilization_metrics = {k:v for k,v in v['utilizations'].items() if v==max_utilization}.keys()
				warnings.append( 'Warning: Low utilizations observed (highest utilizations {}:{}). '\
					'Low utilizations potentially express other latencies that cannot be captured by the model.'.format(', '.join(max_utilization_metrics), max_utilization) )
			if len(warnings)>0:
				print('\n{}\n{}'.format(k,'\n'.join(warnings)), file=sys.stderr)
		print()

	def print_parameters(self, params):
		print("\n------- Primary kernel parameters -------")
		for k,p in params.data.items():
			print('Kernel name: {}'.format(k))
			kernel_type_desc = {'fp_32': 'Floating point (SP)',
			                    'fp_64': 'Floating point (DP)',
			                    'fp_16': 'Floating point (HP)',
								'integer': 'Integer'}
			print("\tKernel type:            {}".format(kernel_type_desc[ p['dominant_op'] ]))
			print("\tCompute operations:     {} ops".format(p['compute_ops']))
			print("\tDRAM accesses:          {} bytes".format(p['dram_bytes']))
			print("\tComputation efficiency: {:.5f}".format(p['opmix_efficiency']))
			print('\tInstruction mix:')
			print("\t\tComputations:  {:6.2f}%".format(100.*p['mix_compute']))
			print("\t\tLoad/Stores:   {:6.2f}%".format(100.*p['mix_ldst']))
			print("\t\tOther:         {:6.2f}%".format(100.*(1.0-p['mix_ldst']-p['mix_compute'])))
			print()

	def kernelInspection(self):
		# Extract gpu metrics and select GPU if required
		extractor = ki.KernelParamExtractor(self.invocation, self.path_nvprof, self.VERSION)
		try:
			gpumetrics = extractor.retrieveGPUInfo()
		except ki.NoneGPUsException as e:
			print('Error:',str(e), file=sys.stderr)
		# Choose device index when more than 1 are available
		if len(gpumetrics)>1:
			print('Multiple GPUs found. Please choose one of them below:')
			for i, gpu in enumerate(g['device'] for g in gpumetrics):
				print('%d. %s' % (i+1, gpu))
			gpuindex = -1
			while not 1<=gpuindex<=2:
				gpuindex = int(input('Select GPU index (1-{}):'.format(len(gpumetrics))))
			extractor.selected_device = gpuindex-1
		else:
			extractor.selected_device = 0
		print("Profiling on GPU: %s" % (gpumetrics[extractor.selected_device]['device']))
		print("Invocation: \""+self.invocation+"\"")
		# Simple profiling
		simpleprofile = extractor.simpleProfiling()
		kernels_by_time = sorted([ [k, float(v['Time(%)']), float(v['Time']), int(v['Calls'])] for k,v in simpleprofile.items()], key=lambda v: v[1], reverse=True)
		# Select kernels under inspection
		subject_kernels = set()
		while not subject_kernels:
			print('GPU kernel functions invoked:')
			for i, kernel in enumerate(kernels_by_time):
				print('%d. %s (Time(%%):%.2f, %.4f msecs, %d invocations)' % (i+1, ki.strip_parenthesis(kernel[0]), kernel[1], kernel[2], kernel[3]))
			userfeedback = input('Please give the subject kernel indices (comma separated) (1-{} or default:all kernels):'.format(len(kernels_by_time))).strip()
			if userfeedback.strip()=='':
				subject_kernels.update( next(zip(*kernels_by_time)) )
				continue
			subject_kernel_indexes = list(map(int, userfeedback.split(',')))
			if not all(map(lambda i:1<=i<=len(kernels_by_time), subject_kernel_indexes)):
				print('Error: Invalid index')
				continue
			subject_kernels = {kernels_by_time[i-1][0] for i in subject_kernel_indexes}
		print('Selected kernels: {}'.format(', '.join(('"{}"'.format(ki.strip_parenthesis(s)) for s in subject_kernels))))
		extractor.setSubjectKernels(subject_kernels)
		# Trace profiling
		extractor.traceProfiling(subject_kernels)
		# Hardware metric profiling
		extractor.flopProfiling(subject_kernels)
		extractor.memoryProfiling(subject_kernels)
		extractor.instructionProfiling(subject_kernels)
		extractor.utilizationProfiling(subject_kernels)
		print("Kernel inspection done!")
		params = extractor.get_params()
		self.check_for_warning_info(params)
		if self.output_params is not None:
			params.save(self.output_params)
			print("Kernel parameters saved to  \"{}\"".format(self.output_params))
		return params

	def performancePrediction(self, params):
		print("Reading GPU specifications from \"{}\"".format(self.gpu_params))
		gpuspecs = gpu_parameter_loader.gpuParamLoader(self.gpu_params)
		gpuspecs.load()
		print("Estimating GPU performance for {} GPUs ({})".format(len(gpuspecs.parameters), ', '.join(gpuspecs.parameters.keys())))
		predictor = performance_predictor.perfPredictor(params, gpuspecs.parameters)
		prediction = predictor.execute()
		print()
		# Output performance prediction
		prediction.save_to_tabular(sys.stdout)
		if self.output_prediction:
			print("Saving prediction results to \"{}\"".format(self.output_prediction))
			with open(self.output_prediction, 'w') as output_file:
				if self.use_csv:
					prediction.save_to_csv(output_file)
				else:
					prediction.save_to_json(output_file)

	def run(self):
		# Parse command line arguments
		print("gpuroofperf-tool v.%s (CLI)\n" % (self.VERSION))
		if( len(self.cmdargs)==0 ):
			self.showHelp()
			return
		try:
			opts, args = getopt.getopt(self.cmdargs, "hp:o:i:g:s:cv", ["profpath=","output=","gpuspecs=","input=","prediction=","csv","nvprof-check"])
		except getopt.GetoptError:
			self.showHelp()
			sys.exit(2)
		for opt, arg in opts:
			if opt in ('-h', "--help"):
				self.showHelp()
				sys.exit()
			elif opt in ("-p", "--profpath"):
				self.path_nvprof = arg
			elif opt in ("-o", "--output"):
				self.output_params = arg
			elif opt in ("-g", "--gpuspecs"):
				self.gpu_params = arg
			elif opt in ("-i", "--input"):
				self.input_params = arg
			elif opt in ("-s", "--prediction"):
				self.output_prediction = arg
			elif opt in ("-c", "--csv"):
				self.use_csv = True
			elif opt in ("-v", "--nvprof-check"):
				self.nvprof_check = True
		self.invocation = ' '.join(args).strip()
		self.mode_kernel_inspection = self.invocation != ""
		self.mode_prediction = self.gpu_params and ((self.input_params is not None) or (self.invocation))
		# just verify nvprof functionality
		if self.nvprof_check:
			self.check_nvprof()
			return
		# check for prohibited option combinations
		if( (not self.mode_prediction) and (not self.mode_kernel_inspection) ):
			print("Error: No GPU invocation nor input parameters provided", file=sys.stderr)
			sys.exit(2)
		if( self.input_params and self.invocation ):
			print("Error: Cannot provide both input parameters and invocation", file=sys.stderr)
			sys.exit(2)
		try:
			# Conduct kernel inspection
			if self.mode_kernel_inspection:
				print("\n------- Running kernel inspection -------")
				params = self.kernelInspection()
			elif self.input_params:
				params = KernelParameters()
				params.load(self.input_params)
				print("Retrieved kernel parameters from \"{}\"".format(self.input_params))
			# Print kernel information
			self.print_parameters(params)
			# Conduct performance prediction
			if self.mode_prediction:
				print("\n--- Conducting performance estimation ---")
				self.performancePrediction(params)
		except profiling_executor.profException as e:
			print("Error: %s" % (e), file=sys.stderr)

if __name__ == "__main__":
	app = gpuroofperftool_CLI(sys.argv[1:])
	app.run()
