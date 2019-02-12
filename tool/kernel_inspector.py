#!/usr/bin/python3

"""
KernelParamExtractor is built for extracting kernel parameter values
by executing the GPU application through the GPU profiler
"""

import profiling_executor
from kernel_parameters import KernelParameters
import re
import csv
import json, datetime, os, platform

def strip_parenthesis(s, ret_argument=False):
	m = re.search('(.+)\((.+?)\)', s)
	return m.group(2 if ret_argument else 1).strip() if m else s

class NoneGPUsException(Exception):
    pass

class KernelParamExtractor:
	IGNORED_CALLS = ('', '[CUDA memcpy HtoD]', '[CUDA memcpy DtoH]', '[CUDA memset]')

	def __init__(self, invocation, nvprofpath=None, appver=None):
		self.invocation = invocation
		self.executor = profiling_executor.profExecutor(nvprofpath)
		self.selected_device = None
		self.gpumetrics = []
		self.subject_kernels = None
		self.simpleprofile = {}
		self.traceprofile = {}
		self.metricprofile = {}
		self.utilizationprofile = {}
		self.appversion = appver
		self.driver = None

	# Get kernel parameters out of metric data
	def get_params(self):
		if self.driver is None:
			self.driver = self.executor.query_driver()
		res = KernelParameters({
				'app': os.path.basename(self.invocation.split(' ', 1)[0]),
				'invocation': self.invocation,
				'datetime': str(datetime.datetime.now()),
				'device': self.gpumetrics[self.selected_device]['device'],
				'driver': self.driver,
				'platform': platform.platform(),
				'version': self.appversion,
				'data': {}
			})
		for kernel in self.subject_kernels:
			# retrieve invocation count
			invocation_count = {
				int(v)
				for i,v in enumerate(self.metricprofile['Invocations'])
				if self.metricprofile['Kernel'][i] == kernel
			}
			if len(invocation_count)>1:
				raise Exception("Only one invocation count expected ({})".format(invocation_count))
			invocation_count = next(iter(invocation_count))
			# filter records for current kernel and apply invocation counts
			metric_counts = {
				v: int(self.metricprofile['Avg'][i]) * int(self.metricprofile['Invocations'][i])
				for i,v in enumerate(self.metricprofile['Metric Name'])
				if self.metricprofile['Kernel'][i] == kernel
			}
			# similarly preprocess utilizations
			utilizations = {
				v: int(strip_parenthesis(self.utilizationprofile['Avg'][i], ret_argument=True))
				for i,v in enumerate(self.utilizationprofile['Metric Name'])
				if self.utilizationprofile['Kernel'][i] == kernel
			}
			# find dominant operation (fp32, fp64, fp16?)
			metric_fp_ops = {k[-5:]:v for k,v in metric_counts.items() if k.startswith('inst_fp_')}
			dominant_op = \
				'integer' if sum(metric_fp_ops.values())==0 else \
				max(metric_fp_ops.keys(), key=lambda k: metric_fp_ops[k])
			#	'fp_32' if metric_counts['inst_fp_32']>metric_counts['inst_fp_64'] else \
			#	'fp_64' if metric_counts['inst_fp_64']>metric_counts['inst_fp_16'] else \
			#	'fp_16';
			inst_compute = metric_counts['inst_'+dominant_op]
			metric_fmas = {
				'fp_32': metric_counts['flop_count_sp_fma'],
				'fp_64': metric_counts['flop_count_dp_fma'],
				'fp_16': metric_counts.get('flop_count_hp_fma', 0),
				'integer': 0
			}[dominant_op]
			feature_compute_ops = inst_compute + metric_fmas
			feature_dram_bytes = (metric_counts['dram_read_transactions']+metric_counts['dram_write_transactions'])*32
			feature_l2_bytes = (metric_counts['l2_read_transactions']+metric_counts['l2_write_transactions'])*32
			try:
				feature_op_mix_efficiency = feature_compute_ops/(2*(feature_compute_ops-metric_fmas))
			except ZeroDivisionError as e:
				feature_op_mix_efficiency = 0
			inst_compute_ld_st = metric_counts['inst_compute_ld_st']
			inst_total = metric_counts['inst_executed'] * 32
			feature_inst_op_pctg = inst_compute / inst_total
			feature_inst_ldst_pctg = inst_compute_ld_st / inst_total
			res[kernel] = {
				'dominant_op': dominant_op,
				'compute_ops': feature_compute_ops,
				'dram_bytes': feature_dram_bytes,
				'l2_bytes': feature_l2_bytes,
				'opmix_efficiency': feature_op_mix_efficiency,
				'mix_compute': feature_inst_op_pctg,
				'mix_ldst': feature_inst_ldst_pctg,
				'reference_time': float(self.simpleprofile[kernel]['Time']),
				'count': invocation_count,
				'utilizations': utilizations
			}
		return res

	# Retrieve GPU metrics for available devices
	def retrieveGPUInfo(self):
		(lines_out, lines_err) = self.executor.execute(['--devices', 'all', '--query-metrics'], 'Retrieving GPU information')
		lines_out = map(lambda l:l.strip(), lines_out)
		lines_out = filter(lambda l:not l=='', lines_out)
		self.gpumetrics = []
		self.selected_device = None
		self.simpleprofile = {}
		metrics = None
		for line in lines_out:
			m = re.search('Device [0-9]+ \((.+?)\)', line)
			if m:
				self.gpumetrics.append( {'device': m.group(1), 'metrics':{}} )
				metrics = self.gpumetrics[-1]
			else:
				metric = line.split(':')
				metric = list(map(lambda l:l.strip(), metric))
				if len(metric)==2 and metrics is not None:
					metrics[ 'metrics' ][ metric[0] ] = metric[1]
		if len(self.gpumetrics)<1:
			raise NoneGPUsException('No GPUs identified')
		return self.gpumetrics

	def simpleProfiling(self):
		(lines_out, lines_err) = self.executor.execute(['-u', 'ms', '--csv']+self.invocation.split(), 'Running simple profiling', self.selected_device)
		# Process CSV rows and exclude unwanted rows
		filtereddata = []
		for row in csv.reader(lines_err):
			if len(row)>=7:
				# Ignore rows when name is empty, [CUDA memcpy HtoD], etc.
				if row[-1] in KernelParamExtractor.IGNORED_CALLS:
					continue
				filtereddata += [row]
			elif len(row)==0:
				# Stop at first empty row
				break
		# Convert filtered list data to dictionary using header row as keys
		dictdata = {}
		for row in zip(*filtereddata):
			dictdata[row[0]] = list(row[1:])
		# Convert to 2 level dictionary using 'Name' column values as key
		KEY_COLUMN = 'Name'
		results = {}
		for i, name in enumerate(dictdata[KEY_COLUMN]):
			if 'Type' in dictdata and dictdata['Type'][i]=='API calls':
				continue
			results[name] = {k:v[i] for k,v in dictdata.items() if k!=KEY_COLUMN}
		self.simpleprofile = results
		return results

	def setSubjectKernels(self, kernel_set):
		if not type(kernel_set) is set:
			raise TypeError("not a set")
		if len(kernel_set-self.simpleprofile.keys())>0:
			raise Exception("Invalid kernel names")
		self.subject_kernels = kernel_set

	def traceProfiling(self, subject_kernels):
		#nvprof -u ms --print-gpu-trace $executable
		#TODO: Consider limiting trace profiling to a particular kernel (--kernels xx)
		(lines_out, lines_err) = self.executor.execute(['-u', 'ms', '--csv', '--print-gpu-trace']+self.invocation.split(), 'Running trace profiling', self.selected_device)
		# Process CSV rows and exclude unwanted rows
		filtereddata = [row for row in csv.reader(lines_err) if len(row)>6]
		# Convert filtered list data to dictionary using header row as keys
		dictdata = {row[0]:list(row[1:]) for row in zip(*filtereddata)}
		# Remove last argument with number in bracket within kernel names
		for i,kname in enumerate(dictdata['Name']):
			m = re.search('(.+)( \[.+?\])', kname)
			if m:
				dictdata['Name'][i] = m.group(1).strip()
		# Remove unwanted rows (keep only subject kernel rows)
		idx_rows = [i for i,v in enumerate(dictdata['Name']) if v in subject_kernels]
		dictdata = {k:list(map(lambda i:v[i],idx_rows)) for k,v in dictdata.items()}
		self.traceprofile = dictdata
		return dictdata

	# Enrich profile metrics with additional profiling data
	def update_metrics(self, profdata):
		for key,col in profdata.items():
			if key in self.metricprofile:
				self.metricprofile[key].extend(col)
			else:
				self.metricprofile[key] = col

	# Profile floating point operations
	def flopProfiling(self, subject_kernels):
		# Collect flop metric counters
		metrics = [ m for m in self.gpumetrics[self.selected_device]['metrics'].keys()
		            if m.startswith('flop_count_') ]
		# Invoke profiling with collected metrics
		profdata = self.__metric_profile(subject_kernels, metrics, 'floating point operations')
		self.update_metrics(profdata)
		return profdata

	# Profile memory transactions
	def memoryProfiling(self, subject_kernels):
		metrics = [ m for m in self.gpumetrics[self.selected_device]['metrics'].keys()
		            if (m.startswith('dram_') or m.startswith('l2_')) and m.endswith('_transactions') ]
		# Invoke profiling with collected metrics
		profdata = self.__metric_profile(subject_kernels, metrics, 'memory transactions')
		self.update_metrics(profdata)
		return profdata

	# Profile instruction counts executed
	def instructionProfiling(self, subject_kernels):
		UNWANTED_METRICS = ['inst_per_warp', 'inst_replay_overhead']
		metrics = [ m for m in self.gpumetrics[self.selected_device]['metrics'].keys()
		            if m.startswith('inst_') and m not in UNWANTED_METRICS ]
		# Invoke profiling with collected metrics
		profdata = self.__metric_profile(subject_kernels, metrics, 'instruction counts')
		self.update_metrics(profdata)
		return profdata

	# Profile utilizations
	def utilizationProfiling(self, subject_kernels):
		UNWANTED_METRICS = ['issue_slot_utilization']
		metrics = [ m for m in self.gpumetrics[self.selected_device]['metrics'].keys()
		            if m.endswith('_utilization') and m not in UNWANTED_METRICS ]
		# Invoke profiling with collected metrics
		profdata = self.__metric_profile(subject_kernels, metrics, 'utilization')
		self.utilizationprofile = profdata
		return profdata

	# Metric profiling helper member function
	def __metric_profile(self, subject_kernels, metrics, metric_des):
		#TODO: Consider limiting trace profiling to a particular kernel (--kernels xx)
		(lines_out, lines_err) = self.executor.execute(['-u', 'ms', '--csv', '--metrics', ','.join(metrics)]+self.invocation.split(), 'Running metric profiling ({}, {} total metrics)'.format(metric_des, len(metrics)), self.selected_device)
		# Process CSV rows and exclude unwanted rows
		filtereddata = [row for row in csv.reader(lines_err) if len(row)>6]
		# Convert filtered list data to dictionary using header row as keys
		dictdata = {row[0]:row[1:] for row in zip(*filtereddata)}
		# Remove unwanted rows (keep only subject kernel rows)
		idx_rows = [i for i,v in enumerate(dictdata['Kernel']) if v in subject_kernels]
		dictdata = {k:list(map(lambda i:v[i],idx_rows)) for k,v in dictdata.items()}
		return dictdata
