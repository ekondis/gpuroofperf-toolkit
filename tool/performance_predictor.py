#!/usr/bin/python3

"""
perfPredictor implements the performance prediction model as described in the reference below:

Elias Konstantinidis, Yiannis Cotronis,
"A quantitative roofline model for GPU kernel performance estimation using
micro-benchmarks and hardware metric profiling",
Journal of Parallel and Distributed Computing, Volume 107, September 2017,
Pages 37-56, ISSN 0743-7315,
https://doi.org/10.1016/j.jpdc.2017.04.002.
URL: http://www.sciencedirect.com/science/article/pii/S0743731517301247
"""

import json
from math import atan
import sys

def str_padder(length):
	def padder(s):
		if len(s)>length:
			return s[:length-3]+'...'
		return s.ljust(length)
	return padder

class perfPrediction:
	def __init__(self):
		self.data = {}

	def add(self, device, prediction):
		self.data[device] = prediction

	def save_to_tabular(self, file):
		# extract kernel and gpu lists
		gpus = self.data.keys()
		kernels = next(iter(self.data.values())).keys()
		# lengths for facilitating alignments
		kernel_lens = (len(k) for k in kernels)
		max_gpu_len = max((len(g) for g in gpus))
		column_len = 32
		str_pad = str_padder(column_len)
		dash_line = "+{}+".format('-' * (max_gpu_len+2)) + ("-" * (column_len+2) + "+") * (len(kernels) + 1)
		# header row
		print(dash_line, file=file)
		print("| {} |".format('GPU'.center(max_gpu_len), 1), end='', file=file)
		for k in kernels:
			print(" {} |".format(str_pad(k)), end='', file=file)
		print(" {} |".format(str_pad('*********** Summary ***********')), end='', file=file)
		print(file=file)
		print("| {} |".format(' '*max_gpu_len, 1), end='', file=file)
		print(" estimated time (msecs) |  bound  |"*len(kernels), end='', file=file)
		print("     total estimated time (msecs) |", end='', file=file)
		print(file=file)
		print(dash_line, file=file)
		# data rows
		for g in gpus:
			print("| {} |".format(g.ljust(max_gpu_len)), end='', file=file)
			for k in kernels:
				print(format(self.data[g][k]['estimated_time'], '23.5f')+' | '+self.data[g][k]['type'].center(7)+' |', end='', file=file)
			print(format(sum((m['estimated_time'] for m in self.data[g].values())), '33.5f')+' |', end='', file=file)
			print(file=file)
		print(dash_line, file=file)

	def save_to_csv(self, file):
		# extract kernel and gpu lists
		gpus = self.data.keys()
		kernels = next(iter(self.data.values())).keys()
		# lengths for facilitating alignments
		kernel_lens = (len(k) for k in kernels)
		max_gpu_len = max((len(g) for g in gpus))
		# header row
		print("\"{}\", ".format('GPU'), end='', file=file)
		print(", ".join(('"'+k+'", ' for k in kernels)), file=file)
		# data rows
		for g in gpus:
			print("\"{}\", ".format(g), end='', file=file)
			for k in kernels:
				print(format(self.data[g][k]['estimated_time'], '.5f')+', "'+self.data[g][k]['type']+'", ', end='', file=file)
			print(file=file)

	def save_to_json(self, file):
		print(json.dumps(self.data), file=file)

class perfPredictor:
	def __init__(self, kernel_parameters, gpu_parameters, optimistic=False):
		self.kernel_params = kernel_parameters
		self.gpu_params = gpu_parameters
		self.optimistic = optimistic

	def execute(self, arguments=None, message=None, device=None):
		res = perfPrediction()
		for dev, dev_params in self.gpu_params.items():
			# convert operation throughput rates to instruction execution rates
			dev_params_inst = {key:spec/2 if key.startswith('GFLOPS') or key=='GIOPS - MAD' else spec for key,spec in dev_params.items()}
			dev_prediction = {}
			for kernel, kernel_params in self.kernel_params.data.items():
				kernel_operation_ratio = float("inf") if kernel_params['dram_bytes']==0 else kernel_params['compute_ops']/kernel_params['dram_bytes']
				enable_int_optimistic_mode = self.optimistic and (kernel_params['dominant_op'] == 'integer')
				# throughput factors
				comp_thoughput = {'fp_32': dev_params['GFLOPS - SP'], 'fp_64': dev_params['GFLOPS - DP'], 'fp_16': -1., 'integer': (dev_params['GIOPS - ADD'] if enable_int_optimistic_mode else dev_params['GIOPS - MAD'])}[kernel_params['dominant_op']]
				# compute ratios of instruction throughput of various instruction types compared to single precision FLOPs (typically the fastest instructions)
				instr_throughput_factors = {
					'fp_32': 1,
					'fp_64':   dev_params_inst['GFLOPS - SP'] / dev_params_inst['GFLOPS - DP'],
					'integer': dev_params_inst['GFLOPS - SP'] / (dev_params_inst['GIOPS - ADD'] if enable_int_optimistic_mode else dev_params_inst['GIOPS - MAD']),
					'ls':      dev_params_inst['GFLOPS - SP'] / dev_params_inst['GOPS - SHMEM'],
					'other':   dev_params_inst['GFLOPS - SP'] / dev_params_inst['GIOPS - ADD']
				}
				operation_throughput_factor = instr_throughput_factors[kernel_params['dominant_op']]
				# evaluate instruction and overall efficiencies
				inst_efficiency = operation_throughput_factor*kernel_params['mix_compute']/((1-kernel_params['mix_compute']-kernel_params['mix_ldst'])*instr_throughput_factors['other']+kernel_params['mix_compute']*operation_throughput_factor+kernel_params['mix_ldst']*instr_throughput_factors['ls'])
				overall_efficiency = (1.0 if enable_int_optimistic_mode else kernel_params['opmix_efficiency'])*inst_efficiency
				adjusted_peak_compute = comp_thoughput * overall_efficiency
				adjusted_operation_ratio = adjusted_peak_compute/dev_params_inst['GBSEC - DRAM']
				is_compute_bound = kernel_operation_ratio>adjusted_operation_ratio
				try:
					kernel_type_balance = (kernel_operation_ratio-adjusted_operation_ratio)/kernel_operation_ratio
				except ZeroDivisionError as e:
					kernel_type_balance = float('nan') if adjusted_operation_ratio>0 else float('-inf')
				ratios_slope = (adjusted_operation_ratio-kernel_operation_ratio)/(1+adjusted_operation_ratio*kernel_operation_ratio)
				ratios_angle = atan(ratios_slope)
				estimated_comp_throughput = adjusted_peak_compute if is_compute_bound else kernel_operation_ratio*dev_params_inst['GBSEC - DRAM']
				estimated_bandwidth = adjusted_peak_compute/kernel_operation_ratio if is_compute_bound else dev_params_inst['GBSEC - DRAM']
				try:
					estimated_exec_time = kernel_params['compute_ops']/(estimated_comp_throughput*10**9)*10**3
				except ZeroDivisionError as e:
					try:
						estimated_exec_time = kernel_params['dram_bytes']/(estimated_bandwidth*10**9)*10**3
					except ZeroDivisionError as e:
						estimated_exec_time = 0
						print('\tWARNING: Could not estimate time due to zero estimated throughput', file=sys.stderr)
				try:
					estimated_l2_bandwidth = (kernel_params['l2_bytes']/10**9)/(estimated_exec_time/10**3)
				except ZeroDivisionError as e:
					estimated_l2_bandwidth = 0
				dev_prediction[kernel] = {
					'estimated_time': estimated_exec_time,
					'type': 'Compute' if is_compute_bound else 'Memory',
					'krn_op_ratio': kernel_operation_ratio,
					'dev_op_ratio': adjusted_operation_ratio,
					'deviation_angle': ratios_angle,
					'throughput': estimated_comp_throughput,
					'bandwidth': estimated_bandwidth,
					'l2_bandwidth': estimated_l2_bandwidth
				}
			res.add(dev, dev_prediction)
		return res
