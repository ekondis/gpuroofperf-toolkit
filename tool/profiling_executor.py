#!/usr/bin/python3

"""
profExecutor ensapsulates the complexity of leveraging the profiler tool
execution
"""

import os
import subprocess
import sys
import time
from threading import Thread

class profException(Exception):
    pass

class profExecutor:
	_NAME = "nvprof"

	def __init__(self, nvprofpath=None):
		self.nvprof = self._seeknvprof() if nvprofpath is None else nvprofpath
		if self.nvprof is None:
			raise profException('Profiler (%s) not found!' % (profExecutor._NAME))

	def _isexecutable(self, path):
		try:
			profExecutor._execute([path])
		except FileNotFoundError as e:
			return False
		return True

	def _seeknvprof(self):
		profiler = profExecutor._NAME
		if( self._isexecutable(profiler) ):
			return profiler
		if 'CUDA_PATH' in os.environ:
			cuda_install_path = os.environ['CUDA_PATH']
			if( self._isexecutable(cuda_install_path+"/bin/"+profiler) ):
				return cuda_install_path+"/bin/"+profiler
		DEFAULT_CUDA_PATH_LIN = '/usr/local/cuda/'
		if( self._isexecutable(DEFAULT_CUDA_PATH_LIN+profiler) ):
			return DEFAULT_CUDA_PATH_LIN+profiler
		DEFAULT_CUDA_PATH_WIN = '%CUDA_PATH%/bin/'
		if( self._isexecutable(DEFAULT_CUDA_PATH_WIN+profiler) ):
			return DEFAULT_CUDA_PATH_WIN+profiler
		return None

	def _progresswait(self, proc):
		CHARS = '/|-\|'
		counter = 0
		while proc.poll() is None:
			print("%c\b" % (CHARS[counter]), end='', flush=True)
			time.sleep(0.1)
			counter = (counter + 1) % len(CHARS)

	def execute(self, arguments, message=None, device=None):
		envvars = {}
		if device is not None:
			envvars['CUDA_VISIBLE_DEVICES'] = str(device)
		proc = profExecutor._execute( [self.nvprof]+arguments, envvars )
		(stdout, stderr) = (proc.stdout, proc.stderr)
		if message is not None:
			print("%s... " % (message), end='', flush=True)
		wait_thread = Thread(target=profExecutor._progresswait, args=(self, proc))
		wait_thread.start()
		(output, errors) = proc.communicate()
		wait_thread.join()
		lines_out = output.splitlines()
		lines_err = errors.splitlines()
		if message is not None:
			if proc.returncode==0:
				print("Done")
			else:
				print("Error code: %d" % (proc.returncode))
				raise profException("Profiling returned non zero error code. Profiler error output follows:\n%s" % (errors))
		return (lines_out, lines_err)

	def query_driver(self):
		"""Query the running NVidia GPU driver via nvidia-smi tool."""
		try:
			proc = profExecutor._execute( ['nvidia-smi', '-q'] )
			(stdout, stderr) = (proc.stdout, proc.stderr)
			print("Querying GPU driver version (via nvidia-smi)... ", end='', flush=True)
			wait_thread = Thread(target=profExecutor._progresswait, args=(self, proc))
			wait_thread.start()
			(output, _) = proc.communicate()
			wait_thread.join()
			lines_out = output.splitlines()
			#lines_err = errors.splitlines()
			if proc.returncode==0:
				print("Done")
			else:
				print("Error code: %d" % (proc.returncode))
				raise profException("Profiling returned non zero error code. Profiler error output follows:\n%s" % (errors))
			ver_line = filter(lambda x: x.startswith('Driver Version'), lines_out)
			return next(ver_line).split()[-1]
		except Exception as e:
			print("Warning: nvidia-smi {}".format(str(e)))
			return '-';

	@staticmethod
	def _execute(arguments, envvars=None):
		#nvprof --devices 0 --query-metrics
		#print("DEBUG: executing:'%s'" % (' '.join(arguments)))
		myenv = os.environ.copy()
		if envvars is not None:
			myenv.update(envvars)
		#if device is not None:
		#	myenv["CUDA_VISIBLE_DEVICES"] = str(device)
		proc = subprocess.Popen(arguments, env=myenv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
		return proc
		(proc_out, proc_err) = (proc.stdout, proc.stderr)
		print('proc_out:',proc_out)
		errors = proc_err.read()
		print(errors)
		if len(errors)>0:
			print( 'Error: '+errors)
		lines_out = proc_out.read()
		if len(lines_out)>0:
			print( 'stdout: '+lines_out)
