#!/usr/bin/python3

"""
KernelParameters implements a container for kernel parameters
along with streaming facilities i.e. loading and saving to local file.
"""

import json
from types import MappingProxyType

class KernelParameters:
	def __init__(self, header={}):
		self._header = header
		self._data = {}

	@property
	def header(self):
		return MappingProxyType(self._header)

	@property
	def data(self):
		return MappingProxyType(self._data)

	def __setitem__(self, kernel, values):
		self._data[kernel] = values

	def load(self, filename):
		with open(filename, 'rt') as f:
			data = json.loads(f.read())
		self._data = data['data']
		del data['data']
		self._header = data

	def save(self, filename):
		with open(filename, 'wt') as f:
			print(json.dumps({**self._header, 'data':self._data}), file=f)

	def clear(self):
		self._data = {}
