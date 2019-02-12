#!/usr/bin/python3

"""
gpuParamLoader is a helper class for loading the GPU parameters from a CSV file
"""

import csv

class gpuParamLoader:
	def __init__(self, gpuspecsFile):
		self._file = gpuspecsFile
		self.parameters = None

	def load(self):
		csvReader = csv.reader(open(self._file))
		header = next(csvReader)
		res = {}
		for row in csvReader:
			res[row[0]] = {v.strip('" '):float(row[i]) for i,v in enumerate(header) if i>0}
		self.parameters = res
		return res
