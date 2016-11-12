import numpy as np
from pybedtools import BedTool
import pybedtools

class BedFile:
	'''
	API for processing BedFiles
	'''
	def __init__(self, filename, reference_genome, flank_length):
		self.filename = filename
		self.peak_boundaries = {}
		self.raw = BedTool(self.filename)
		self.merged = self.raw.sort().merge()
		self.slopped = None 
		self.chromosomes = [x[0] for x in self.merged]
		self.startvals = None
		self.endvals = None
		self.reference_genome = reference_genome
		self.flank_length = flank_length

	def extractIntervals(self):
		midpointlist = []
		for peak in self.merged:
			midpoint = round((int(peak[1]) + int(peak[2]))/2)
			midpointlist.append((peak[0], midpoint, midpoint+1))

		midpoints = BedTool(midpointlist)

		chrom = pybedtools.chromsizes(self.reference_genome)
		self.slopped = midpoints.slop(b=self.flank_length, g=chrom)
		
		self.startvals = [int(x[1]) for x in self.slopped]
		self.endvals = [int(x[2]) for x in self.slopped]

		return self.chromosomes, self.startvals, self.endvals


