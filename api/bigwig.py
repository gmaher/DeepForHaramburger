import numpy as np
import pyBigWig as pbw

class BigWigReader:
    def __init__(self, fname):
        """ API wrapper for pyBigWig. Provides read access to a BigWig file. """
        self.bw = pbw.open(fname)

    def cleanUp(self):
        """ Cleans up the reader by closing the BigWig file. Future calls to
        get_* methods will throw exceptions. """
        self.bw.close()

    def getBatch(self, chrList, startList, endList):
        """ Return a batch of signal sequences from the BigWig file
	Args:
            chrList (list): List of chromosome strings (e.g. 'chr3')
            startList (list): List of start positions
            endList (list): List of end positions

        Returns:
            numpy array, shape = (N,L), L = sequence length

        Note:
            chrList, startList and endList all must have same length
	"""
        assert len(chrList) == len(startList) == len(endList)
        # TODO: okay if sequences are different length? if not, check w/ asserts
        seqList = [self.getSeqStartEnd(c, s, e)
                for c, s, e in zip(chrList, startList, endList)]
        return np.asarray(seqList)

    def getSeqString(self, chromosome, midpoint, length):
        """ Return a BigWig signal at the provided chromosome with the given
        length and midpoint. """
        # TODO: shouldn't length be odd? 2k+1 for k-length padding on each side
        assert length % 2 == 0
        assert midpoint >= length / 2
        start = midpoint - length / 2
        end = midpoint + length / 2
        return self.getSeqStartEnd(chromosome, start, end)

    def getSeqStartEnd(self, chromosome, start, end):
        """ Return a BigWig signal at the provided chromosome with the given
        start and end. """
        return np.array(self.bw.values(chromosome, start, end))
