import numpy as np
import pyBigWig as pbw

class bigwig_reader:
    def __init__(self, fname):
        """ API wrapper for pyBigWig. Provides read access to a BigWig file. """
        self.bw = pbw.open(fname)

    def clean_up(self):
        """ Cleans up the reader by closing the BigWig file. Future calls to
        get_* methods will throw exceptions. """
        self.bw.close()

    def get_batch(self, chr_list, start_list, end_list):
        """ Return a batch of signal sequences from the BigWig file
	Args:
            chr_list (list): List of chromosome strings (e.g. 'chr3')
            start_list (list): List of start positions
            end_list (list): List of end positions

        Returns:
            numpy array, shape = (N,L), L = sequence length

        Note:
            chr_list, start_list and end_list all must have same length
	"""
        assert len(chr_list) == len(start_list) == len(end_list)
        # TODO: okay if sequences are different length? if not, check w/ asserts
        seq_list = [self.get_seq_start_end(c, s, e)
                for c, s, e in zip(chr_list, start_list, end_list)]
        return np.asarray(seq_list)

    def get_seq_string(self, chromosome, midpoint, length):
        """ Return a BigWig signal at the provided chromosome with the given
        length and midpoint. """
        # TODO: shouldn't length be odd? 2k+1 for k-length padding on each side
        assert length % 2 == 0
        assert midpoint >= length / 2
        start = midpoint - length / 2
        end = midpoint + length / 2
        return self.get_seq_start_end(chromosome, start, end)

    def get_seq_start_end(self, chromosome, start, end):
        """ Return a BigWig signal at the provided chromosome with the given
        start and end. """
        return np.array(self.bw.values(chromosome, start, end))
