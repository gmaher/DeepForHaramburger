import bigwig
import fasta
import random
from BedFile import BedFile
from pybedtools import BedTool, example_filename

class DataReader:
    def __init__(self, bedfilename, reference_genome, flank_length,
            fasta_filename, bigwig_filename):
        """ Batch reader over a FASTA file and BigWig file as indexed by a
        bedfile.

        Args:
            bedfilename         BedFile filename
            reference_genome    #todo
            flank_length        context length; returned sequences are of length
                                2 * flank_length + 1
            fasta_filename      FASTA filename
            bigwig_filename     BigWig filename
        """
        bedobj = BedFile(bedfilename, reference_genome, flank_length)
        self.chr_list, self.start_list, self.end_list = bedobj.extractIntervals()
        self.n_seqs = len(self.chr_list)
        self.fasta = fasta.fasta_api(fasta_filename)
        self.bigwig = bigwig.bigwig_reader(bigwig_filename)

    def get_batch(self, batch_size=10):
        """ Returns the FASTA sequence and BigWig signal for a batch of size
        batch_size randomly sampled from this reader's set of sequences. """
        idxs = [random.randint(0, self.n_seqs) for _ in range(batch_size)]
        chr_batch = [self.chr_list[idx] for idx in idxs]
        start_batch = [self.start_list[idx] for idx in idxs]
        end_batch = [self.end_list[idx] for idx in idxs]
        X = self.fasta.get_batch(chr_batch, start_batch, end_batch)
        y = self.bigwig.get_batch(chr_batch, start_batch, end_batch)
        return (X, y)
