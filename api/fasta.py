from Bio import SeqIO
import numpy as np

class fasta_api:
    """Wrapper to act as an api to a fasta file
    """
    def __init__(self,filename):
        """Store the fasta filename and load it as a Biopython index object
        """
        self.filename = filename

        self.fa_dict = SeqIO.to_dict(SeqIO.parse(filename,'fasta'))

        bp_map = {}
        bp_map['N'] = [0,0,0,0]
        bp_map['A'] = [1,0,0,0]
        bp_map['T'] = [0,1,0,0]
        bp_map['C'] = [0,0,1,0]
        bp_map['G'] = [0,0,0,1]
        bp_map['a'] = bp_map['A']
        bp_map['t'] = bp_map['T']
        bp_map['c'] = bp_map['C']
        bp_map['g'] = bp_map['G']
        bp_map['n'] = bp_map['N']
        self.bp_map = bp_map

    def get_batch(self, chr_list, start_list, end_list):
        """Return a batch of one-hot encoded dna sequences

        Args:
            chr_list (list): List of chromosome strings (e.g. 'chr3')
            start_list (list): List of start positions
            end_list (list): List of end positions

        Returns:
            numpy array, shape = (N,L,4), L = sequence length

        Note:
            chr_list, start_list and end_list all must have same length
        """
        seq_list = [self.get_seq_start_end(c,s,e) for c,s,e in zip(chr_list,start_list,end_list)]
        return np.asarray(seq_list)

    def get_seq_string(self,chromosome, midpoint,length):
        """Returns the one-hot encoded DNA sequence for a particular chromosome
        and midpoint and sequence length

        note: length assumed to be even
        midpoint must be >= length/2
        """
        start = midpoint - length/2
        end = midpoint + length/2
        return self.get_seq_start_end(chromosome,start,end)

    def get_seq_start_end(self, chromosome, start, end):
        """Returns the one-hot encoded DNA sequence for a particular chromosome
        and start end location
        """
        seq = self.fa_dict[chromosome][start:end]

        one_hot = [self.bp_map[bp] for bp in seq]
        return one_hot
