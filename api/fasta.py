from Bio import SeqIO
import numpy as np

class FastaApi:
    """Wrapper to act as an api to a fasta file
    """
    def __init__(self,filename):
        """Store the fasta filename and load it as a Biopython index object
        """
        self.filename = filename

        self.faDict = SeqIO.to_dict(SeqIO.parse(filename,'fasta'))

        bpMap = {}
        bpMap['N'] = [0,0,0,0]
        bpMap['A'] = [1,0,0,0]
        bpMap['T'] = [0,1,0,0]
        bpMap['C'] = [0,0,1,0]
        bpMap['G'] = [0,0,0,1]
        bpMap['a'] = bpMap['A']
        bpMap['t'] = bpMap['T']
        bpMap['c'] = bpMap['C']
        bpMap['g'] = bpMap['G']
        bpMap['n'] = bpMap['N']
        self.bpMap = bpMap

    def getBatch(self, chrList, startList, endList):
        """Return a batch of one-hot encoded dna sequences

        Args:
            chrList (list): List of chromosome strings (e.g. 'chr3')
            startList (list): List of start positions
            endList (list): List of end positions

        Returns:
            numpy array, shape = (N,L,4), L = sequence length

        Note:
            chrList, startList and endList all must have same length
        """
        seqList = [self.getSeqStartEnd(c,s,e)
                for c,s,e in zip(chrList,startList,endList)]
        return np.asarray(seqList)

    def getSeqString(self,chromosome, midpoint,length):
        """Returns the one-hot encoded DNA sequence for a particular chromosome
        and midpoint and sequence length

        note: length assumed to be even
        midpoint must be >= length/2
        """
        start = midpoint - length/2
        end = midpoint + length/2
        return self.getSeqStartEnd(chromosome,start,end)

    def getSeqStartEnd(self, chromosome, start, end):
        """Returns the one-hot encoded DNA sequence for a particular chromosome
        and start end location
        """
        seq = self.faDict[chromosome][start:end]

        oneHot = [self.bpMap[bp] for bp in seq]
        return oneHot
