import bedfile
import bigwig
import fasta
import random
import numpy as np
class DataReader:
    def __init__(self, bedfilename, referenceGenome, flankLength,
            fastaFilename, bigwigFilename):
        """ Batch reader over a FASTA file and BigWig file as indexed by a
        bedfile.

        Args:
            bedfilename         BedFile filename
            referenceGenome     #TODO: what is this?
            flankLength         context length; returned sequences are of length
                                    2 * flankLength + 1
            fastaFilename       FASTA filename
            bigwigFilename      BigWig filename
        """
        bedobj = bedfile.BedFile(bedfilename, referenceGenome, flankLength)
        self.flankLength = flankLength
        self.chrList, self.startList, self.endList = bedobj.extractIntervals()
        self.nSeqs = len(self.chrList)
        self.fasta = fasta.FastaApi(fastaFilename)
        self.bigwig = bigwig.BigWigReader(bigwigFilename)

        #set up for negative set sampling
        self.chroms = list(set(self.chrList))
        self.chr_sizes = {}
        for x in self.chroms:
            self.chr_sizes[x] = len(self.fasta.faDict[x])

    def getBatch(self, batchSize=10):
        """ Returns the FASTA sequence and BigWig signal for a batch of size
        batchSize randomly sampled from this reader's set of sequences. """
        idxs = [random.randint(0, self.nSeqs) for _ in range(batchSize)]
        chrBatch = [self.chrList[idx] for idx in idxs]
        startBatch = [self.startList[idx] for idx in idxs]
        endBatch = [self.endList[idx] for idx in idxs]
        X = self.fasta.getBatch(chrBatch, startBatch, endBatch)
        y = self.bigwig.getBatch(chrBatch, startBatch, endBatch)
        y[np.isnan(y)] = 0.0
        return (X, y)

    def getNegativeBatch(self, batchSize=10):
        """returns FASTA sequence and BigWig signal for a batch of indices
        sampled from arbitrary parts of the genome"""
        idxs = [random.randint(0,len(self.chroms)-1) for i in range(batchSize)]
        chrBatch = [self.chroms[i] for i in idxs]
        startBatch = [random.randint(0,self.chr_sizes[x]-self.flankLength) for x in chrBatch]
        endBatch = [startBatch[i]+self.flankLength for i in range(0,len(startBatch))]

        X = self.fasta.getBatch(chrBatch, startBatch, endBatch)
        y = self.bigwig.getBatch(chrBatch, startBatch, endBatch)
        y[np.isnan(y)] = 0.0
        return (X, y)

    def getChromosome(self,chrs, exclude=False):
        """ Returns all the sequences coming from a list of chromosomes

        Args:
            chrs    list of strings of chromosome names
            exclude boolean, whether to select for the provided chromosome, or
            all chromosomes except the provided one
        """
        if exclude:
            idxs = [i for i in range(self.nSeqs) if\
                not any(self.chrList[i]==chro for chro in chrs)]
        else:
            idxs = [i for i in range(self.nSeqs) if\
                any(self.chrList[i]==chro for chro in chrs)]
        chrBatch = [self.chrList[idx] for idx in idxs]
        startBatch = [self.startList[idx] for idx in idxs]
        endBatch = [self.endList[idx] for idx in idxs]
        X = self.fasta.getBatch(chrBatch, startBatch, endBatch)
        y = self.bigwig.getBatch(chrBatch, startBatch, endBatch)
        y[np.isnan(y)] = 0.0
        return (X, y)
