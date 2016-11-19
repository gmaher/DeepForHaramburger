import bedfile
import bigwig
import fasta
import random

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
        self.chrList, self.startList, self.endList = bedobj.extractIntervals()
        self.nSeqs = len(self.chrList)
        self.fasta = fasta.FastaApi(fastaFilename)
        self.bigwig = bigwig.BigWigReader(bigwigFilename)

    def getBatch(self, batchSize=10):
        """ Returns the FASTA sequence and BigWig signal for a batch of size
        batchSize randomly sampled from this reader's set of sequences. """
        idxs = [random.randint(0, self.nSeqs) for _ in range(batchSize)]
        chrBatch = [self.chrList[idx] for idx in idxs]
        startBatch = [self.startList[idx] for idx in idxs]
        endBatch = [self.endList[idx] for idx in idxs]
        X = self.fasta.getBatch(chrBatch, startBatch, endBatch)
        y = self.bigwig.getBatch(chrBatch, startBatch, endBatch)
        return (X, y)
