import bigwig
import fasta
from bedfile import BedFile
from pybedtools import BedTool, example_filename

#BedFile test
referenceGenome = 'hg19'
bedfilename = '/data/seq2seq-data/GM12878_cells/peak/macs2/overlap/E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford.DNase-seq.merged.20bp.filt.50m.pf.pval0.1.500000.naive_overlap.narrowPeak.gz'
flankLength = 1000
bedobj = BedFile(bedfilename, referenceGenome, flankLength)
chrList, startList, endList = bedobj.extractIntervals()
print(chrList[0], startList[0], endList[0])

#use pybedtools to read in batches from the fasta_file
# fastaFn = '/data/seq2seq-data/' + referenceGenome + '.fa'
# batchSize = 10
# batchBed = BedTool(bedobj.slopped[0:batchSize])
# batchedFasta = open(batchBed.sequence(fi = exampleFilename(fastaFn)).seqfn).read()
# print(batchedFasta)

#fasta_api SeqIO is too slow -- need to fix, or use above pybedtools reader
print("loading api")
api = fasta.FastaApi('/data/seq2seq-data/'+ referenceGenome +'.fa')
print("getting sequences")
x = api.getBatch(chrList[0:128],startList[0:128],endList[0:128])
print(x.shape)

# BigWig test
print("building BW reader")
sampleBwFile = ("/data/seq2seq-data/GM12878_cells/signal/macs2/rep1/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.fc.signal.bigwig")
bwReader = bigwig.BigWigReader(sampleBwFile)
print("Getting BW sequences")
bwBatch = bwReader.getBatch(chrList[:128], startList[:128], endList[:128])
print(bwBatch.shape)
