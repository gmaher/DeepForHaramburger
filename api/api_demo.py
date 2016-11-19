from data import DataReader

bedfilename = ("/data/seq2seq-data/GM12878_cells/peak/macs2/overlap/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.pval0.1.500000."
                "naive_overlap.narrowPeak.gz")
referenceGenome = 'hg19'
flankLength = 400
fastaFname = "/data/seq2seq-data/hg19.fa"
bigwigFname = ("/data/seq2seq-data/GM12878_cells/signal/macs2/rep1/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.fc.signal.bigwig")

reader = DataReader(bedfilename, referenceGenome, flankLength, fastaFname,
        bigwigFname)

"""
Iterate over batches of size 20:

for _ in range(numBatches)
    batchX, batchY = reader.getBatch(20)
    doSomething(batchX, batchY)

"""
