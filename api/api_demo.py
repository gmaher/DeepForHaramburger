from api import DataReader

bedfilename = ("/data/seq2seq-data/GM12878_cells/peak/macs2/overlap/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.pval0.1.500000."
                "naive_overlap.narrowPeak.gz")
reference_genome = 'hg19'
flank_length = 400
fasta_fname = '/data/seq2seq-data/'+ reference_genome +'.fa'
bigwig_fname = ("/data/seq2seq-data/GM12878_cells/signal/macs2/rep1/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.fc.signal.bigwig")

reader = DataReader(bedfilename, reference_genome, flank_length, fasta_fname,
        bigwig_fname)

"""
Iterate over batches of size 20:

for _ in range(num_batches)
    batch_X, batch_y = reader.get_batch(20)
    do_something(batch_X, batch_y)

"""
