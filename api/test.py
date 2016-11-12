import fasta
from BedFile import BedFile
from pybedtools import BedTool, example_filename

#BedFile test
reference_genome = 'hg19'
bedfilename = '/data/seq2seq-data/GM12878_cells/peak/macs2/overlap/E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford.DNase-seq.merged.20bp.filt.50m.pf.pval0.1.500000.naive_overlap.narrowPeak.gz'
flank_length = 1000
bedobj = BedFile(bedfilename, reference_genome, flank_length)
chr_list, start_list, end_list = bedobj.extractIntervals()
print(chr_list[0], start_list[0], end_list[0])

#use pybedtools to read in batches from the fasta_file
fasta_fn = '/data/seq2seq-data/' + reference_genome + '.fa'
batch_size = 10
batch_bed = BedTool(bedobj.slopped[0:batch_size])
batched_fasta = open(batch_bed.sequence(fi = example_filename(fasta_fn)).seqfn).read()
print(batched_fasta)

#fasta_api SeqIO is too slow -- need to fix, or use above pybedtools reader
#api = fasta.fasta_api('/data/seq2seq-data/'+ reference_genome +'.fa')
#x = api.get_batch(chr_list[0:128],start_list[0:128],end_list[0:128])
#print(x.shape)
