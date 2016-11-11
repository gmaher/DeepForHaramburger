import fasta

api = fasta.fasta_api('/home/marsdenlab/datasets/seq2seq-data/hg38.fa')
chr_list = ['chr3','chr4','chr5','chr6','chr7']
start_list = [10000, 50000, 100000, 4, 250000]
end_list = [11000, 51000, 101000,1004, 251000]
x = api.get_batch(chr_list,start_list,end_list)
print x.shape
