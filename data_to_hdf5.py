import h5py
from api.data import DataReader
import matplotlib.pyplot as plt
import argparse
import numpy as np

def normalize(y, C):
    y = np.sqrt(y/C)
    y = y.reshape((y.shape[0],y.shape[1],1))
    return y

parser = argparse.ArgumentParser()
parser.add_argument('seq2seq_dir')
parser.add_argument('flankLength')

args = parser.parse_args()

dataDir = args.seq2seq_dir
flankLength = args.flankLength

bedfilename = (dataDir+"GM12878_cells/peak/macs2/overlap/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.pval0.1.500000."
                "naive_overlap.narrowPeak.gz")
referenceGenome = 'hg19'
fastaFname = dataDir+"hg19.fa"
bigwigFname = (dataDir+"GM12878_cells/signal/macs2/rep1/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.fc.signal.bigwig")

reader = DataReader(bedfilename, referenceGenome, flankLength, fastaFname,
        bigwigFname)

Xtrain,ytrain = reader.getChromosome(['chr6','chr7','chr8'], exclude=True)
C = np.max(ytrain)
#C=1
ytrain = normalize(ytrain,C)
print("Xtrain shape = {}, ytrain shape = {}".format(Xtrain.shape,ytrain.shape))

Xval,yval = reader.getChromosome(['chr6'])
yval = normalize(yval,C)
print("Xval shape = {}, yval shape = {}".format(Xval.shape,yval.shape))

Xtest,ytest = reader.getChromosome(['chr7','chr8'])
ytest = normalize(ytest,C)
print("Xtest shape = {}, ytest shape = {}".format(Xtest.shape,ytest.shape))
