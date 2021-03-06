import sys
import os
sys.path.append(os.path.abspath('./api'))
print(sys.path)

from api.data import DataReader
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('seq2seq_dir')
args = parser.parse_args()

dataDir = args.seq2seq_dir

bedfilename = (dataDir+"GM12878_cells/peak/macs2/overlap/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.pval0.1.500000."
                "naive_overlap.narrowPeak.gz")
referenceGenome = 'hg19'
flankLength = 400
fastaFname = dataDir+"hg19.fa"
bigwigFname = (dataDir+"GM12878_cells/signal/macs2/rep1/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.fc.signal.bigwig")

reader = DataReader(bedfilename, referenceGenome, flankLength, fastaFname,
        bigwigFname)

X,y = reader.getBatch(10000)

print(X.shape)
print(y.shape)

plt.figure()
plt.hist(np.ravel(y),bins=50)
plt.show()

plt.figure()
plt.plot(y[:10].T)
plt.show()

#square root transform
y = y/np.max(y)
plt.figure()
plt.hist(np.ravel(np.sqrt(y)),bins=50)
plt.show()

plt.figure()
plt.plot(np.sqrt(y[:10].T))
plt.show()

#asinh + square root transform
y = y/np.max(y)
plt.figure()
plt.hist(np.ravel(np.arcsinh(np.sqrt(y))),bins=50)
plt.show()

plt.figure()
plt.plot(np.arcsinh(np.sqrt(y[:10].T)))
plt.show()
