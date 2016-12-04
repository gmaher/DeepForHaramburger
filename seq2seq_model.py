import sys
import os
sys.path.append(os.path.abspath('./api'))

from api.data import DataReader
import matplotlib.pyplot as plt
import argparse
import numpy as np

import seq2seq
from seq2seq.models import Seq2Seq

def normalize(y, C):
    y = np.sqrt(y/C)
    y = y.reshape((y.shape[0],y.shape[1],1))
    return y

parser = argparse.ArgumentParser()
parser.add_argument('seq2seq_dir')
args = parser.parse_args()

dataDir = args.seq2seq_dir

bedfilename = (dataDir+"GM12878_cells/peak/macs2/overlap/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.pval0.1.500000."
                "naive_overlap.narrowPeak.gz")
referenceGenome = 'hg19'
flankLength = 401
fastaFname = dataDir+"hg19.fa"
bigwigFname = (dataDir+"GM12878_cells/signal/macs2/rep1/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.fc.signal.bigwig")

print("Initializing data reader...")
reader = DataReader(bedfilename, referenceGenome, flankLength, fastaFname,
        bigwigFname)
print("done!")

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

batchSize = 64

nTrain = Xtrain.shape[0]
nTrainRound = (nTrain // batchSize) * batchSize
nVal = Xval.shape[0]
nValRound = (nVal // batchSize) * batchSize
nTest = Xtest.shape[0]
nTestRound = (nTest // batchSize) * batchSize

Xtrain = Xtrain[:nTrainRound,:,:]
ytrain = ytrain[:nTrainRound,:,:]
Xval = Xval[:nValRound,:,:]
yval = yval[:nValRound,:,:]
Xtest = Xtest[:nTestRound,:,:]
ytest = ytest[:nTestRound,:,:]

model = Seq2Seq(batch_input_shape=(batchSize, 803, 4), hidden_dim=100,
        output_length=803, output_dim=1, depth=2)
model.compile(loss='mse', optimizer='adam')
model.fit(Xtrain, ytrain, batch_size=batchSize, nb_epoch=3,
        validation_data=(Xval,yval))

yval_preds = model.predict(Xval, batchSize, verbose=1)

def saveFigs():
    for i in range(3):
        plt.figure()
        plt.plot(yval[i, :, 0], label="Actual")
        plt.plot(yval_preds[i, :, 0], label="Predicted")
        plt.legend()
        plt.savefig("seq2seq_output_%d_comparison.png" % i)

saveFigs()
