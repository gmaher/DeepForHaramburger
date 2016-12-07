import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../api"))

from api.data import DataReader

from sequential_danq_model import getSequentialDanQ

def normalize(y, C):
    y = np.sqrt(y/C)
    y = y.reshape((y.shape[0],y.shape[1],1))
    return y

parser = argparse.ArgumentParser()
parser.add_argument("--seq2seqDir", "-s", type=str, default="/data/seq2seq-data/")
parser.add_argument("--lstmSteps", "-l", type=int, default=51)
parser.add_argument("--poolWindow", "-p", type=int, default=13)
parser.add_argument("--batchSize", "-b", type=int, default=64)
parser.add_argument("--epochs", "-e", type=int, default=3)
parser.add_argument("--normalize", "-n", type=bool, default=True)
args = parser.parse_args()

dataDir = args.seq2seqDir

bedfilename = (dataDir+"GM12878_cells/peak/macs2/overlap/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.pval0.1.500000."
                "naive_overlap.narrowPeak.gz")
referenceGenome = "hg19"
fastaFname = dataDir+"hg19.fa"
bigwigFname = (dataDir+"GM12878_cells/signal/macs2/rep1/"
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.fc.signal.bigwig")

seqLength = args.lstmSteps * args.poolWindow
assert seqLength % 2 == 1
flankLength = (seqLength - 1) / 2

print("Initializing data reader...")
reader = DataReader(bedfilename, referenceGenome, flankLength, fastaFname,
        bigwigFname)
print("done!")

Xtrain, ytrain = reader.getChromosome(["chr6","chr7","chr8"], exclude=True)
Xval, yval = reader.getChromosome(["chr6"])
Xtest, ytest = reader.getChromosome(["chr7","chr8"])

if args.normalize:
    C = np.max(ytrain)
    ytrain = normalize(ytrain, C)
    yval = normalize(yval, C)
    ytest = normalize(ytest, C)

batchSize = args.batchSize
epochs = args.epochs
seqLength = 2 * flankLength + 1

# Truncate array sizes to a multiple of batchSize
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
print("Xtrain shape = {}, ytrain shape = {}".format(Xtrain.shape,ytrain.shape))
print("Xval shape = {}, yval shape = {}".format(Xval.shape,yval.shape))
print("Xtest shape = {}, ytest shape = {}".format(Xtest.shape,ytest.shape))

sess = tf.InteractiveSession()

modelX, keepProbPool, keepProbLstm, modely, modely_ = getSequentialDanQ(
        batchSize=batchSize,
        poolWindow=args.poolWindow,
        lstmTimeSteps=args.lstmSteps
        )

print(modelX, modely, modely_)

def dataSampler(X, y, batchSize=batchSize):
    assert X.shape[0] == y.shape[0]
    while True:
        sampleIdxs = np.random.choice(np.arange(X.shape[0]), size=batchSize)
        yield (X[sampleIdxs, :, :], y[sampleIdxs, :, :])

mse = tf.reduce_mean(tf.pow(tf.sub(modely, modely_), 2.0))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(mse)
sess.run(tf.global_variables_initializer())

trainIt = dataSampler(Xtrain, ytrain)
valIt = dataSampler(Xval, yval)
for i in range(10000):
    XtrainBatch, ytrainBatch = next(trainIt)
    if i % 100 == 0:
        XvalBatch, yvalBatch = next(valIt)
        batchLoss, valOutput = sess.run([mse, modely], feed_dict={
            modelX: XvalBatch,
            modely_: yvalBatch,
            keepProbPool: 1.0,
            keepProbLstm: 1.0
            })
        print("Step %d: MSE = %f" % (i, batchLoss))
        with open("val_%d_steps.npz" % (i), "wb") as f:
            np.savez(f,
                    X=XvalBatch,
                    y=yvalBatch,
                    yPred=valOutput)
    train_step.run(feed_dict={
        modelX: XtrainBatch,
        modely_: ytrainBatch,
        keepProbPool: 0.8,
        keepProbLstm: 0.5
        })
