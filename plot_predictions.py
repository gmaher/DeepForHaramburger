import sys
import os
sys.path.append(os.path.abspath('./api'))
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from api.data import DataReader
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import configparser

from keras.models import Model, load_model
from keras.layers import Input, Convolution1D, BatchNormalization, Dense, merge
from keras.layers import Reshape, Flatten, UpSampling2D
from keras.layers import AveragePooling1D, UpSampling1D, Activation
from keras.optimizers import Adam
from keras.regularizers import l2

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
matplotlib.rc('figure',**{'autolayout':True})

parser = argparse.ArgumentParser()
parser.add_argument('seq2seq_dir')
parser.add_argument('flankLength')
args = parser.parse_args()

dataDir = args.seq2seq_dir
flankLength = args.flankLength
def normalize(y, C):
    y = np.sqrt(y/C)
    y = y.reshape((y.shape[0],y.shape[1],1))
    return y
##########################
# Parse Data
##########################
bedfilename = (dataDir+
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.pval0.1.500000."
                "naive_overlap.narrowPeak.gz")
referenceGenome = 'hg19'
fastaFname = dataDir+"hg19.fa"
bigwigFname = (dataDir+
                "E116.GM12878_Lymphoblastoid_Cells.ENCODE.Duke_Crawford."
                "DNase-seq.merged.20bp.filt.50m.pf.fc.signal.bigwig")

reader = DataReader(bedfilename, referenceGenome, flankLength, fastaFname,
        bigwigFname)

C = 22.0

Xval_pos,yval_pos = reader.getChromosome(['chr6','chr7','chr8'])
yval_pos = normalize(yval_pos,C)
Xval_neg,yval_neg = reader.getChromosome(['chr6','chr7','chr8'],exclude=False,negative=True)
yval_neg = normalize(yval_neg,C)
Xval = np.vstack((Xval_pos,Xval_neg))
yval = np.vstack((yval_pos,yval_neg))
print("Xval shape = {}, yval shape = {}".format(Xval_pos.shape,yval_pos.shape))

net = load_model("./run1/model.h5")

yhat = net.predict(Xval)

plot_ids = np.random.choice(yval.shape[0],50)

for i in plot_ids:
    plt.figure()
    plt.plot(yval[i], label='truth')
    plt.plot(yhat[i], label='prediction')
    plt.ylim((0,1))
    plt.xlabel('sequence position')
    plt.ylabel('DHS value')
    plt.legend()
    plt.savefig('./plots/fcn_{}'.format(i))
    plt.close()

net = load_model("./models/cnn_lstmnegTrain_2048.h5")
yhat = net.predict(Xval)
for i in plot_ids:
    plt.figure()
    plt.plot(yval[i], label='truth')
    plt.plot(yhat[i], label='prediction')
    plt.ylim((0,1))
    plt.xlabel('sequence position')
    plt.ylabel('DHS value')
    plt.legend()
    plt.savefig('./plots/hybrid_{}'.format(i))
    plt.close()
