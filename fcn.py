import sys
import os
sys.path.append(os.path.abspath('./api'))

from api.data import DataReader
import matplotlib.pyplot as plt
import argparse
import numpy as np

from keras.models import Model
from keras.layers import Input, Convolution2D, Convolution1D, BatchNormalization, Dense, merge
from keras.layers import Reshape, Flatten, UpSampling2D
from keras.layers import MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.regularizers import l2

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

#Create neural network
def FCN(input_shape=(800,4), Nfilters=32, Wfilter=3,num_conv=3,
output_channels=1, output_width=1, output_height=4,
output_activation='sigmoid', l2_reg=0.0):
    '''
    Makes an FCN neural network
    '''
    x = Input(shape=input_shape)

    d = Convolution1D(Nfilters,Wfilter,input_dim=4,activation='relu',
    border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    d = BatchNormalization(mode=2)(d)

    for i in range(0,num_conv):
        d = Convolution1D(Nfilters,Wfilter,input_dim=4,activation='relu',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)
        d = BatchNormalization(mode=2)(d)

    d = Convolution1D(output_channels,Wfilter,
    input_dim=4,activation=output_activation, border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    FCN = Model(x,d)
    return FCN

input_shape = (2*flankLength+1,4)
Nfilters = 32
Wfilter = 3
num_conv=5
l2_reg=0
lr = 1e-3
opt = Adam(lr=lr)
batch_size=32
nb_epoch = 5

net = FCN(input_shape,Nfilters,Wfilter,num_conv,l2_reg=l2_reg)
net.compile(optimizer=opt,loss='mse')

net.fit(Xtrain,ytrain, batch_size=batch_size, nb_epoch=nb_epoch,
validation_data=(Xval,yval))
