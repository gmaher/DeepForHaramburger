import sys
import os
sys.path.append(os.path.abspath('./api'))

from api.data import DataReader
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import configparser

from keras.models import Model
from keras.layers import Input, Convolution1D, BatchNormalization, Dense, merge
from keras.layers import Reshape, Flatten, UpSampling2D
from keras.layers import AveragePooling1D, UpSampling1D, Activation
from keras.optimizers import Adam
from keras.regularizers import l2

parser = argparse.ArgumentParser()
parser.add_argument('seq2seq_dir')
parser.add_argument('config_file')
args = parser.parse_args()

dataDir = args.seq2seq_dir

#################################
#parse config file
#################################
config_file = args.config_file
cp = configparser.ConfigParser()
cp.read(config_file)
params = cp['learn_params']

def normalize(y, C):
    y = np.sqrt(y/C)
    y = y.reshape((y.shape[0],y.shape[1],1))
    return y

def getFullBatch(chrs):
    xpos,ypos = reader.getBatch(batch_size,chrs)
    xneg,yneg = reader.getNegativeBatch(batch_size,chrs)
    X = np.vstack((xpos,xneg))
    Y = np.vstack((ypos,yneg))
    Y = Y.reshape((Y.shape[0],Y.shape[1],1))
    Y = np.sqrt(Y/C)
    return X,Y

Nfilters = int(params['Nfilters'])
Wfilter = int(params['Wfilter'])
num_conv= int(params['num_conv'])
num_layers = int(params['num_layers'])
l2_reg= float(params['l2_reg'])
lr = float(params['lr'])
opt = Adam(lr=lr)
batch_size=int(params['batch_size'])
nb_epoch = int(params['nb_epoch'])
Niter = int(2e3)
print_step = int(5e1)
flankLength = int(params['flankLength'])
model_str = params['model']
input_shape = (2*flankLength,4)
outputDir = params['outputDir']

##########################
# Parse Data
##########################
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
#Xtrain_neg,ytrain_neg = reader.getChromosome(['chr6','chr7','chr8'], exclude=True,negative=True)

C = np.max(ytrain)

ytrain = normalize(ytrain,C)
#ytrain_neg = normalize(ytrain_neg,C)
#Xtrain = np.vstack((Xtrain,Xtrain_neg))
#ytrain = np.vstack((ytrain,ytrain_neg))
print("Xtrain shape = {}, ytrain shape = {}".format(Xtrain.shape,ytrain.shape))

Xval,yval = reader.getChromosome(['chr6','chr7','chr8'])
yval = normalize(yval,C)
Xval_neg,yval_neg = reader.getChromosome(['chr6','chr7','chr8'],exclude=False,negative=True)
yval_neg = normalize(yval_neg,C)
Xval = np.vstack((Xval,Xval_neg))
yval = np.vstack((yval,yval_neg))
print("Xval shape = {}, yval shape = {}".format(Xval.shape,yval.shape))

##############################
#Neural Network Definitions
##############################
def FCN(input_shape=(800,4), Nfilters=32, Wfilter=3,num_conv=3,
output_channels=1, output_activation='sigmoid', l2_reg=0.0):
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

def MSFCN(input_shape=(800,4), Nfilters=32, Wfilter=3,num_layers=3,
num_conv=2,output_channels=1, output_activation='sigmoid', pool_length=2, l2_reg=0.0):
    '''
    Makes a multiscale FCN
    '''
    outs = []

    inp = Input(shape=input_shape)

    x = Convolution1D(Nfilters,Wfilter,input_dim=4,activation='relu',
    border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(inp)
    x = BatchNormalization(mode=2)(x)

    for i in range(num_layers):
        for j in range(num_conv):
                x = Convolution1D(Nfilters,Wfilter,input_dim=4,activation='relu',
                border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
                x = BatchNormalization(mode=2)(x)

        x = Convolution1D(Nfilters,Wfilter,input_dim=4,activation='relu',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)

        if i > 0:
            x_up = UpSampling1D(length=pool_length**i)(x)
            x_up = Convolution1D(Nfilters,Wfilter,input_dim=4,activation='relu',
            border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x_up)
            outs.append(x_up)
        else:
            outs.append(x)

        x = AveragePooling1D(pool_length=pool_length)(x)

    out = merge(outs, mode="sum")

    out = Convolution1D(output_channels,Wfilter,
    input_dim=4,activation=output_activation, border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(out)

    MSFCN = Model(inp,out)
    return MSFCN

if model_str == "fcn":
    net = FCN(input_shape,Nfilters,Wfilter,num_conv,l2_reg=l2_reg)
if model_str == "msfcn":
    net = MSFCN(input_shape,Nfilters,Wfilter,num_layers=num_layers,num_conv=num_conv, l2_reg=l2_reg)

net.compile(optimizer=opt,loss='mse')

#pretrain on positive set
net.fit(Xtrain,ytrain, batch_size=batch_size, nb_epoch=nb_epoch,
validation_data=(Xval,yval))

loss = []
val_loss = []
for i in range(Niter):
    #sample positive and negative batch
    X,Y = getFullBatch(['chr6','chr7','chr8'])
    l = net.train_on_batch(X,Y)
    loss.append(l)
    if i%print_step == 0:
        print("loss = {}".format(l))
        vl = net.evaluate(Xval,yval)
        val_loss.append(vl)
################################
# Make directory and save output
################################
if not os.path.exists('./{}'.format(outputDir)):
    os.makedirs('./{}'.format(outputDir))
    os.makedirs('./{}/plots'.format(outputDir))

net.save('./{}/model.h5'.format(outputDir))

yhat = net.predict(Xval)
#loss = net.evaluate(Xval,yval)

np.save("./{}/yhat".format(outputDir),yhat)
np.save("./{}/loss".format(outputDir),np.asarray(loss))
np.save("./{}/val_loss".format(outputDir),np.asarray(val_loss))

plot_ids = np.random.choice(yval.shape[0],50)
for i in plot_ids:
    plt.figure()
    plt.plot(yval[i], label='truth')
    plt.plot(yhat[i], label='prediction')
    plt.legend()
    plt.savefig('./{}/plots/{}'.format(outputDir,i))

plt.figure()
plt.plot(loss, label="training loss")
plt.plot(val_loss, label="validation loss")
plt.legend()
plt.savefig('./{}/plots/loss'.format(outputDir))

f = open('mse.txt','a')
f.write('{},{},{},{},{},{}\n'.format(model_str, flankLength,num_layers,Nfilters,Wfilter, loss))
f.close()
