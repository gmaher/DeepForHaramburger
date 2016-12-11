import sys
import os
sys.path.append(os.path.abspath('./api'))

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
parser.add_argument('config_file')
args = parser.parse_args()

dataDir = args.seq2seq_dir

def normalize(y, C):
    y = np.sqrt(y/C)
    y = y.reshape((y.shape[0],y.shape[1],1))
    return y
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


Nfilters = int(params['Nfilters'])
Wfilter = int(params['Wfilter'])
num_conv= int(params['num_conv'])
num_layers = int(params['num_layers'])
l2_reg= float(params['l2_reg'])
lr = float(params['lr'])
opt = Adam(lr=lr)
batch_size=int(params['batch_size'])
nb_epoch = int(params['nb_epoch'])
Niter = int(1e3)
print_step = int(1e1)
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

Xtrain_pos,ytrain_pos = reader.getChromosome(['chr6','chr7','chr8'], exclude=True)
Xtrain_neg,ytrain_neg = reader.getChromosome(['chr6','chr7','chr8'], exclude=True,negative=True)

Xtrain = np.vstack((Xtrain_pos, Xtrain_neg))
ytrain = np.vstack((ytrain_pos, ytrain_neg))
C = np.max(ytrain_pos)

ytrain = normalize(ytrain,C)
ytrain_pos = normalize(ytrain_pos,C)
ytrain_neg = normalize(ytrain_neg,C)
#ytrain_neg = normalize(ytrain_neg,C)
#Xtrain = np.vstack((Xtrain,Xtrain_neg))
#ytrain = np.vstack((ytrain,ytrain_neg))
print("Xtrain shape = {}, ytrain shape = {}".format(Xtrain.shape,ytrain.shape))

Xval_pos,yval_pos = reader.getChromosome(['chr6','chr7','chr8'])
yval_pos = normalize(yval_pos,C)
Xval_neg,yval_neg = reader.getChromosome(['chr6','chr7','chr8'],exclude=False,negative=True)
yval_neg = normalize(yval_neg,C)
Xval = np.vstack((Xval_pos,Xval_neg))
yval = np.vstack((yval_pos,yval_neg))
print("Xval shape = {}, yval shape = {}".format(Xval_pos.shape,yval_pos.shape))

net = load_model("./{}/model.h5".format(outputDir))

train_loss_pos = net.evaluate(Xtrain_pos,ytrain_pos)
train_loss_neg = net.evaluate(Xtrain_neg,ytrain_neg)
train_loss = net.evaluate(Xtrain,ytrain)
val_loss_pos = net.evaluate(Xval_pos,yval_pos)
val_loss_neg = net.evaluate(Xval_neg,yval_neg)
val_loss = net.evaluate(Xval,yval)

f = open('mse.txt','a')
f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(model_str, flankLength,num_layers,Nfilters,Wfilter,\
 train_loss_pos, train_loss_neg, train_loss, val_loss_pos, val_loss_neg, val_loss))
f.close()

plot_ids = np.random.choice(yval.shape[0],50)

for i in plot_ids:
    plt.figure()
    plt.plot(yval[i], label='truth')
    plt.plot(yhat[i], label='prediction')
    plt.ylim((0,1))
    plt.xlabel('sequence position')
    plt.ylabel('DHS value')
    plt.legend()
    plt.savefig('./{}/plots/{}'.format(outputDir,i))
