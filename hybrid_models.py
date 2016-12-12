import sys
import os
sys.path.append(os.path.abspath('./api'))
#os.environ["CUDA_VISIBLE_DEVICES"]=""
from api.data import DataReader
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Embedding, \
                         Convolution1D, BatchNormalization, Lambda, Permute, merge
from keras.layers.recurrent import LSTM, GRU
from keras.layers.local import LocallyConnected1D
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers import Reshape, Flatten, UpSampling2D
from keras.layers import MaxPooling1D, AveragePooling1D, UpSampling1D, Activation
from keras.optimizers import Adam, RMSprop
from keras.layers.core import Reshape, Dropout, Highway
from keras.callbacks import EarlyStopping, ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('flankLength')
parser.add_argument('model')
parser.add_argument('seq2seq_dir')

args = parser.parse_args()

dataDir = args.seq2seq_dir
flankLength = int(args.flankLength)
model_str = args.model

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
Xtrain_neg,ytrain_neg = reader.getChromosome(['chr6','chr7','chr8'], exclude=True,negative=True)
C = np.max(ytrain)
#C=1
ytrain = normalize(ytrain,C)
print("Xtrain shape = {}, ytrain shape = {}".format(Xtrain.shape,ytrain.shape))


Xval,yval = reader.getChromosome(['chr6','chr7','chr8'])
yval = normalize(yval,C)
Xval_neg,yval_neg = reader.getChromosome(['chr6','chr7','chr8'],exclude=False,negative=True)
yval_neg = normalize(yval_neg,C)
Xval = np.vstack((Xval,Xval_neg))
yval = np.vstack((yval,yval_neg))
print("Xval shape = {}, yval shape = {}".format(Xval.shape,yval.shape))

#Xtest,ytest = reader.getChromosome(['chr7','chr8'])
#ytest = normalize(ytest,C)
#print("Xtest shape = {}, ytest shape = {}".format(Xtest.shape,ytest.shape))

#Create neural network
def CNNAttn(input_shape):
	inputs = Input(input_shape)
	conv = Convolution1D(nb_filter=128, filter_length=24, activation='relu', border_mode='valid')(inputs)
	pool = AveragePooling1D(pool_length=12, stride=12)(conv)
	drop1 = Dropout(0.2)(pool)
	forward_lstm = LSTM(output_dim=128,consume_less='gpu',return_sequences=True)(drop1)
	backward_lstm = LSTM(output_dim=128,consume_less='gpu', return_sequences=True, go_backwards=True)(drop1)
	merged = merge([forward_lstm, backward_lstm], mode='concat', concat_axis=-1)
	drop2 = Dropout(0.5)(merged)
	attention_hidden = TimeDistributed(Dense(256, activation='relu'))(drop2)
	attention_out = TimeDistributed(Dense(1, activation='linear'))(attention_hidden)
	attention_softmax = Activation('softmax')(attention_out)

	attention_dot = merge([attention_softmax, drop2], mode='dot', dot_axes=(1,1))

	lambda_1 = Lambda(lambda x: x[:,0,:])(attention_dot)

	#dense_1 = Dense(100, activation='sigmoid')(lambda_1)
	dense = Dense(input_shape[0], activation = 'sigmoid')(lambda_1)
	output = Reshape((input_shape[0],1))(dense)
	model = Model(input=inputs, output=output)
	return model

def cnn_lstm(input_shape):
	inputs = Input(input_shape)
	conv = Convolution1D(nb_filter=128, filter_length=24, activation='relu', border_mode='valid')(inputs)
	pool = AveragePooling1D(pool_length=12, stride=12)(conv)
	drop1 = Dropout(0.2)(pool)
	forward_lstm = LSTM(output_dim=128,consume_less='gpu',return_sequences=True)(drop1)
	backward_lstm = LSTM(output_dim=128,consume_less='gpu', return_sequences=True, go_backwards=True)(drop1)
	merged = merge([forward_lstm, backward_lstm], mode='concat', concat_axis=-1)
	drop2 = Dropout(0.5)(merged)
	flat = Flatten()(drop2)
	#dense = Dense(256, activation='relu')(flat)
	#dense = Dropout(0.25)(dense)
	dense = Dense(input_shape[0], activation='sigmoid')(flat)
	output = Reshape((input_shape[0],1))(dense)
	model = Model(input=inputs, output=output)
	return model


input_shape = (2*flankLength,4)
Nfilters = 32
Wfilter = 5
num_conv=2
num_layers = 8
l2_reg=0
lr = 0.002
opt = RMSprop(lr=lr,rho=0.9,epsilon=1e-8)
batch_size= 128
nb_epoch = 20
Niter = 4000

if model_str == "cnnattn":
	net = CNNAttn(input_shape)
if model_str == "cnn_lstm":
	net = cnn_lstm(input_shape)

checkpointer = ModelCheckpoint(filepath="./models/{}_{}.h5".format(model_str, flankLength),monitor='val_loss', verbose=1, save_best_only=True)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=1, verbose=0)
]

net.compile(optimizer=opt,loss='mse')
net.fit(Xtrain,ytrain, batch_size=batch_size, nb_epoch=nb_epoch,
validation_data=(Xval,yval), callbacks=callbacks)
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
        if vl <= val_loss[-1]:
            val_loss.append(vl)
        else:
            break
net.save('./models/{}_{}.h5'.format(model_str+"negTrain",flankLength))

yhat = net.predict(Xval)
#loss = net.evaluate(Xval,yval)

np.save("./predictions/{}_FL{}".format(model_str,flankLength),yhat)
np.save("./predictions/{}_FL{}_loss".format(model_str, flankLength), np.asarray(loss))
np.save("./predictions/{}_FL{}_val_loss".format(model_str, flankLength), np.asarray(val_loss))

#plot a random selection of sequences
#if not os.path.exists('./plots/{}'.format(flankLength)):
#    os.makedirs('./plots/{}'.format(flankLength))

#plot_ids = np.random.choice(yval.shape[0],50)
#for i in plot_ids:
#    plt.figure()
#    plt.plot(yval[i], label='truth')
#    plt.plot(yhat[i], label='prediction')
#    plt.legend()
#    plt.savefig('./plots/{}/{}'.format(flankLength,i))

f = open('mse.txt','a')
f.write('{},{},{}\n'.format(model_str, flankLength, loss))
f.close()
