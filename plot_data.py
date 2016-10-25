import h5py
from scipy.io import loadmat
from matplotlib import pyplot as plt

train = h5py.File('./data/train.mat')
val = loadmat('./data/valid.mat')
test = loadmat('./data/test.mat')

def summary(data, name):
    '''
    prints a summary of an h5py dataset or a dictionary

    inputs:
        data - (h5py File or python dictionary), data to summarize
        name - name to use when printing the summary
    '''
    print '{} keys={}'.format(name,data.keys())
    for key in data.keys():
        if 'shape' in dir(data[key]):
            print '{} {} shape={}'.format(name,key,data[key].shape)

summary(train, 'Training data')
summary(val, 'Validation data')
summary(test, 'Validation data')

#Let's look at one example
y = train['traindata'][:,0]
x = train['trainxdata'][:,:,0]

plt.pcolor(x.T)
plt.show()

plt.scatter(range(0,len(y)),y)
plt.show()
