import h5py
import numpy as np
class HDF5Generator:

    def __init__(self,filename,batch_size):
        self.filename = filename
        self.batch_size = batch_size

        self.data = h5py.File(filename,'r')

    def generate(self):
