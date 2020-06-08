import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

with h5py.File('results.hdf5', 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get('losses_batch')
        losses_batch = np.array(data)
        print('Shape of the array dataset_1: \n', losses_batch.shape)
        data = hf.get('val_losses')
        val_losses = np.array(data)
        print('Shape of the array dataset_2: n', val_losses)

        
