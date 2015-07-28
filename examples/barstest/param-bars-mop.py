from pulp.utils.barstest import generate_bars
import numpy as np

# Filename for training data and ground truth parameters
#data_fname = "data.h5"    # if not specified, the data will be read from 
                           # 'data.h5' in the output_directory.

# Number of datapoints to generate
N = 50000

# Each datapoint is of D = size*size
size = 5

# Diemnsionality of the model
H=2*size     # number of latents
D=size**2    # dimensionality of observed data

# If we want individual data points to be normalized to sum to a constant
A = np.nan # default setting with no normalization of the data
#A = 10*D # switch on the data normalization. The normalization constant A has to be greater than D

# Import and instantiate a model
from pulp.em.mixturemodels.MoP import MoP

model = MoP(D, H, A = A)

# Model parameters used when artificially generating 
# ground-truth data. This will NOT be used for the learning
# process.
params_gt = {
    'W'     :  10*generate_bars(H),   # this function is in bars-create-data
    'pies'    : 1./H * np.ones(H)   # must sum to 1 for a mixture model
}

if not np.isnan(A):
    params_gt['W'] = model.normalize(params_gt['W'].T).T


