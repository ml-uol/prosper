from prosper.utils.barstest import generate_bars_dict
import numpy as np

# Filename for training data and ground truth parameters
#data_fname = "data.h5"    # if not specified, the data will be read from 
                           # 'data.h5' in the output_directory.

# Number of datapoints to generate
N = 20000

# Each datapoint is of D = size*size
size = 5

# Diemnsionality of the model
H=2*size     # number of latents
D=size**2    # dimensionality of observed data

# If the sigma(s) to be learnt should only be diagonal or full matrices
#sigmas_sq_type = 'full'
sigmas_sq_type = 'diagonal'

# Import and instantiate a model
from prosper.em.mixturemodels.MoG import MoG

model = MoG(D, H, sigmas_sq_type = sigmas_sq_type)

# Model parameters used when artificially generating 
# ground-truth data. This will NOT be used for the learning
# process.
params_gt = {
    'W'     :  10*generate_bars_dict(H),   # this function is in bars-create-data
    'pies'    : 1./H * np.ones(H)   # must sum to 1 for a mixture model
}

if sigmas_sq_type == 'full':
    sigma = np.eye(D)
elif sigmas_sq_type == 'diagonal':
    sigma = np.ones(D) 
           
sigmas_sq = np.zeros(tuple([H])+sigma.shape)

for h in range(H):
    sigmas_sq[h] = sigma

params_gt['sigmas_sq'] = sigmas_sq
