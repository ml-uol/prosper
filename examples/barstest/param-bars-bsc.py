from pulp.utils.barstest import generate_bars

# Filename for training data and ground truth parameters
#data_fname = "data.h5"    # if not specified, the data will be read from 
                           # 'data.h5' in the output_directory.

# Number of datapoints to generate
N = 1000

# Each datapoint is of D = size*size
size = 5

# Diemnsionality of the model
H=2*size     # number of latents
D=size**2    # dimensionality of observed data

# Approximation parameters for Expectation Truncation
Hprime = 8
gamma = 5
 
# Import and instantiate a model
from pulp.em.camodels.bsc_et import BSC_ET
model = BSC_ET(D, H, Hprime, gamma)

# Model parameters used when artificially generating 
# ground-truth data. This will NOT be used for the learning
# process.
params_gt = {
    'W'     :  10*generate_bars(H),   # this function is in bars-create-data
    'pi'    :  2. / H,
    'sigma' :  1.0
}

