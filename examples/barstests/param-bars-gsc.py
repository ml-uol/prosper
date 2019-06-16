from prosper.utils.barstest import generate_bars_dict
import numpy as np

#np.random.seed(1)

# Number of datapoints to generate
N = 10000

# Each datapoint is of D = size*size
size = 5

# Probability for the generated bars
p_bar = 1. / size

# Dimensionality of the model
H=2*size     # number of latents
D=size**2    # dimensionality of observed data

# Approximation parameters for Expectation Truncation
Hprime = 7
gamma = 4
 
# Type of observation noise assumed by the model
sigma_sq_type = 'scalar' # defaul noise is assumed to be scalar

#sigma_sq_type = 'diagonal' # uncomment to make the model assume independent observation
                         # noise per observed dimension d of D. Makes the algorithm 
                         # slightly more costly.

#sigma_sq_type = 'full'   # uncomment to make the model assume independent observation
                         # noise per observed dimension d of D. Makes the algorithm 
                         # even more costly in terms of both computation and memory. 
                         # Should be avoided for large D.

# Import and instantiate a model
from prosper.em.camodels.gsc_et import GSC
model = GSC(D, H, Hprime, gamma, sigma_sq_type)



# Model parameters used when artificially generating 
# ground-truth data. This will NOT be used for the learning
# process.
params_gt = {
    'W'     :  10*generate_bars_dict(H),   # this function is in bars-create-data
    'pi'    :  2./H * np.ones(H),
    'mu'    :  np.ones(H),
    'psi_sq'   :  np.eye(H)
}

if sigma_sq_type == 'scalar':
    params_gt['sigma_sq'] =  1.0
elif sigma_sq_type == 'diagonal':
    params_gt['sigma_sq'] =  np.ones(D)
elif sigma_sq_type == 'full':
    params_gt['sigma_sq'] =  np.eye(D)


# Choose annealing schedule
from prosper.em.annealing import LinearAnnealing
anneal = LinearAnnealing(150)
anneal['T'] = [(0, 4.), (.8, 1.)]
anneal['Ncut_factor'] = [(0,0.),(2./3,1.)]
anneal['anneal_prior'] = False