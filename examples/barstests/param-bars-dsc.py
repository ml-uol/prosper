import numpy as np
from prosper.utils.barstest import generate_bars_dict


# Number of datapoints to generate
N = 1000

# Each datapoint is of D = size*size
size = 5

# Dimensionality of the model
H=2*size     # number of latents
D=size**2    # dimensionality of observed data

# Approximation parameters for Expectation Truncation
Hprime = 7
gamma = 5

# Latent states of dsc
states = np.array([0.,1.,2.])
pi = np.array([0.8,0.15,0.05])
 
# Import and instantiate a model
from prosper.em.camodels.dsc_et import DSC_ET
model = DSC_ET(D, H, Hprime, gamma, states=states)


# Ground truth parameters. Only used to generate training data.
params_gt = {
    'W'     :  10*generate_bars_dict(H),
    'pi'    :  pi,
    'sigma' :  2.0
}

from prosper.em.annealing import LinearAnnealing
anneal = LinearAnnealing(10)
anneal['T'] = [(0, 2.), (.7, 1.)]
anneal['Ncut_factor'] = [(0,0.),(2./3,1.)]
anneal['anneal_prior'] = False