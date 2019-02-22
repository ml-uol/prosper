import numpy as np
from prosper.utils.barstest import generate_bars_dict

np.random.seed(1)

# Number of datapoints to generate
N = 1000

# Each datapoint is of D = size*size
size = 5

# Dimensionality of the model
H=2*size     # number of latents
D=size**2    # dimensionality of observed data

# Approximation parameters for Expectation Truncation
Hprime = 8
gamma = 5
 
# Import and instantiate a model
from prosper.em.camodels.mca_et import MCA_ET
model = MCA_ET(D, H, Hprime, gamma)


# Ground truth parameters. Only used to generate training data.
params_gt = {
    'W'     :  10*generate_bars_dict(H),
    'pi'    :  2.0 / size,
    'sigma' :  2.0
}

# Choose annealing schedule
from prosper.em.annealing import LinearAnnealing
anneal = LinearAnnealing(300)
anneal['T'] = [(0, 4.), (.8, 1.)]
