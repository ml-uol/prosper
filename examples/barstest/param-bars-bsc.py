
# Number of datapoints to generate
N = 1000

# Each datapoint is of D = size*size
size = 5

# Probability for the generated bars
p_bar = 1. / size

# Dimensionality of the model
H=2*size     # number of latents
D=size**2    # dimensionality of observed data

# Approximation parameters for Expectation Truncation
Hprime = 8
gamma = 5
 
# Import and instantiate a model
from pulp.em.camodels.bsc_et import BSC_ET
model = BSC_ET(D, H, Hprime, gamma)
