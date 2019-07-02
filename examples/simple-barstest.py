#!/usr/bin/env python
import sys
sys.path.insert(0, '..')

import numpy as np
from mpi4py import MPI

from prosper.utils import create_output_path 
from prosper.utils.parallel import pprint, stride_data
from prosper.utils.barstest import generate_bars_dict

from prosper.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt

from prosper.em import EM
from prosper.em.annealing import LinearAnnealing
from prosper.em.camodels.bsc_et import BSC_ET


#=============================================================================
# Parameters

D2     = 5
N      = 1000
Hprime = 6
gamma  = 5

#=============================================================================
# Main
comm = MPI.COMM_WORLD

pprint("="*70)
pprint(" Running %d parallel processes" % comm.size) 
pprint("="*70)
 
H = 2*D2      # number of latent units
D = D2**2     # total size of image in pixels

my_N = N // comm.size

# Some sanity checks
assert Hprime <= H
assert gamma <= Hprime
assert D == D2**2

# Configure DataLogger
print_list = ('T', 'pi', 'sigma')
dlog.set_handler(print_list, TextPrinter)

# Invent some ground truth parameter models
params_gt = {
    'W'     :  10*generate_bars_dict(H),  
    'pi'    :  2. / H,
    'sigma' :  1.0
}

# Use model to generate data 
model = BSC_ET(D, H, Hprime, gamma)
my_data = model.generate_data(params_gt, my_N)

model_params = model.standard_init(my_data)
    
# Choose annealing schedule
anneal = LinearAnnealing(50)
anneal['T'] = [(15, 1.), (-10, 1.)]
anneal['Ncut_factor'] = [(0,0.),(2./3,1.)]
anneal['anneal_prior'] = False
    
# Create and start EM annealing
em = EM(model=model, anneal=anneal)
em.data = my_data
em.lparams = model_params
em.run()

dlog.close()
pprint("Done")

