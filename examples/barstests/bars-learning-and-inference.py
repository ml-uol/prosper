#!/usr/bin/env python
#
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

import sys
sys.path.insert(0, '../..')

import numpy as np
from mpi4py import MPI

from pulp.utils import create_output_path 
from pulp.utils.parallel import pprint, stride_data

from pulp.utils.barstest import generate_bars_dict
from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing

# Main
if __name__ == "__main__":
    import argparse

    comm = MPI.COMM_WORLD

    if len(sys.argv) != 2:
        pprint("Usage: %s <parameter-file>" % sys.argv[0])
        pprint()
        exit(1)

    param_fname = sys.argv[1]

    params = {}
    execfile(param_fname, params)

    # Extract some parameters
    N = params.get('N', 5000)            # no. of datapoint in the testset
    N_train = int(0.9*N)
    N_test  = N-N_train
    size = params.get('size', 5)         # width / height of bars images
    p_bar = params.get('p_bar', 1./size) # prob. for a bar to be active
    D = params.get('D', size**2)         # observed dimensionality
    H = params.get('H', 2*size)          # latent dimensionality
    model = params['model']              # the actual generative model

    # Ground truth parameters -- only used for generation
    params_gt = params.get('params_gt')  # Ground truth param 

    # Create output path
    output_path = create_output_path()

    # Disgnostic output
    pprint("="*40)
    pprint(" Running bars experiment (%d parallel processes)" % comm.size) 
    pprint("  size of training set:   %d" % N)
    pprint("  size of bars images:    %d x %d" % (size, size))
    pprint("  number of hiddens:      %d" % H)
    pprint("  saving results to:      %s" % output_path)
    pprint()

    # Generate bars data
    # params_gt = {
        # 'W'     :  10*generate_bars_dict(H),
        # 'pi'    :  p_bar,
        # 'sigma' :  2.0
    # }
    my_data = model.generate_data(params_gt, N_train // comm.size)
    my_test_data = model.generate_data(params_gt, N_test // comm.size)

    # Configure DataLogger
    print_list = ('T', 'Q', 'pi', 'sigma', 'N', 'MAE')
    dlog.set_handler(print_list, TextPrinter)
    dlog.set_handler(print_list, StoreToTxt, output_path +'/terminal.txt')

    model_params = model.standard_init(my_data)
    
    if 'anneal' in params:
        anneal = params.get('anneal')
    else:
        # Choose annealing schedule
        anneal = LinearAnnealing(50)
        anneal['T'] = [(0, 2.), (.7, 1.)]
        anneal['Ncut_factor'] = [(0,0.),(2./3,1.)]
        anneal['anneal_prior'] = False
    
    # Create and start EM annealing
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params
    em.lparams = params_gt
    em.run()
    pprint(" Infering Bars from the test set")
    res=model.inference(anneal,em.lparams,my_test_data,no_maps=-1)

    dlog.close(True)
    pprint("Done")

