#!/usr/bin/env python
#
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

import sys
sys.path.insert(0, '../..')

import numpy as np
from mpi4py import MPI

import tables

# Import 
from pulp.utils import create_output_path 
from pulp.utils.parallel import pprint, stride_data

from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
from pulp.visualize.gui import GUI, RFViewer, YTPlotter

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing

# Main
if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    pprint("="*40)
    pprint(" Running %d parallel processes" % comm.size) 
    pprint("="*40)
    
    # Parse command line arguments TODO: bit of error checking
    if "param_file" not in globals():
        param_fname = sys.argv[1]
    else:
        param_fname = param_file
    if "output_path" not in globals():
        output_path = sys.argv[2]

    params = {}
    execfile(param_fname, params)

    # Extract some parameters
    N = params['N']
    D = params['D']
    H = params['H']

    D2 = H // 2
    assert D2**2 == D

    # Load data 
    data_fname = params.get('data_fname', output_path+"/data.h5")
    with tables.openFile(data_fname, 'r') as data_h5:
        N_file = data_h5.root.y.shape[0]
        if N_file < N: 
            dlog.progress("WARNING: N=%d chosen but only %d data points available. " % (N, N_file))
            N = N_file

        first_y, last_y = stride_data(N)
        my_y = data_h5.root.y[first_y:last_y]
    
    my_data = {
        'y': my_y
    }

    # Prepare model...
    model = params['model']
    #anneal = params['anneal']

    # Configure DataLogger
    dlog.start_gui(GUI)
    print_list = ('T', 'Q', 'pi', 'sigma', 'N', 'MAE')
    dlog.set_handler(print_list, TextPrinter)
    dlog.set_handler(print_list, StoreToTxt, output_path +'/terminal.txt')
    #dlog.set_handler('Q', YTPlotter)
    dlog.set_handler('W', RFViewer, rf_shape=(D2, D2))
    #dlog.set_handler(('W', 'pi', 'sigma', 'mu', 'y', 'MAE', 'N'), StoreToH5, output_path +'/result.h5')
    if 'pi' in model.to_learn:
        dlog.set_handler(['pi'], YTPlotter)
    if 'pies' in model.to_learn:
        dlog.set_handler(['pies'], YTPlotter)
    if 'sigma' in model.to_learn:
        dlog.set_handler(['sigma'], YTPlotter)
    if 'mu' in model.to_learn:
        dlog.set_handler(['mu'], YTPlotter)
    #dlog.set_handler('y', RFViewer, rf_shape=(D2, D2))

    model_params = model.standard_init(my_data)
    
    # Choose annealing schedule
    anneal = LinearAnnealing(50)
    anneal['T'] = [(0, 2.), (.7, 1.)]
    anneal['Ncut_factor'] = [(0,0.),(2./3,1.)]
    anneal['anneal_prior'] = False
    
    # Create and start EM annealing
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params
    em.run()


    dlog.close(True)
    pprint("Done")

