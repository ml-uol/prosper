#!/usr/bin/env python
#
#  Lincense: Academic Free License (AFL) v3.0
#

import sys
sys.path.insert(0, '../..')

import numpy as np
from mpi4py import MPI

from prosper.utils import create_output_path 
from prosper.utils.parallel import pprint, stride_data

from prosper.utils.barstest import generate_bars_dict
from prosper.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt

from prosper.em import EM
from prosper.em.annealing import LinearAnnealing


if __name__ == "__main__":
    import argparse

    comm = MPI.COMM_WORLD

    if len(sys.argv) != 2:
        pprint("Usage: %s <parameter-file>" % sys.argv[0])
        pprint()
        exit(1)

    param_fname = sys.argv[1]

    params = {}
    exec(compile(open(param_fname).read(), param_fname, 'exec'), params)

    # Extract some parameters
    N = params.get('N', 5000)            # no. of datapoint in the testset
    N_train = int(0.9*N)
    N_test  = N-N_train
    size = params.get('size', 5)         # width / height of bars images
    p_bar = params.get('p_bar', 1./size) # prob. for a bar to be active
    D = params.get('D', size**2)         # observed dimensionality
    H = params.get('H', 2*size)          # latent dimensionality
    model = params['model']              # the actual generative model
    model_str = model.__class__.__name__

    # Ground truth parameters -- only used for generation
    params_gt = params.get('params_gt')  # Ground truth param 

    # Create output path
    output_path = create_output_path('learning-and-inference-' + param_fname)

    # Disgnostic output
    pprint("="*40)
    pprint(" Running bars experiment (%d parallel processes)" % comm.size) 
    pprint("  size of training set:   %d" % N)
    pprint("  size of bars images:    %d x %d" % (size, size))
    pprint("  number of hiddens:      %d" % H)
    pprint("  saving results to:      %s" % output_path)
    pprint()

    my_data = model.generate_data(params_gt, N_train // comm.size)
    my_test_data = model.generate_data(params_gt, N_test // comm.size)

    # Configure DataLogger
    store_list = ('*')
    print_list = ('T', 'Q', 'pi', 'sigma', 'N', 'MAE')
    dlog.set_handler(print_list, TextPrinter)    
    dlog.set_handler(print_list, StoreToTxt, output_path +'/terminal.txt')
    dlog.set_handler(store_list, StoreToH5, output_path +'/result.h5')

    dlog.append('Hprime_start', model.Hprime)
    dlog.append('gamma_start', model.gamma)

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
    em.run()
    
    pprint(" Infering Bars from the test set")    
    prms = {'topK' : 5, 'adaptive' : True, 'Hprime_max' : model.Hprime+3, 'gamma_max' : model.gamma+3}
    params_for_inference = em.lparams # alternatively: params_for_inference = params_gt
    res=model.inference(anneal,params_for_inference,my_test_data,**prms)

    # Store results
    N_test_store = 30
    N_test_store = np.min([N_test_store, my_test_data['y'].shape[0]])
    dlog.append('test_data', my_test_data['y'][:N_test_store,:])
    for n in range(N_test_store):
        if model_str == 'GSC':
            tmp = params_gt['W'] * my_test_data['z'][n,:][None,:]
        elif model_str == 'Ternary_ET':
            tmp = params_gt['W'] * my_test_data['s'][n,:][None,:].astype(float)
        else:
            tmp = params_gt['W']
        key = 'test_n%i_comps_gt' % n
        if my_test_data['s'][n,:].astype(bool).sum() > 0:
            dlog.append(key, tmp[:,my_test_data['s'][n,:].astype(bool)])
        else:
            dlog.append(key, 0)
        dlog.append('test_n%i_p_top%i' % (n,prms['topK']), res['p'][n,:prms['topK']])
        dlog.append('test_n%i_Hprime' % n, res['Hprime'][n])
        dlog.append('test_n%i_gamma' % n, res['gamma'][n])        
        for k in range(prms['topK']):
            key = 'test_n%i_comps_top%i' % (n,k)
            if res['s'][n,k,:].astype(bool).sum() > 0:
                if model_str == 'Ternary_ET':
                    tmp = params_for_inference['W'] * res['s'][n,k,:][None,:].astype(float)
                else:
                    tmp = params_for_inference['W']
                dlog.append(key, tmp[:,res['s'][n,k,:].astype(bool)])
            else:
                dlog.append(key, 0)

    dlog.close()
    pprint("Done")    
