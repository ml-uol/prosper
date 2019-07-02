# -*- coding: utf-8 -*-
#
#  Author:   Abdul-Saboor Sheikh <sheikh@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#



# import sys
# sys.path.insert(0, '..')

import numpy as np
from mpi4py import MPI
from scipy import stats

from abc import ABCMeta, abstractmethod

import prosper.utils.tracing as tracing
import prosper.utils.parallel as parallel

from prosper.em import Model
from prosper.utils.datalog import dlog


#=============================================================================#
# Abstract base class for mixture models

class MixtureModel(Model, metaclass=ABCMeta):
    def __init__(self, D, H, to_learn=['W', 'pies'], comm=MPI.COMM_WORLD):
        """

        """
        Model.__init__(self, comm)
        self.to_learn = to_learn

        # Model meta-parameters
        self.D = D
        self.H = H

    @tracing.traced
    def standard_init(self, data):
        """ Standard Initial Estimation for *W* and *sigma*.

        each *W* raw is set to the average over the data plus WGN of mean zero
        and var *sigma*/4. *sigma* is set to the variance of the data around 
        the computed mean. *pi* is set to 1./H . Returns a dict with the 
        estimated parameter set with entries "W", "pi" and "sigma".
        """
        comm = self.comm
        H = self.H
        my_y = data['y']
        my_N, D = my_y.shape

        assert D == self.D

        # Calculate averarge W
        W_mean = parallel.allmean(my_y, axis=0, comm=comm)               # shape: (D, )

        # Calculate data variance
        sigma_sq = parallel.allmean((my_y-W_mean)**2, axis=0, comm=comm) # shape: (D, )
        sigma_init = np.sqrt(sigma_sq).sum() / D                         # scalar

        # Initial W
        noise = sigma_init/4.
        W_init = W_mean[:,None] + np.random.normal(scale=noise, size=[D, H])    # shape: (H, D)`

        #Create and set Model Parameters, W columns have the same average!
        model_params = {'W'     : W_init }

        if 'pies' in self.to_learn:
            model_params['pies'] = np.ones(H) * 1./H

        return model_params

    @abstractmethod
    def check_params(self, model_params):
        """ Sanity check.

        Sanity-check the given model parameters. Raises an exception if 
        something is severely wrong.
        """
        pass

    @tracing.traced
    def generate_data(self, model_params, my_N):
        """ 
        Generate data according to the model. Internally uses generate_data_from_hidden.

        This method does _not_ obey gamma: The generated data may have more
        than gamma active causes for a given datapoint.
        """

        D = self.D
        H = self.H
        
        s = stats.rv_discrete(values = (np.arange(H),model_params['pies']), name = 'compProbDistr').rvs(size=my_N)

        return self.generate_from_hidden(model_params, {'s': s})

    @tracing.traced
    def select_partial_data(self, anneal, data):
        """ Select a partial data-set from data and return it.

        The fraction of datapoints selected is determined by anneal['partial'].
        If anneal['partial'] is equal to either 1 or 0 the whole dataset will 
        be returned.
        """
        partial = anneal['partial']

        if partial == 0 or partial == 1:              # partial == full data
            return data

        my_N, D = data.shape            
        my_pN = int(np.ceil(my_N * partial))

        if my_N == my_pN:                            # partial == full data
            return data

        # Choose subset...
        sel = np.random.permutation(my_N)[:my_pN]
        #sel.sort()

        # Construct partial my_pdata...
        return data[sel]

    @tracing.traced
    def step(self, anneal, model_params, data):
        """ Perform an EM-step """

        # Noisify model parameters
        model_params = self.noisify_params(model_params, anneal)

        # Sanity check model parameters
        model_params = self.check_params(model_params)

        # For partial EM-step: select batch
        pdata = self.select_partial_data(anneal, data)

        # Do E-step and calculate joint-probabilities
        post_comp_distr = self.E_step(anneal, model_params, pdata)

        # Use joint-probabilities to derive new parameter set
        new_model_params = self.M_step(anneal, model_params, post_comp_distr, pdata)

        # Log iboth model parameters and annealing parameters
        dlog.append_all(new_model_params)
        dlog.append_all(anneal.as_dict())
    
        return new_model_params

    @tracing.traced
    def inference(self, anneal, model_params, my_data, no_maps=10):
        """ To be implemented """
