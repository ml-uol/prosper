# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from mpi4py import MPI

from itertools import combinations
from abc import ABCMeta, abstractmethod

import pulp.utils.tracing as tracing
import pulp.utils.parallel as parallel

from pulp.utils.datalog import dlog
from pulp.em import Model

#=============================================================================#
# Abstract base class for component analysis models

class CAModel(Model):
    __metaclass__ = ABCMeta
    """ Abstract base class for Sparse Coding models with binary latent variables
        and expectation tuncation (ET) based training scheme.
    
        This 
    """

    def __init__(self, D, H, Hprime, gamma, to_learn=['W', 'pi', 'sigma'], comm=MPI.COMM_WORLD):
        """ Constructor for ET based Sparse Coding models with binary latent variables.

        :param D: Dimension of observed data. 
        :type  D: int
        :param H: Number of dictionary elements to learn.
        :type  H: int
        :param Hprime: ET approximation parameter: number of latent units to 
            choose during preselction 
        :type  Hprime: int
        :param gamma: maximum number of active latent binary variables. This 
            parameter should be choosen to be larger than the expected sparseness
            of the model to be learned.
        :type  gamma: int

        The set of model parameters of an CAModel derived model typically consist of::

            model_param['W']:  dictionary elements (shape D times H)
            model_param['pi']: prior activation probability for the observed variables.
            model_param['sigma']: std-variance of observation noise
                
        """
        Model.__init__(self, comm)
        self.to_learn = to_learn

        # Model meta-parameters
        self.D = D
        self.H = H
        self.Hprime = Hprime
        self.gamma = gamma

        # some sanity checks
        assert Hprime <= H
        assert gamma <= Hprime
    
        # Noise Policy
        tol = 1e-5
        self.noise_policy = {
            'W'    : (-np.inf, +np.inf, False ),
            'pi'   : (    tol,  1.-tol, False ),
            'sigma': (     0., +np.inf, False )
        }

        # Generate state-space list
        sl = []
        for g in xrange(2,gamma+1):
            for s in combinations(range(Hprime), g):
                sl.append( np.array(s, dtype=np.int8) )
        self.state_list = sl

        no_states = len(sl)
        self.no_states = no_states
        
        # Generate state-matrix
        sm = np.zeros((no_states, Hprime), dtype=np.uint8)
        for i in xrange(no_states):
            s = sl[i]
            sm[i, s] = 1
        self.state_matrix = sm
        self.state_abs = sm.sum(axis=1)
        
    def generate_data(self, model_params, my_N):
        """ 
        Generate data according to the model. Internally uses generate_data_from_hidden.

        :param model_params: Ground-truth model parameters to use
        :type  model_params: dict
        :param my_N: number of datapoints to generate on this MPI rank
        :type  my_N: int

        This method does _not_ obey gamma: The generated data may have more
        than gamma active causes for a given datapoint.
        """
        H, D = self.H, self.D
        pies = model_params['pi']

        p = np.random.random(size=(my_N, H))    # Create latent vector
        s = p < pies                            # Translate into boolean latent vector

        return self.generate_from_hidden(model_params, {'s': s})
    
    @tracing.traced
    def select_partial_data(self, anneal, my_data):
        """ Select a partial data-set from my_data and return it.

        The fraction of datapoints selected is determined by anneal['partial'].
        If anneal['partial'] is equal to either 1 or 0 the whole dataset will 
        be returned.
        """
        partial = anneal['partial']

        if partial == 0 or partial == 1:              # partial == full data
            return my_data

        my_N, D = my_data['y'].shape            
        my_pN = int(np.ceil(my_N * partial))

        if my_N == my_pN:                            # partial == full data
            return my_data

        # Choose subset...
        sel = np.random.permutation(my_N)[:my_pN]
        sel.sort()

        # Construct partial my_pdata...
        my_pdata = {}
        for key, val in my_data.items():
            my_pdata[key] = val[sel]

        return my_pdata

    def check_params(self, model_params):
        """ Perform a sanity check on the model parameters. 
            Throw an exception if there are major violations; 
            correct the parameter in case of minor violations
        """
        return model_params


    @tracing.traced
    def step(self, anneal, model_params, my_data):
        """ Perform an EM-step """

        # Noisify model parameters
        model_params = self.noisify_params(model_params, anneal)

        # Sanity check model parameters
        model_params = self.check_params(model_params)

        # For partial EM-step: select batch
        my_pdata = self.select_partial_data(anneal, my_data)

        # Annotate partial dataset with hidden-state candidates
        my_pdata = self.select_Hprimes(model_params, my_pdata)

        # Do E-step and calculate joint-probabilities
        my_joint_prob = self.E_step(anneal, model_params, my_pdata)

        # Use joint-probabilities to derive new parameter set
        new_model_params = self.M_step(anneal, model_params, my_joint_prob, my_pdata)
        
        # Calculate Objecive Function
        #[Q, theo_Q] = self.objective(my_joint_prob, model_params, my_pdata)
        #dlog.append('Q', Q)
        #dlog.append('theo_Q', theo_Q)

        # Log both model parameters and annealing parameters
        dlog.append_all(new_model_params)
        dlog.append_all(anneal.as_dict())

        return new_model_params

    @tracing.traced
    def standard_init(self, data):
        """ Standard onitial estimation for model parameters.

        This implementation 
    
*W* and *sigma*.



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
        model_params = {
            'W'     : W_init, 
            'pi'    : 1./H,
            'sigma' : sigma_init
        }

        return model_params

    def inference(self, anneal, model_params, candidates, test_data, topK=10, logprob=False):
        """
        Perform inference with the learned model on test data and return the top K configurations with their posterior probabilities. 
        :param anneal: Annealing schedule, e.g., em.anneal 
        :type  anneal: pulp.em.annealling.Annealing
        :param model_params: Learned model parameters, e.g., em.lparams 
        :type  model_params: dict
        :param candidates: The list of candidate binary configurations, e.g., my_data['candidates'] 
        :type  candidates: numpy.ndarray
        :param test_data: The test data 
        :type  test_data: numpy.ndarray
        :param topK: The number of returned configurations 
        :type  topK: int
        :param logprob: Return probability or log probability
        :type  logprob: boolean
        """

        W = model_params['W']
        my_y = test_data
        H = self.H
        my_N, D = my_y.shape

        # Prepare return structure
        res = {
            's': np.zeros( (my_N, topK, H), dtype=np.int),
            'p': np.zeros( (my_N, topK) )
        }

        my_cand = candidates
        my_data = {
        'y': test_data,
        'candidates':my_cand
        }

        my_suff_stat = self.E_step(anneal, model_params, my_data)
        my_logpj  = my_suff_stat['logpj']
        my_corr   = my_logpj.max(axis=1)           # shape: (my_N,)
        my_logpjc = my_logpj - my_corr[:, None]    # shape: (my_N, no_states)
        my_pjc    = np.exp(my_logpjc)              # shape: (my_N, no_states)
        my_denomc = my_pjc.sum(axis=1)             # shape: (my_N)
        my_pjc = my_pjc/my_denomc[:,None]
        my_logpjc += -np.log(my_denomc)[:,None]
        
        idx = np.argsort(my_logpjc, axis=-1)[:, -1:-(topK+1):-1]
        for n in xrange(my_N):                                   # XXX Vectorize XXX
            for m in xrange(topK):
                this_idx = idx[n,m]
                if logprob:
                    res['p'][n,m] = my_logpjc[n, this_idx] 
                else:
                    res['p'][n,m] = my_pjc[n, this_idx] 
                if this_idx == 0:
                    pass
                elif this_idx < (H+1):
                    res['s'][n,m,this_idx-1] = 1
                else:
                    s_prime = self.state_matrix[this_idx-H-1]
                    res['s'][n,m,my_cand[n,:]] = s_prime

        return res

    def inference_marginal(self, anneal, model_params, candidates, test_data, logprob=False):
        """
        Perform inference with the learned model on test data and return the top K configurations with their posterior probabilities. 
        :param anneal: Annealing schedule, e.g., em.anneal 
        :type  anneal: pulp.em.annealling.Annealing
        :param model_params: Learned model parameters, e.g., em.lparams 
        :type  model_params: dict
        :param candidates: The list of candidate binary configurations, e.g., my_data['candidates'] 
        :type  candidates: numpy.ndarray
        :param test_data: The test data 
        :type  test_data: numpy.ndarray
        :param topK: The number of returned configurations 
        :type  topK: int
        :param logprob: Return probability or log probability
        :type  logprob: boolean
        """

        W = model_params['W']
        my_y = test_data
        H = self.H
        my_N, D = my_y.shape

        # Prepare return structure
        logp_m = np.zeros((my_N, H))

        my_cand = candidates
        my_data = {
        'y': test_data,
        'candidates':my_cand
        }

        my_suff_stat = self.E_step(anneal, model_params, my_data)
        my_logpj  = my_suff_stat['logpj']
        my_corr   = my_logpj.max(axis=1)           # shape: (my_N,)
        my_logpjc = my_logpj - my_corr[:, None]    # shape: (my_N, no_states)
        my_pjc    = np.exp(my_logpjc)              # shape: (my_N, no_states)
        my_denomc = my_pjc.sum(axis=1)             # shape: (my_N)
        my_pjc = my_pjc/my_denomc[:,None]
        my_logpjc += -np.log(my_denomc)[:,None]

        from scipy.misc import logsumexp
        
        for n in range(my_N):
            for h in range(H):
                if h in my_cand[n,:]:
                    idx = np.where(my_cand[n]==h)[0][0]
                    logp = np.hstack([my_logpjc[n,h+1],my_logpjc[n,H+1:][self.state_matrix[:,idx]==1]])
                    logp_m[n,h] = logsumexp(logp)
                else:
                    logp_m[n,h] = my_logpjc[n,h+1]

        if logprob:
            return logp_m
        else:
            return np.exp(logp_m)
