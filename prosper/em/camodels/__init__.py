# -*- coding: utf-8 -*-



import numpy as np
from mpi4py import MPI

from scipy.special import logsumexp

from itertools import combinations
from abc import ABCMeta, abstractmethod
import six

import prosper.utils.tracing as tracing
import prosper.utils.parallel as parallel

from prosper.utils.datalog import dlog
from prosper.em import Model


def generate_state_matrix(Hprime, gamma):
    """Full combinatorics of Hprime-dim binary vectors with at most gamma ones.

    :param Hprime: Vector length
    :type Hprime: int
    :param gamma: Maximum number of ones
    :param gamma: int

    """
    sl = []
    for g in range(2,gamma+1):
        for s in combinations(list(range(Hprime)), g):
            sl.append( np.array(s, dtype=np.int8) )
    state_list = sl

    no_states = len(sl)
    no_states = no_states

    sm = np.zeros((no_states, Hprime), dtype=np.uint8)
    for i in range(no_states):
        s = sl[i]
        sm[i, s] = 1
    state_matrix = sm
    state_abs = sm.sum(axis=1)
    #print("state matrix updated")

    return state_list, no_states, state_matrix, state_abs


#=============================================================================#
# Abstract base class for component analysis models
six.add_metaclass(ABCMeta)
class CAModel(Model):
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
        self.state_list, self.no_states, self.state_matrix, self.state_abs = generate_state_matrix(Hprime, gamma)
        
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


    def compute_lpj(self, anneal, model_params, my_data):
        """Determine candidates and compute log-pseudo-joint.

        :param anneal: Annealing schedule, e.g., em.anneal 
        :type  anneal: prosper.em.annealling.Annealing
        :param model_params: Learned model parameters, e.g., em.lparams 
        :type  model_params: dict        
        :param my_data: Data stored in field 'y'.
        :type  my_data: dict
        """

        assert 'y' in my_data, "Key 'y' in my_data dict not defined."        
        my_data = self.select_Hprimes(model_params, my_data)

        my_suff_stat = self.E_step(anneal, model_params, my_data)
        return my_suff_stat['logpj'], my_data['candidates']


    def inference(self, anneal, model_params, test_data, topK=10, logprob=False, adaptive=True,
        Hprime_max=None, gamma_max=None):
        """
        Perform inference with the learned model on test data and return the top K configurations with their
        posterior probabilities. 
        :param anneal: Annealing schedule, e.g., em.anneal 
        :type  anneal: prosper.em.annealling.Annealing
        :param model_params: Learned model parameters, e.g., em.lparams 
        :type  model_params: dict        
        :param test_data: The test data stored in field 'y'. Candidates stored in 'candidates' (optional).
        :type  test_data: dict
        :param topK: The number of returned configurations 
        :type  topK: int
        :param logprob: Return probability or log probability
        :type  logprob: boolean
        :param adaptive: Adjust Hprime, gamma to be greater than the number of active units in the MAP state
        :type adaptive: boolean
        :param Hprime_max: Upper limit for Hprime adjustment 
        :type Hprime_max: int
        :param gamma_max: Upper limit for gamma adjustment 
        :type gamma_max: int
        """

        assert 'y' in test_data, "Key 'y' in test_data dict not defined."
        
        model_params = self.check_params(model_params)

        comm = self.comm
        my_y = test_data['y']
        my_N, D = my_y.shape
        H = self.H
        Hprime_start, gamma_start = self.Hprime, self.gamma

        # Prepare return structure
        if topK==-1:
            topK=self.state_matrix.shape[0]
        res = {
            's': np.zeros( (my_N, topK, H), dtype=np.int8),
            'm': np.zeros( (my_N, H) ),
            'p': np.zeros( (my_N, topK) ),
            'gamma': np.zeros( (my_N,) ),
            'Hprime': np.zeros( (my_N,) )
        }

        test_data_tmp = {'y' : my_y}
        which = np.ones(my_N,dtype=bool)
        
        while which.any():

            ind_n = np.where(which)[0]            

            my_logpj, my_cand = self.compute_lpj(anneal, model_params, test_data_tmp)
            my_corr   = my_logpj.max(axis=1)           # shape: (my_N,)
            my_logpjc = my_logpj - my_corr[:, None]    # shape: (my_N, no_states)
            my_pjc    = np.exp(my_logpjc)              # shape: (my_N, no_states)
            my_denomc = my_pjc.sum(axis=1)             # shape: (my_N)
            my_logpjc += -np.log(my_denomc)[:,None]            
            idx = np.argsort(my_logpjc, axis=-1)[:, ::-1]

            for n in range(my_N):                                   # XXX Vectorize XXX
                n_ = ind_n[n]
                res['Hprime'][n_] = self.Hprime
                res['gamma'][n_] = self.gamma                
                for m in range(topK):                    
                    this_idx = idx[n,m]
                    if logprob:
                        res['p'][n_,m] = my_logpjc[n, this_idx] 
                    else:
                        res['p'][n_,m] = my_pjc[n, this_idx] 
                    if this_idx == 0:
                        pass
                    elif this_idx < (H+1):
                        res['s'][n_,m,this_idx-1] = 1
                    else:
                        s_prime = self.state_matrix[this_idx-H-1]                        
                        res['s'][n_,m,my_cand[n,:]] = s_prime

                for h in range(H):
                    if h in my_cand[n,:]:
                        idx_ = np.where(my_cand[n]==h)[0][0]
                        logp = np.hstack([my_logpjc[n,h+1],my_logpjc[n,H+1:][self.state_matrix[:,idx_]==1]])
                        res['m'][n_,h] = logsumexp(logp)
                    else:
                        res['m'][n_,h] = my_logpjc[n,h+1]

            if not adaptive:
                break

            which = ((res['s'][:,0,:]!=0).sum(-1)==self.gamma) # shape: (my_N,)
            if not which.any():
                break
            else:
                if (Hprime_max is not None and self.Hprime == Hprime_max) and (gamma_max is not None and self.gamma == gamma_max):
                    break
            test_data_tmp['y']=my_y[which]
            my_N=np.sum(which)            
            print("Rank %i: For %i data points MAP state has activity equal to gamma." % (comm.rank, my_N))

            if (self.Hprime == self.H) or (Hprime_max is not None and self.Hprime == Hprime_max):
                pass
            else:
                self.Hprime+=1

            if (self.gamma == self.H) or (gamma_max is not None and self.gamma == gamma_max):
                continue
            else:
                self.gamma+=1

            print("Rank %i: Updating state matrix and running again." % comm.rank)
            self.state_list, self.no_states, self.state_matrix, self.state_abs = generate_state_matrix(self.Hprime, self.gamma)

        if not logprob:
            res['m'] = np.exp(res['m'])

        comm.Barrier()

        self.Hprime, self.gamma = Hprime_start, gamma_start
        self.state_list, self.no_states, self.state_matrix, self.state_abs = generate_state_matrix(self.Hprime, self.gamma)

        return res
