# -*- coding: utf-8 -*-
#
#  Author:   Abdul-Saboor Sheikh <sheikh@tu.berlin.de)
#  Lincense: Academic Free License (AFL) v3.0
#
# Reference: Keck, C., Savin, C., & Lücke, J. (2012). Feedforward 
# Inhibition and Synaptic Scaling–Two Sides of the Same Coin?. 
# PLoS computational biology, 8(3), e1002432.

import numpy as np
from mpi4py import MPI

import prosper.utils.tracing as tracing

from prosper.utils.datalog import dlog
from prosper.em.mixturemodels import MixtureModel

class MoP(MixtureModel):

    def __init__(self, D, H, to_learn=['pies', 'W'], A = np.nan, comm=MPI.COMM_WORLD):
        MixtureModel.__init__(self, D = D, H = H, to_learn = to_learn, comm = comm)

        if not np.isnan(A) and A <= D:
            A = 10*D 

        self.A = A

    @tracing.traced
    def standard_init(self, my_data):
        """ Standard Initial Estimation for W, sigma and mu.

        """
        comm = self.comm
        
        model_params = MixtureModel.standard_init(self, my_data)

        model_params = comm.bcast(model_params)       

        return model_params

    @tracing.traced
    def resume_init(self, h5_output):
        """ Standard Initial Estimation for W, sigma and mu.

        """
        comm = self.comm

        model_params = {}
        
        h5 = openFile(h5_output, 'r')
        # General information
        steps, H, D = h5.root.W.shape

        if 'W' in self.to_learn:
            model_params['W'] = h5.root.W[steps-1]

        if 'pies' in self.to_learn:
            model_params['pies'] = h5.root.pies[steps-1]


        model_params = comm.bcast(model_params)       

        return model_params

    def generate_from_hidden(self, model_params, my_hdata):
        """ Generate datapoints according to the model.

        Given the model parameters *model_params* return a dataset 
        of *N* datapoints.
        """
        D = self.D
        H = self.H
        
        # Create output arrays, y is data, s is ground-truth
        s = my_hdata['s']
        my_N = s.size

        y = np.zeros( (my_N, D) )

        W = model_params['W']

        for n in xrange(my_N):
            comp = s[n]

            for d in xrange(D):
                y[n,d] = np.random.poisson(W[d,s[n]], 1)


        # Build return structure
        return { 'y': y, 's': s }

    @tracing.traced
    def E_step(self, anneal, model_params, my_data):
        comm = self.comm
        beta = 1./anneal['T']
        A    = self.A

        my_y = my_data['y']
        if not np.isnan(A):
            my_y = self.normalize(my_y)

        return self.posterior(model_params, my_y, beta)

    @tracing.traced
    def M_step(self, anneal, model_params, suff_stats, my_data):

        comm = self.comm  
        H    = self.H
        D    = self.D
        A    = self.A      
 
        my_y = my_data['y']
        if not np.isnan(A):
            my_y = self.normalize(my_y)


        tiny = np.finfo(np.float64).tiny
        eps = np.finfo(np.float64).eps

        my_N, _ = my_y.shape
        N = comm.allreduce(my_N)        

        if 'W' in self.to_learn:
            
            my_W_num = np.zeros((D,H))

            for h in range(H):
                batch_size = 500
                y_ind = 0
                while y_ind < my_N:
                    if y_ind + batch_size > my_N:
                        batch_size = my_N - y_ind 

                    y_batch = my_y[y_ind:y_ind+batch_size]
                    y_batch = y_batch.reshape((batch_size,D))

                    my_W_num[:, h] += np.sum(my_y[y_ind:y_ind+batch_size] * suff_stats['posteriors_h'][y_ind:y_ind+batch_size,h][:,None],0)
                    
                    y_ind += batch_size

            
            W_num = np.empty_like(my_W_num)
            comm.Allreduce( [my_W_num, MPI.DOUBLE], [W_num, MPI.DOUBLE] )

            if np.isnan(A):
                my_sum_posteriors = np.sum(suff_stats['posteriors_h'], 0)
                sum_posteriors = np.empty_like(my_sum_posteriors)
                comm.Allreduce( [my_sum_posteriors, MPI.DOUBLE], [sum_posteriors, MPI.DOUBLE] )
            else:
                sum_posteriors = np.sum(W_num,0)/(self.A) + eps

            model_params['W'] = (W_num/sum_posteriors[None,:]) + eps

        if 'pies' in self.to_learn:

            my_posteriors = suff_stats['posteriors_h']
            
            sum_posteriors = np.empty( (H,) , dtype=np.float64)
            comm.Allreduce( [my_posteriors.sum(axis=0), MPI.DOUBLE], [sum_posteriors, MPI.DOUBLE] )

            sum_posteriors += tiny

            model_params['pies'] = sum_posteriors/np.sum(sum_posteriors)
        
        
        return model_params

    def check_params(self, model_params):
        assert np.isfinite(model_params['W']).all()
        assert np.isfinite(model_params['pies']).all()

        return model_params


    @tracing.traced
    def posterior(self, model_params, my_y, beta = 1.0):

        tiny = np.finfo(np.float64).tiny
        mx = np.finfo(np.float64).max / (self.H)

        my_N = my_y.shape[0]

        log_posteriors = self.log_p_y(model_params, my_y, beta = beta)
        log_posteriors += np.log(model_params['pies'])[None,:]  * beta
        posteriors = np.float64(np.exp(np.float128(log_posteriors)))
        posteriors[np.isnan(posteriors)] = tiny 
        posteriors[posteriors < tiny] = tiny
        posteriors[np.isinf(posteriors)] = mx

        return { 'posteriors_h' : posteriors/np.sum(posteriors,1)[:,None], 'logpj' : log_posteriors}

    @tracing.traced
    def log_p_y(self, model_params, my_y, beta = 1.0):

        comm = self.comm

        tiny = np.finfo(np.float64).tiny

        W = model_params['W']

        D         = self.D
        H         = self.H
        A         = self.A
        my_N,_    = my_y.shape
        N         = comm.allreduce(my_N)

        
        log_prob_y = np.zeros((my_N, H))

        
        for h in range(H):
            cur_w = np.float128(model_params['W'][:,h])
            log_cur_w = np.log(cur_w)
            batch_size = 500
            y_ind = 0
            while y_ind < my_N:
                if y_ind + batch_size > my_N:
                    batch_size = my_N - y_ind 

                y_batch = my_y[y_ind:y_ind+batch_size]

                if np.isnan(A):
                    log_prob_y_batch = np.sum((np.float128(y_batch) * log_cur_w[None,:]) - cur_w[None,:], 1)
                else:
                    log_prob_y_batch = np.sum((np.float128(y_batch) * log_cur_w[None,:]), 1)

                log_prob_y[y_ind:y_ind+batch_size, h] = log_prob_y_batch * beta
                
                y_ind += batch_size
            
            
        return log_prob_y


    @tracing.traced
    def normalize(self, my_y):

        D   = self.D
        A   = self.A

        eps = np.finfo(np.float64).eps
        my_y_sum = np.sum(my_y,1) + eps

        return ((A-D)/my_y_sum[:,None]) * my_y + 1

