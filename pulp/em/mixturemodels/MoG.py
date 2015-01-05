# -*- coding: utf-8 -*-
#
#  Author:   Abdul-Saboor Sheikh <sheikh@tu-berlin.de)
#  Lincense: Academic Free License (AFL) v3.0
#

import numpy as np
from mpi4py import MPI

import pulp.utils.tracing as tracing

from pulp.utils.datalog import dlog
from pulp.em.mixturemodels import MixtureModel

class MoG(MixtureModel):

    def __init__(self, D, H, to_learn=['pies', 'W','sigmas_sq'], sigmas_sq_type = 'full', comm=MPI.COMM_WORLD):
        MixtureModel.__init__(self, D = D, H = H, to_learn = to_learn, comm = comm)

        self.sigmas_sq_type = sigmas_sq_type

    @tracing.traced
    def standard_init(self, my_data):
        """ Standard Initial Estimation for W, sigmas and pies.

        """
        comm = self.comm
        
        H       = self.H
        my_y    = my_data['y']
        N, D    = my_y.shape
           

        model_params = MixtureModel.standard_init(self, my_data)

        if 'sigmas_sq' in self.to_learn:
            # Calculate sigma
            if self.sigmas_sq_type == 'full':
                sigma = comm.bcast( np.cov(my_y.T) + (0.001 * np.eye(D)) )
            elif self.sigmas_sq_type == 'diagonal':
                sigma = comm.bcast( np.var(my_y, axis = 0) + 0.001)   
           
            
            y_min = np.min(comm.allgather(np.min(my_y,0)))
            y_max = np.max(comm.allgather(np.max(my_y,0)))
            y_step = (y_max - y_min)/H

            sigmas_sq = np.zeros(tuple([H])+sigma.shape)

            for h in range(H):
                sigmas_sq[h] = sigma
                

            
            model_params['sigmas_sq'] = sigmas_sq
            
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


        if 'sigmas_sq' in self.to_learn:

            prev_sigmas_sq = h5.root.sigmas_sq[steps-1]
            sigmas_sq = prev_sigmas_sq.copy()

            if len(h5.root.sigmas_sq.shape) == 4:
                if self.sigmas_sq_type == 'diagonal':
                    sigmas_sq = np.zeros((H,D))
                    for h in range(H):
                        sigmas_sq[h] = prev_sigmas_sq[h].diagonal()
            elif len(h5.root.sigmas_sq.shape) == 3:
                if self.sigmas_sq_type == 'full':
                    sigmas_sq = np.zeros((H,D,D))
                    for h in range(H):
                        sigmas_sq[h] = np.diag(prev_sigmas_sq[h])
            model_params['sigmas_sq'] = sigmas_sq

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

        W = model_params['W'].T

        for n in xrange(my_N):
            comp = s[n]
            if self.sigmas_sq_type == 'full':
                sigma = model_params['sigmas_sq'][comp].diagonal()
            elif self.sigmas_sq_type == 'diagonal':
                sigma = model_params['sigmas_sq'][comp]
   
            y[n] =  W[comp] + np.sqrt(sigma) * np.random.randn(D)


        # Build return structure
        return { 'y': y, 's': s }

    @tracing.traced
    def E_step(self, anneal, model_params, my_data):
        comm = self.comm
        beta = 1./anneal['T']
        
        my_y = my_data['y']

        return self.posterior(model_params, my_y, beta)

    @tracing.traced
    def M_step(self, anneal, model_params, suff_stats, my_data):

        comm = self.comm        
 
        my_y = my_data['y']

        tiny = np.finfo(np.float64).tiny
        eps = np.finfo(np.float64).eps

        posteriors = suff_stats['posteriors_h']

        my_N, H = posteriors.shape
        my_N, D = my_y.shape
        N = comm.allreduce(my_N)        
        
        sum_posteriors = np.empty( (H,) , dtype=np.float64)
        comm.Allreduce( [posteriors.sum(axis=0), MPI.DOUBLE], [sum_posteriors, MPI.DOUBLE] )
        
        sum_posteriors += tiny

        # now compute new mu(s) and sigma(s) for each of the components
        if 'W' in self.to_learn:    
            my_num = np.zeros((D,H))
            for idx in range(my_N):
                my_num += posteriors[idx][None,:] * my_y[idx][:,None]
                
            W_num = np.empty_like(my_num)
            comm.Allreduce( [my_num, MPI.DOUBLE], [W_num,MPI.DOUBLE])
            model_params['W'] = W_num * np.power(sum_posteriors, -1)[None,:]
                
                
        if 'sigmas_sq' in self.to_learn:    
        
            if self.sigmas_sq_type == 'full':
                my_sigmas_sq = np.zeros((H,D,D))
                for idx in range(my_N):                    
                    my_sigmas_sq += posteriors[idx][:,None,None] * np.outer(my_y[idx],my_y[idx])[None,:,:]
                
                sigmas_sq = np.empty_like( my_sigmas_sq )
                comm.Allreduce( [my_sigmas_sq, MPI.DOUBLE], [sigmas_sq, MPI.DOUBLE] )
                model_params['sigmas_sq'] = (sigmas_sq * np.power(sum_posteriors, -1)[:,None,None])
                
                for h in range(H):
                    model_params['sigmas_sq'][h,:,:] -= np.outer(model_params['W'][:,h],model_params['W'][:,h]) 
                    
            elif self.sigmas_sq_type == 'diagonal':
                my_sigmas_sq =  np.zeros( (H,D) )
                for idx in range(my_N):
                    my_sigmas_sq += posteriors[idx][:,None] * (my_y[idx]**2)[None,:]
                
                sigmas_sq = np.empty_like( my_sigmas_sq )
                comm.Allreduce( [my_sigmas_sq, MPI.DOUBLE], [sigmas_sq, MPI.DOUBLE] )
                model_params['sigmas_sq'] = (sigmas_sq * np.power(sum_posteriors, -1)[:,None])
                
                model_params['sigmas_sq'] -= model_params['W'].T**2
            
        if 'pies' in self.to_learn:
            model_params['pies'] = sum_posteriors/np.sum(sum_posteriors)
            
        
        return model_params

    def check_params(self, model_params):
        assert np.isfinite(model_params['W']).all()
        assert np.isfinite(model_params['sigmas_sq']).all()
        assert np.isfinite(model_params['pies']).all()

        return model_params


    @tracing.traced
    def posterior(self, model_params, my_y, beta = 1.0):

        tiny = np.finfo(np.float64).tiny
        mx = np.finfo(np.float64).max / (self.H)

        my_N = my_y.shape[0]

        log_posteriors = self.log_p_y(model_params, my_y, beta = beta)
        log_posteriors += np.log(model_params['pies'])[None,:]  * beta
        posteriors = np.float64(np.exp(log_posteriors))
        posteriors[np.isnan(posteriors)] = tiny 
        posteriors[posteriors < tiny] = tiny
        posteriors[np.isinf(posteriors)] = mx

        return { 'posteriors_h' : posteriors/np.sum(posteriors,1)[:,None], 'logpj' : log_posteriors}


    @tracing.traced
    def log_p_y(self, model_params, my_y, beta = 1.0):

        comm = self.comm

        tiny = np.finfo(np.float64).tiny

        W = model_params['W'].T

        D         = self.D
        H         = self.H
        my_N,_    = my_y.shape
        N         = comm.allreduce(my_N)

        
        log_prob_y = np.zeros((my_N, H))

        for h in range(H):

            if self.sigmas_sq_type == 'full':
                sigma_h = model_params['sigmas_sq'][h]
                sigma_inv = np.linalg.inv(sigma_h)
                log_sigma_det = np.linalg.slogdet(sigma_h)[1]
            elif self.sigmas_sq_type == 'diagonal':
                sigma_h = model_params['sigmas_sq'][h]
                sigma_inv = 1./sigma_h
                log_sigma_det = np.sum(np.log(sigma_h))
                log_sigma_inv = -np.log(sigma_h) 

            batch_size = 100
            y_ind = 0
            while y_ind < my_N:
                if y_ind + batch_size > my_N:
                    batch_size = my_N - y_ind 

                y_norm = my_y[y_ind:y_ind+batch_size] - W[h]

                if self.sigmas_sq_type == 'full':
                    log_prob_y[y_ind:y_ind+batch_size, h] = -(log_sigma_det + np.sum(np.array(y_norm * np.matrix(sigma_inv)) * y_norm,1)) * beta
                elif self.sigmas_sq_type == 'diagonal':
                    temp = np.exp(2*np.log(np.abs(y_norm) + tiny) + log_sigma_inv[None,:])
                    log_prob_y[y_ind:y_ind+batch_size, h] = -(log_sigma_det + np.sum(temp,1)) * beta

                y_ind += batch_size
            


        return log_prob_y

   
    
        
