# -*- coding: utf-8 -*-
#
#  Author:   Abdul-Saboor Sheikh <sheikh@tu-berlin.de)
#  Lincense: Academic Free License (AFL) v3.0
#
# Reference: Sheikh, A. S., Shelton, J. A., & Luecke, J. 
# A Truncated EM Approach for Spike-and-Slab Sparse 
# Coding. JMLR (accepted, 2014).

from __future__ import division

import numpy as np
from math import pi
from mpi4py import MPI

try:
    from scipy import comb
except ImportError:
    from scipy.misc import comb

import prosper.em as em
import prosper.utils.parallel as parallel
import prosper.utils.tracing as tracing

from prosper.utils.datalog import dlog
from prosper.em.camodels import CAModel
from scipy import stats
from tables import openFile


class GSC(CAModel):
    def __init__(self, D, H, Hprime = 0, gamma = 0, sigma_sq_type = 'scalar', to_learn=['W', 'pi', 'mu','sigma_sq', 'psi_sq' ], comm=MPI.COMM_WORLD):
        CAModel.__init__(self, D, H, Hprime, gamma, to_learn, comm)

        # Noise Policy
        tol = 1e-5
        self.noise_policy = {
            'W'    : (-np.inf, +np.inf, False ),
            'pi'   : (    tol,  1.-tol, False ),
            'sigma_sq': (     0., +np.inf, False ),
            'mu'    : (-np.inf, +np.inf, False ),
            'psi_sq': (     0., +np.inf, False )
        }
        
        if gamma <= 0 or gamma > H:
            self.gamma = self.H
        if Hprime <= 0 or Hprime > H:
            self.Hprime = self.H
        elif Hprime < gamma:
            self.gamma = self.Hprime


        self.sigma_sq_type = sigma_sq_type
        
        self.dtype_precision = np.float64
        

    @tracing.traced
    def standard_init(self, my_data):
        """ Standard Initial of the model parameters.

        """
        comm = self.comm
                
        temp_params = CAModel.standard_init(self, my_data)
        model_params = {}
        model_params['W'] = temp_params['W'].copy()

        # Initial pi
        pi = comm.bcast(np.random.rand(self.H)) * 0.95
        pi[pi<0.05] = 0.05

        model_params['pi'] = pi

        my_y = my_data['y']

        # Calculate averarge W
        W_mean = parallel.allmean(my_y, axis=0, comm=comm)               # shape: (D, )

        # Calculate data variance
        sigma_sq_sq = parallel.allmean((my_y-W_mean)**2, axis=0, comm=comm) # shape: (D, )
        
        # Calculate sigma_sq
        if self.sigma_sq_type == 'full':
            model_params['sigma_sq'] =  np.diag(np.diag(sigma_sq_sq)) + (0.001 * np.eye(self.D)) 
        elif self.sigma_sq_type == 'diagonal':
            model_params['sigma_sq'] =  sigma_sq_sq + 0.001
        else:
            model_params['sigma_sq'] =  np.mean(sigma_sq_sq) + 0.001

     
        if 'mu' in self.to_learn:
            mu =  comm.bcast( np.random.normal(0,1,[self.H]) )
        else:
            mu =  np.zeros(self.H)

        model_params['mu'] = mu

        if 'psi_sq' in self.to_learn:
            psi_sq_diag = comm.bcast(np.random.rand(self.H)) * 2
            psi_sq_diag[psi_sq_diag<0.05] = 0.05
            psi_sq = np.diag(psi_sq_diag)
        else:
            psi_sq = np.eye(self.H)

        model_params['psi_sq'] = psi_sq

        model_params = comm.bcast(model_params)       

        return model_params

    @tracing.traced
    def resume_init(self, h5_result_file):
        """ Initialize model parameters to previously inferred values.

        """
        comm = self.comm
        
        h5 = openFile(h5_output, 'r')
        # General information
        steps, H, D = h5.root.W_T.shape

        W = h5.root.W[steps-1]

        pi = h5.root.pi[steps-1]

        mu = h5.root.mu[steps-1]

        psi_sq = h5.root.psi_sq[steps-1]
        
        sigma_sq = h5.root.sigma_sq[steps-1]
        
        if len(h5.root.sigma_sq.shape) == 3:
            if self.sigma_sq_type == 'diagonal':
                sigma_sq = sigma_sq.diagonal()
            elif self.sigma_sq_type == 'scalar':
                sigma_sq = np.mean(sigma_sq.diagonal())
        elif len(h5.root.sigma_sq.shape) == 2:
            if self.sigma_sq_type == 'full':
                sigma_sq = np.diag(sigma_sq)
            elif self.sigma_sq_type == 'scalar':
                sigma_sq = np.mean(sigma_sq)
        else:
            if self.sigma_sq_type == 'full':
                sigma_sq = sigma_sq * np.eye(D)
            elif self.sigma_sq_type == 'diagonal':
                sigma_sq = sigma_sq * np.ones(D)
            
        model_params = {
            'W'       : W, 
            'pi'      : pi,
            'sigma_sq' : sigma_sq,
            'mu'    : mu,
            'psi_sq' : psi_sq
        }

        model_params = comm.bcast(model_params)   

        return model_params

    @tracing.traced
    def check_params(self, model_params):
        """ Sanity check.

        Sanity-check the given model parameters. Raises an exception if 
        something is severely wrong.
        """
        comm = self.comm

        if comm.rank == 0:       

            assert np.isfinite(model_params['W']).all()      # check W

            assert np.isfinite(model_params['mu']).all()   # check mu

            assert np.isfinite(model_params['pi']).all()   # check pi
            
            assert np.isfinite(model_params['psi_sq']).all()  # check sigma_sq
           

            if self.sigma_sq_type == 'full':
                assert np.isfinite(model_params['sigma_sq']).all()  # check sigma_sq
                assert np.sum(model_params['sigma_sq'][range(self.D),range(self.D)] <= 0) == 0
            elif self.sigma_sq_type == 'diagonal':
                assert np.isfinite(model_params['sigma_sq']).all()  # check sigma_sq
                assert np.sum(model_params['sigma_sq'] <= 0) == 0
            else:# 'scalar'
                assert np.isfinite(model_params['sigma_sq'])  # check sigma_sq
                assert model_params['sigma_sq'] > 0

             
        return model_params

    @tracing.traced
    def generate_data(self, model_params, my_N):
        """ 
            given ground truth model parameters, generate data of size my_N
        """

        H = self.H
        
        # Create ground-truth s
        s = np.zeros( (my_N, H), dtype=np.bool )

        for n in xrange(my_N):
            
            p = np.random.random(H)        # create a latent vector
            s[n] = p <= model_params['pi']                # translate into a boolean latent vector


        # return the generated data
        return self.generate_from_hidden(model_params, {'s': s})

    @tracing.traced
    def generate_from_hidden(self, model_params, my_hdata):
        """ Generate data according to the MCA model while the latents are 
        given in my_hdata['s'].

        This method does _not_ obey gamma: The generated data may have more
        than gamma active causes for a given datapoint.
        """ 
        
        D = self.D
        H = self.H
        
        # Create output arrays, y is data, s and z are ground-truth hiddens
        s = my_hdata['s']
        my_N, _ = s.shape
        y = np.zeros( (my_N, D) )
        z = np.zeros( (my_N, H) )

        if self.sigma_sq_type == 'full':
            sigma_sq = np.sqrt(model_params['sigma_sq'].diagonal())
        elif self.sigma_sq_type == 'diagonal':
            sigma_sq = np.sqrt(model_params['sigma_sq'])
        else:# 'scalar'
            sigma_sq = np.sqrt(model_params['sigma_sq']) * np.ones(D)

        for n in xrange(my_N):
            
            # active component indexes 
            actv_comps_inds = np.nonzero(s[n])[0]

            if np.sum(actv_comps_inds) == 0:
                continue
            Ws = np.matrix(model_params['W'][:,actv_comps_inds])
            
            z_n = np.random.multivariate_normal(model_params['mu'][actv_comps_inds],(model_params['psi_sq'][actv_comps_inds,:])[:,actv_comps_inds],1).flatten()
            
            z[n,actv_comps_inds] = z_n

            y[n] = np.array(Ws * z_n[:,None]).flatten() + sigma_sq * np.random.randn(D)
   


        # return the generated data
        return { 'y': y, 's': s, 'z': z }

    @tracing.traced
    def compute_posterior_hprime(self, anneal, model_params, my_data):

        comm = self.comm
        
        beta     = 1./anneal['T']

        D = self.D
        H         = self.H

        sigma_sq_inv = model_params['sigma_sq_inv']
        B = model_params['B']

        y_l_ind = 0
        
        cur_y = my_data['y']
        
        if len(cur_y.shape) == 1:
            if cur_y.shape[0] == D:
                cur_y = cur_y[None,:]
            else:
                cur_y = cur_y[:,None]
        cur_N, _ = cur_y.shape
        
        tiny = np.finfo(np.float64).tiny
        
        cur_comps = my_data['candidates']
        cur_W = np.array(model_params['W'][:,cur_comps])
        cur_mu = np.array(model_params['mu'][cur_comps])
        cur_psi_sq = (model_params['psi_sq'][cur_comps,:])[:,cur_comps]
        cur_y = my_data['y']
        
        log_pi_pr = np.log(model_params['pi']) - np.log(1 - np.array(model_params['pi']))

        # to accumulate posterior probabilities per data point in the current data cluster
        post_nfac_n = np.zeros([cur_N], dtype=self.dtype_precision)

        pstr_s = np.zeros([cur_N, H], dtype=self.dtype_precision)
        pstr_ss = np.zeros([cur_N, H, H], dtype=self.dtype_precision)
        pstr_sz = np.zeros([cur_N, H], dtype=self.dtype_precision)
        pstr_szsz = np.zeros([cur_N, H, H], dtype=self.dtype_precision)

        #Loop over all the non-singleton states in Hprime space with minimum 2 and maximum gamma active causes#
        # for all Hprime choose k in [2,..,gamma]
        
        for state_ind in range(self.no_states):
            cur_state = self.state_matrix[state_ind]
            ind_comps = cur_state > 0
            # active component indexes in the current H_prime dimensional state
            actv_comps_inds = np.nonzero(ind_comps)[0]
            # actual component indexes in H-dimensional hidden space
            actual_comps_inds = cur_comps[actv_comps_inds]

            num_act_comps = np.sum(ind_comps)
            if num_act_comps == 0:
                continue

            W_s = np.matrix(cur_W[:,actv_comps_inds])
            psi_sq_s = (cur_psi_sq[actv_comps_inds,:])[:,actv_comps_inds]
            mu_s = np.array(cur_mu[actv_comps_inds])

            if self.sigma_sq_type == 'full' or self.sigma_sq_type == 'scalar':
                sigma_sq_inv_W_s = sigma_sq_inv * W_s
            elif self.sigma_sq_type == 'diagonal':
                sigma_sq_inv_W_s = sigma_sq_inv[:,None] * np.array(W_s)

            lambda_s = sigma_sq_inv_W_s.T * W_s + np.linalg.inv(psi_sq_s)
            lambda_s_inv = np.linalg.inv(lambda_s)

            if self.sigma_sq_type == 'full' or self.sigma_sq_type == 'scalar':
                lambda_s_inv_W_s = (lambda_s_inv * W_s.T) * sigma_sq_inv
            elif self.sigma_sq_type == 'diagonal':
                lambda_s_inv_W_s = np.matrix(np.array(lambda_s_inv * W_s.T) * sigma_sq_inv[None,:])
            
            # expectation value of the continuous latent z given s,y and the model parameters
            kappa_s_n =  (lambda_s_inv_W_s * np.array(cur_y - np.dot(W_s, mu_s)).T).T
            kappa_s_n += mu_s
            kappa_s_n = np.array(kappa_s_n)

            # expectation value of z^2 given s,y and the model parameters
            kappa_sq_s_n = (kappa_s_n[:,None,:].T*kappa_s_n[:,:,None].T).T + np.array(lambda_s_inv)[None,:,:]

            # posterior probability of y_n given s and model params the current state (in log space)
            C_inv = B - sigma_sq_inv_W_s * lambda_s_inv_W_s
            C_det = np.linalg.slogdet(psi_sq_s)[1] + np.linalg.slogdet(lambda_s)[1]
            norm_const = -C_det
            cur_y_norm = np.array(cur_y - np.dot(W_s, mu_s))
            post_s = (norm_const  - np.sum(np.array(cur_y_norm * np.matrix(C_inv)) * cur_y_norm,1)) 
            
            on_comps_ind = np.zeros(self.H, dtype=np.int)
            on_comps_ind[actual_comps_inds] = 1
            
            prob_s = np.sum(log_pi_pr[actual_comps_inds])
            
            # convert the log-probability to actual probability
            post_s = np.exp((post_s + prob_s) * beta) 
            post_s[np.isnan(post_s)] = tiny
            post_s[post_s < tiny] = tiny
            
            post_nfac_n += post_s

            kappa_s_n = np.array(kappa_s_n, dtype=self.dtype_precision)
            kappa_s_n *= post_s[:,None]

            kappa_sq_s_n = np.array(kappa_sq_s_n, dtype=self.dtype_precision)                
            kappa_sq_s_n *= post_s[:,None,None]

            s_temp = np.zeros((cur_N, self.H), dtype=self.dtype_precision)
            s_temp[:,actual_comps_inds] = np.tile(post_s[:,None],(1,np.sum(on_comps_ind)))
            pstr_s += s_temp

            # compute one dimensional indexes of the collapsed active components by active components second moments ss and szsz
            actual_comps_inds_rshp = np.array(np.tile((actual_comps_inds*self.H)[:,None],actual_comps_inds.size) + actual_comps_inds[None,:]).reshape(actual_comps_inds.size**2)

            ss = np.ones(kappa_sq_s_n.shape, dtype=self.dtype_precision) * post_s[:,None,None]
            ss_temp = np.zeros((cur_N,self.H**2), dtype=self.dtype_precision)

            # reshape (cur_N,active components,active components) ss to (cur_N,active components^2)
            ss_rshp =  ss.reshape(cur_N, actual_comps_inds.size**2)
            ss_temp[:,actual_comps_inds_rshp] = ss_rshp 
            ss_temp = ss_temp.reshape(cur_N, self.H, self.H)
            pstr_ss += ss_temp

            kappa_s_n_temp = np.zeros((cur_N, self.H), dtype=self.dtype_precision)
            kappa_s_n_temp[:,actual_comps_inds] = kappa_s_n 
     
            pstr_sz += kappa_s_n_temp
     
            kappa_sq_s_n_temp = np.zeros((cur_N,self.H**2), dtype=self.dtype_precision)

            # reshape (cur_N,active components,active components) kappa_sq_s_n to (cur_N,active components^2)
            kappa_sq_s_n_rshp =  kappa_sq_s_n.reshape(cur_N, actual_comps_inds.size**2)
            kappa_sq_s_n_temp[:,actual_comps_inds_rshp] = kappa_sq_s_n_rshp 
            kappa_sq_s_n_temp = kappa_sq_s_n_temp.reshape(cur_N, self.H, self.H)
            pstr_szsz += kappa_sq_s_n_temp
        
       
        suff_stats = { 'pstr_s': pstr_s, 'pstr_ss': pstr_ss, 'pstr_sz': pstr_sz, 'pstr_szsz': pstr_szsz, 'post_nfac_n': post_nfac_n }

        return suff_stats

    @tracing.traced
    def E_step(self, anneal, model_params, my_data):
           
        comm = self.comm
        
        data_clusters = my_data['data_clusters']
    
        beta     = 1./anneal['T']

        D = self.D
        H         = self.H
        my_N      = 0
        for cluster_key in data_clusters.iterkeys():
            data_cluster = data_clusters[cluster_key]
            my_N += data_cluster['data'].shape[0]

        if self.sigma_sq_type == 'full':
            sigma_sq_inv = np.linalg.inv(model_params['sigma_sq'])
            B = sigma_sq_inv 
        elif self.sigma_sq_type == 'diagonal':
            sigma_sq_inv = 1./model_params['sigma_sq']
            B = np.diag(sigma_sq_inv)
        else:# 'scalar'
            sigma_sq_inv = 1./model_params['sigma_sq']
            B = sigma_sq_inv * np.eye(self.D)

        model_params['sigma_sq_inv'] = sigma_sq_inv
        model_params['B'] = B

        y_l_ind = 0
        

        my_y = np.zeros([my_N, D])
        my_y_cands = np.zeros([my_N, self.Hprime])
        # sufficient statistics to be returned
        xpt_s = np.empty([my_N, H])
        xpt_ss = np.empty([my_N, H, H])
        xpt_sz = np.empty([my_N, H])
        xpt_szsz = np.empty([my_N, H, H])

        
        tiny = np.finfo(np.float64).tiny
        
        
        for cluster_key in data_clusters.iterkeys():
            cur_cluster = data_clusters[cluster_key]
            cur_comps = cur_cluster['hprimes']
            cur_W = np.array(model_params['W'][:,cur_comps])
            cur_mu = np.array(model_params['mu'][cur_comps])
            cur_psi_sq = (model_params['psi_sq'][cur_comps,:])[:,cur_comps]
            cur_y = cur_cluster['data']
            y_inds = np.arange(y_l_ind,y_l_ind+cur_y.shape[0])
            y_l_ind += cur_y.shape[0]
            my_y[y_inds] = cur_y
            my_y_cands[y_inds] = cur_comps[None,:]
                        
            cur_N, D = cur_y.shape

            log_pi_pr = np.log(model_params['pi']) - np.log(1 - np.array(model_params['pi']))

            # to accumulate posterior probabilities per data point in the current data cluster
            post_nfac_n = np.zeros([cur_N], dtype=self.dtype_precision)

            pstr_s = np.zeros([cur_N, H], dtype=self.dtype_precision)
            pstr_ss = np.zeros([cur_N, H, H], dtype=self.dtype_precision)
            pstr_sz = np.zeros([cur_N, H], dtype=self.dtype_precision)
            pstr_szsz = np.zeros([cur_N, H, H], dtype=self.dtype_precision)

            ################################## Calc prob of null state #########################################

            post_s = - np.sum(np.array(self.dtype_precision(cur_y) * np.matrix(B)) * cur_y,1)
            
            post_s = np.exp(post_s * beta) 

            post_nfac_n += post_s

            ############################# Loop through all the singleton states ################################
            for cur_h in range(self.H):
            
                W_s = np.matrix(model_params['W'][:,cur_h])
                mu_s = model_params['mu'][cur_h]
                psi_sq_s = model_params['psi_sq'][cur_h,cur_h]
                
                num_act_comps = 1
                
                if self.sigma_sq_type == 'full' or self.sigma_sq_type == 'scalar':
                    sigma_sq_inv_W_s = np.dot(W_s , sigma_sq_inv)
                elif self.sigma_sq_type == 'diagonal':
                    sigma_sq_inv_W_s = np.matrix(sigma_sq_inv * np.array(W_s))

                lambda_s = np.dot(sigma_sq_inv_W_s, W_s.T) + 1/psi_sq_s 
                lambda_s_inv = 1./lambda_s

                if self.sigma_sq_type == 'full' or self.sigma_sq_type == 'scalar':
                    lambda_s_inv_W_s = np.dot(lambda_s_inv * W_s, sigma_sq_inv)
                elif self.sigma_sq_type == 'diagonal':
                    lambda_s_inv_W_s = np.matrix(np.array(lambda_s_inv * W_s) * sigma_sq_inv)
                               
                # expectation value of the continuous latent z given s,y and the model parameters
                kappa_s_n =  np.dot(lambda_s_inv_W_s, (cur_y - np.dot(W_s, mu_s)).T)
                kappa_s_n += mu_s
                kappa_s_n = np.array(kappa_s_n.T)
                
                # expectation value of z^2 given s,y and the model parameters
                kappa_sq_s_n = kappa_s_n**2 + np.array(lambda_s_inv)

                # compute the posterior (in log space)
                C_inv = B - sigma_sq_inv_W_s.T * lambda_s_inv_W_s
                C_det = np.log(psi_sq_s) + np.linalg.slogdet(lambda_s)[1]
                norm_const = -C_det
                cur_y_norm = np.array(cur_y- (mu_s * W_s))
                post_s = (norm_const  - np.sum(np.array(cur_y_norm * np.matrix(C_inv)) * cur_y_norm,1)) 

                on_comps_ind = np.zeros(H, dtype=np.int)
                on_comps_ind[cur_h] = 1
           
                prob_s = np.sum(log_pi_pr[cur_h])
           
                post_s = np.exp((post_s + prob_s) * beta) 

                # convert the log-probability to actual probability   
                post_s[np.isnan(post_s)] = tiny
                post_s[post_s < tiny] = tiny
                post_nfac_n += post_s

                kappa_s_n = np.array(kappa_s_n, dtype=self.dtype_precision)
                kappa_s_n *= post_s[:,None]

                kappa_sq_s_n = np.array(kappa_sq_s_n, dtype=self.dtype_precision)
                kappa_sq_s_n *= post_s[:,None]

                s_temp = np.zeros((cur_N, H), dtype=self.dtype_precision)
                s_temp[:,cur_h] = post_s
                pstr_s += s_temp

                ss_temp = np.zeros((cur_N, H, H), dtype=self.dtype_precision)
                ss_temp[:,cur_h,cur_h] = post_s 
                pstr_ss += ss_temp

                kappa_s_n_temp = np.zeros((cur_N, H), dtype=self.dtype_precision)
                kappa_s_n_temp[:,cur_h] = kappa_s_n.flatten()
                pstr_sz += kappa_s_n_temp
         
                kappa_sq_s_n_temp = np.zeros((cur_N, H, H), dtype=self.dtype_precision)
                kappa_sq_s_n_temp[:,cur_h,cur_h] = kappa_sq_s_n.flatten()
                pstr_szsz += kappa_sq_s_n_temp

            
            ################### Now compute posterior over all the non-singleton states ########################
            
            cur_suff_stats = self.compute_posterior_hprime(anneal, model_params, {'y': cur_y ,'candidates': cur_comps})

            pstr_s += cur_suff_stats['pstr_s']

            pstr_ss += cur_suff_stats['pstr_ss']
         
            pstr_sz += cur_suff_stats['pstr_sz']
         
            pstr_szsz += cur_suff_stats['pstr_szsz']

            post_nfac_n += cur_suff_stats['post_nfac_n']
            
            ####################################################################################################
   
            post_nfac_n = 1.0/np.array(post_nfac_n + np.finfo(np.float64).tiny)
            
            xpt_s[y_inds] = pstr_s * post_nfac_n [:,None]           # shape: ., H
            xpt_ss[y_inds] = pstr_ss * post_nfac_n [:,None,None] # shape: ., H, H
            xpt_sz[y_inds] = pstr_sz * post_nfac_n [:,None]       # shape: ., H
            xpt_szsz[y_inds] = pstr_szsz * post_nfac_n [:,None,None] # shape: ., H, H


        my_data['y'] = my_y
        my_data['candidates'] = my_y_cands

        del model_params['sigma_sq_inv']
        del model_params['B']

        suff_stats = { 'xpt_s': xpt_s, 'xpt_ss': xpt_ss, 'xpt_sz': xpt_sz, 'xpt_szsz': xpt_szsz }

        return suff_stats
        

    @tracing.traced
    def M_step(self, anneal, model_params, suff_stats, my_data):
        comm = self.comm        
 
        my_xpt_sz = np.array(suff_stats['xpt_sz'], dtype=np.float64)
        my_y = my_data['y']

        _, H = my_xpt_sz.shape
        my_N, D = my_y.shape
        N = comm.allreduce(my_N) 

        eps = 1e-5       
        
        # to store accumulated sufficient statistics from all processes 
        sum_xpt_s = np.empty( (H,) , dtype=np.float64)
        sum_xpt_sz = np.empty( (H,) , dtype=np.float64)
        sum_xpt_szsz = np.empty( (H, H) , dtype=np.float64)
        sum_xpt_ss = np.empty( (H, H) , dtype=np.float64)

        # collect sufficient statistics of this process
        my_xpt_szsz = np.array(suff_stats['xpt_szsz'], dtype=np.float64)
        my_xpt_s = np.array(suff_stats['xpt_s'], dtype=np.float64)
        my_xpt_ss = np.array(suff_stats['xpt_ss'], dtype=np.float64)

        # now collect sufficient statistics from all processes
        comm.Allreduce( [my_xpt_s.sum(axis=0), MPI.DOUBLE], [sum_xpt_s, MPI.DOUBLE] )
        comm.Allreduce( [my_xpt_sz.sum(axis=0), MPI.DOUBLE], [sum_xpt_sz, MPI.DOUBLE] )
        comm.Allreduce( [my_xpt_szsz.sum(axis=0), MPI.DOUBLE], [sum_xpt_szsz, MPI.DOUBLE] )
            
        
        my_Wp = np.zeros((D,H))
        for ind_n in range(my_N):
            obs_space = np.arange(D)
            my_Wp[obs_space,:] += my_xpt_sz[ind_n][None,:] * my_y[ind_n,obs_space][:,None]

        Wp = np.empty_like(my_Wp)
     
        comm.Allreduce( [my_Wp, MPI.DOUBLE], [Wp,MPI.DOUBLE])        
        
                    
        # first we update the W matrix
        try:
            sum_xpt_szsz_inv = np.linalg.inv(sum_xpt_szsz)
            W_n = np.dot(Wp, sum_xpt_szsz_inv)
            #W_n  = np.linalg.lstsq(sum_xpt_szsz, Wp)[0]
        except np.linalg.linalg.LinAlgError:
            # if Sum of the expected values of the second moments was not invertable. Adding some noise and taking pinv.
            try:
                noise = np.random.normal(0,eps,self.H)
                noise = np.outer(noise,noise)
                sum_xpt_szsz_inv = np.linalg.pinv(sum_xpt_szsz + noise)
                W_n = np.dot(Wp, sum_xpt_szsz_inv)
            except np.linalg.linalg.LinAlgError:
                # Sum of the expected values of the second moments was not invertable. Skip the update of parameter W but
                # add some noise to it.
                W_n = model_params['W'] + (eps * np.random.normal(0,1,[self.D,self.H]))    

        if 'pi' in self.to_learn:
            pi_eps = 5e-5
            pi_new = sum_xpt_s/N
            pi_new[pi_new <= pi_eps] = pi_eps
            pi_new[pi_new >= (1 - pi_eps)] = 1- pi_eps
            model_params['pi'] = pi_new
      
      

        if 'W' in self.to_learn:
            model_params['W'] = W_n
                  

        if 'mu' in self.to_learn:
            model_params['mu'] = sum_xpt_sz * 1./(sum_xpt_s + np.finfo(np.float64).eps)
      
        
        if 'psi_sq' in self.to_learn:
            mu = model_params['mu']

            my_psi_sq = np.zeros((self.H,self.H))
            mu_outer =  np.outer(mu,mu)
            for ind in range(my_N):
                my_psi_sq += (mu_outer * my_xpt_ss[ind])
                my_psi_sq += my_xpt_szsz[ind]
                my_psi_sq -= 2*np.outer(mu*my_xpt_s[ind], my_xpt_sz[ind])                

            psi_sq = np.empty( (self.H,self.H) )
            comm.Allreduce( [my_psi_sq, MPI.DOUBLE], [psi_sq, MPI.DOUBLE] )

            sum_xpt_ss = np.empty_like(sum_xpt_szsz)
            comm.Allreduce( [my_xpt_ss.sum(axis=0), MPI.DOUBLE], [sum_xpt_ss , MPI.DOUBLE] )

            model_params['psi_sq'] = (np.array(psi_sq) * np.linalg.inv(sum_xpt_ss + eps * np.eye(self.H))) + (eps * np.eye(self.H))
            
        if 'sigma_sq' in self.to_learn:

            if self.sigma_sq_type == 'full':

                my_sigma_sq =  np.zeros((D,D))
                for ind in range(my_N):
            
                    my_sigma_sq += np.outer(my_y[ind],my_y[ind]) 
                    my_sigma_sq -= np.dot(W_n,np.dot(np.outer(my_xpt_sz[ind],my_xpt_sz[ind]), W_n.T) )
                    

                my_sigma_sq /= N 
                sigma_sq = np.empty( (D,D) )
                comm.Allreduce( [my_sigma_sq, MPI.DOUBLE], [sigma_sq, MPI.DOUBLE] )
                sigma_sq = np.array(sigma_sq) + (eps * np.eye(self.D))
        
            elif self.sigma_sq_type == 'diagonal':

                my_sigma_sq =  np.zeros(D)
                for ind in range(my_N):
          
                    my_sigma_sq += my_y[ind]**2
                    my_sigma_sq -= np.dot(W_n, my_xpt_sz[ind].T)**2

                          
                sigma_sq = np.empty(D)
                comm.Allreduce( [my_sigma_sq, MPI.DOUBLE], [sigma_sq, MPI.DOUBLE] )

                sigma_sq = sigma_sq / N + eps 

            elif self.sigma_sq_type == 'scalar':
               
                my_sigma_sq = 0.0
                WT_outer = np.dot(W_n.T,W_n)
                for ind in range(my_N):
                    my_sigma_sq += np.dot(my_y[ind],my_y[ind])
                    my_sigma_sq -= np.trace(np.dot(np.outer(my_xpt_sz[ind],my_xpt_sz[ind]),WT_outer))
                    
                sigma_sq = (comm.allreduce(my_sigma_sq) / N / D) + eps 
        
            model_params['sigma_sq'] = sigma_sq
            

        return model_params

    @tracing.traced
    def select_Hprimes(self, model_params, my_data):

        comm = self.comm

        my_y = my_data['y']
        comp_scores = np.zeros((my_y.shape[0], self.H))
        
        my_N, D = my_y.shape
        N = comm.allreduce(my_N) 
        
        
        comp_scores = self.component_scores(model_params, my_data)
        cand_comps = np.argsort(comp_scores, axis = 1)[:,-self.Hprime:]
        strd_cand_comps = np.sort(cand_comps, axis = 1)
        
        
        data_clusters = {}
        for ind in range(cand_comps.shape[0]):
            cluster_key = np.str(strd_cand_comps[ind,:]) 
            if cluster_key in data_clusters:
                data_cluster = data_clusters[cluster_key]
                data_cluster['data'] = np.append(data_cluster['data'],my_y[ind].reshape((1,D)), axis=0)
            else:
                data_cluster = {}
                data_cluster['hprimes'] = strd_cand_comps[ind,:]
                data_cluster['data'] = my_y[ind].reshape((1,D))
            data_clusters[cluster_key] = data_cluster


        my_data['data_clusters'] = data_clusters

        return my_data

    @tracing.traced
    def component_scores(self, model_params, my_data):
        comm = self.comm

        my_y = my_data['y']

        comp_fac_probs = np.zeros((my_y.shape[0], self.H))
        
        my_N, D = my_y.shape
        N = comm.allreduce(my_N) 

        log_tiny = np.finfo(np.float64).min

        if self.sigma_sq_type == 'full':
            sigma_sq_inv = np.linalg.inv(model_params['sigma_sq'])
            B = sigma_sq_inv
        elif self.sigma_sq_type == 'diagonal':
            sigma_sq_inv = 1./model_params['sigma_sq']
            B = np.diag(sigma_sq_inv)
        else:# 'scalar'
            sigma_sq_inv = 1./model_params['sigma_sq']
            B = sigma_sq_inv * np.eye(self.D)

        W = model_params['W'].copy()
        
        for cur_h in range(self.H):

            W_s = np.matrix(W[:,cur_h])
            mu_s = model_params['mu'][cur_h]
            psi_sq_s = model_params['psi_sq'][cur_h,cur_h]
            
            num_act_comps = 1


            if self.sigma_sq_type == 'full' or self.sigma_sq_type == 'scalar':
                sigma_sq_inv_W_s = np.dot(W_s , sigma_sq_inv)
            elif self.sigma_sq_type == 'diagonal':
                sigma_sq_inv_W_s = np.matrix(sigma_sq_inv * np.array(W_s))

            M_s = np.dot(sigma_sq_inv_W_s, W_s.T) + 1/psi_sq_s 
            M_s_inv = 1./M_s

            if self.sigma_sq_type == 'full' or self.sigma_sq_type == 'scalar':
                M_s_inv_W_s = np.dot(M_s_inv * W_s, sigma_sq_inv)
            elif self.sigma_sq_type == 'diagonal':
                M_s_inv_W_s = np.matrix(np.array(M_s_inv * W_s) * sigma_sq_inv)
                           
            C_inv = B - sigma_sq_inv_W_s.T * M_s_inv_W_s
            C_det = np.log(psi_sq_s) + np.linalg.slogdet(M_s)[1]
            norm_const = -C_det
            cur_y_norm = np.array(my_y- (mu_s * W_s))
            post_s = (norm_const  - np.sum(np.array(cur_y_norm * np.matrix(C_inv)) * cur_y_norm,1))
            
            post_s[np.isnan(post_s)] = log_tiny
            post_s[post_s < log_tiny] = log_tiny
            post_s[np.isinf(post_s)] = 0

            
            comp_fac_probs[:,cur_h] = post_s

        return comp_fac_probs


    

