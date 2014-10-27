#
#  Author:   Marc Henniges
#        and Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

import numpy as np
from math import pi
from mpi4py import MPI

try:
    from scipy import comb
except ImportError:
    from scipy.misc import comb

import pulp.em as em
import pulp.utils.parallel as parallel
import pulp.utils.tracing as tracing

from pulp.utils.datalog import dlog
from pulp.em.camodels import CAModel

class BSC_ET(CAModel):
    def __init__(self, D, H, Hprime, gamma, to_learn=['W', 'pi', 'sigma'], comm=MPI.COMM_WORLD):
        CAModel.__init__(self, D, H, Hprime, gamma, to_learn, comm)

    @tracing.traced
    def generate_from_hidden(self, model_params, my_hdata):
        """ Generate data according to the MCA model while the latents are 
        given in my_hdata['s'].

        This method does _not_ obey gamma: The generated data may have more
        than gamma active causes for a given datapoint.
        """
        W     = model_params['W'].T
        pies  = model_params['pi']
        sigma = model_params['sigma']
        H, D  = W.shape

        s       = my_hdata['s']
        my_N, _ = s.shape
        
        # Create output arrays, y is data
        y = np.zeros( (my_N, D) )

        for n in xrange(my_N):
            # Combine accoring do magnitude-max rulew
            for h in xrange(H):
                if s[n,h]:
                    y[n] += W[h]

        # Add noise according to the model parameters
        y += np.random.normal( scale=sigma, size=(my_N, D) )

        # Build return structure
        return { 'y': y, 's': s }

    @tracing.traced
    def select_Hprimes(self, model_params, data):
        """
        Return a new data-dictionary which has been annotated with
        a data['candidates'] dataset. A set of self.Hprime candidates
        will be selected.
        """
        my_y = data['y']
        W = model_params['W'].T
        Hprime = self.Hprime
        my_N, D = my_y.shape

        candidates = np.zeros( (my_N, Hprime), dtype=np.int )
        for n in xrange(my_N):
            sim = np.inner(W,my_y[n])/ np.sqrt(np.diag(np.inner(W,W)))/ np.sqrt(np.inner(my_y[n],my_y[n]))
            candidates[n] = np.argsort(sim)[-Hprime:]
        
        data['candidates'] = candidates
        return data


    @tracing.traced
    def E_step(self, anneal, model_params, my_data):
        """ BSC E_step

        my_data variables used:
            
            my_data['y']           Datapoints
            my_data['can']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """
        comm      = self.comm
        my_y      = my_data['y'].copy()
        my_cand   = my_data['candidates']
        my_N, D   = my_data['y'].shape
        H = self.H

        SM = self.state_matrix        # shape: (no_states, Hprime)
        state_abs = self.state_abs           # shape: (no_states,)

        W         = model_params['W'].T
        pies      = model_params['pi']
        sigma     = model_params['sigma']
        try:
            mu = model_params['mu']
        except:
            mu = np.zeros(D)
            model_params['mu'] = mu

        # Precompute 
        beta     = 1./anneal['T']
        pre1     = -1./2./sigma/sigma
        pil_bar  = np.log( pies/(1.-pies) )

        # Allocate return structures
        F = np.empty( [my_N, 1+H+self.no_states] )
        pre_F = np.empty( [my_N, 1+H+ self.no_states] )
        denoms = np.zeros(my_N)
        
        # Pre-fill pre_F:
        pre_F[:,0] = 0.
        pre_F[:,1:H+1] = pil_bar
        pre_F[:,1+H:] = pil_bar * state_abs   # is (no_states,)
        
        # Iterate over all datapoints
        tracing.tracepoint("E_step:iterating")
        for n in xrange(my_N):
            y    = my_data['y'][n,:] - mu
            cand = my_data['candidates'][n,:]

            # Zero active hidden causes
            log_prod_joint = pre1 * (y**2).sum()
            F[n,0] = log_prod_joint

            # Hidden states with one active cause
            log_prod_joint = pre1 * ((W-y)**2).sum(axis=1)
            F[n,1:H+1] = log_prod_joint

            # Handle hidden states with more than 1 active cause
            W_ = W[cand]                          # is (Hprime x D)

            Wbar = np.dot(SM,W_)
            log_prod_joint = pre1 * ((Wbar-y)**2).sum(axis=1)
            F[n,1+H:] = log_prod_joint

        if anneal['anneal_prior']:
            F = beta * (pre_F + F)
        else:
            F = pre_F + beta * F

        return { 'logpj': F } 

    @tracing.traced
    def M_step(self, anneal, model_params, my_suff_stat, my_data):
        """ BSC M_step

        my_data variables used:
            
            my_data['y']           Datapoints
            my_data['candidates']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """

        comm      = self.comm
        H, Hprime = self.H, self.Hprime
        gamma     = self.gamma
        W         = model_params['W'].T
        pies      = model_params['pi']
        sigma     = model_params['sigma']
        mu        = model_params['mu']

        # Read in data:
        my_y       = my_data['y'].copy()
        candidates = my_data['candidates']
        logpj_all  = my_suff_stat['logpj']
        all_denoms = np.exp(logpj_all).sum(axis=1)

        my_N, D   = my_y.shape
        N         = comm.allreduce(my_N)
        
        # Joerg's data noise idea
        data_noise_scale = anneal['data_noise']
        if data_noise_scale > 0:
            my_y += my_data['data_noise']

        SM = self.state_matrix        # shape: (no_states, Hprime)

        # To compute et_loglike:
        my_ldenom_sum = 0.0
        ldenom_sum = 0.0

        # Precompute factor for pi update
        A_pi_gamma = 0
        B_pi_gamma = 0
        for gamma_p in range(gamma+1):
            A_pi_gamma += comb(H,gamma_p) * (pies**gamma_p) * ((1-pies)**(H-gamma_p))
            B_pi_gamma += gamma_p * comb(H,gamma_p) * (pies**gamma_p) * ((1-pies)**(H-gamma_p))
        E_pi_gamma = pies * H * A_pi_gamma / B_pi_gamma
        
        # Truncate data
        if anneal['Ncut_factor'] > 0.0:
            tracing.tracepoint("M_step:truncating")
            #alpha = 0.9 # alpha from ET paper
            #N_use = int(alpha * (N * (1 - (1 - A_pi_gamma) * anneal['Ncut_factor'])))
            N_use = int(N * (1 - (1 - A_pi_gamma) * anneal['Ncut_factor']))
            cut_denom = parallel.allsort(all_denoms)[-N_use]
            which   = np.array(all_denoms >= cut_denom)
            candidates = candidates[which]
            logpj_all = logpj_all[which]
            my_y    = my_y[which]
            my_N, D = my_y.shape
            N_use = comm.allreduce(my_N)
        else:
            N_use = N
        dlog.append('N', N_use)
        
        # Calculate truncated Likelihood
        L = H * np.log(1-pies) - 0.5 * D * np.log(2*pi*sigma**2) - np.log(A_pi_gamma)
        Fs = np.log(np.exp(logpj_all).sum(axis=1)).sum()
        L += comm.allreduce(Fs)/N_use
        dlog.append('L',L)

        # Precompute
        pil_bar   = np.log( pies/(1.-pies) )
        corr_all  = logpj_all.max(axis=1)                 # shape: (my_N,)
        pjb_all   = np.exp(logpj_all - corr_all[:, None]) # shape: (my_N, no_states)

        # Allocate 
        my_Wp = np.zeros_like(W)   # shape (H, D)
        my_Wq = np.zeros((H,H))    # shape (H, H)
        my_pi = 0.0                #
        my_sigma = 0.0             #
        #my_mup = np.zeros_like(W)  # shape (H, D)
        #my_muq = np.zeros((H,H))   # shape (H, H)
        my_mus = np.zeros(H)       # shape D
        data_sum = my_y.sum(axis=0)   # sum over all data points for mu update
        
        ## Calculate mu
        #for n in xrange(my_N):
            #tracing.tracepoint("Calculationg offset")
            #y     = my_y[n,:]                  # length D
            #cand  = candidates[n,:] # length Hprime
            #logpj = logpj_all[n,:] # length no_states
            #corr  = corr_all[n]        # scalar
            #pjb = pjb_all[n, :]

            ## Zero active hidden cause (do nothing for the W and pi case) 
            ## this_Wp += 0.     # nothing to do
            ## this_Wq += 0.     # nothing to do
            ## this_pi += 0.     # nothing to do

            ## One active hidden cause
            #this_mup = np.outer(pjb[1:(H+1)],y)
            #this_muq = pjb[1:(H+1)] * np.identity(H)
            #this_mus = pjb[1:(H+1)]

            ## Handle hidden states with more than 1 active cause
            #this_mup[cand]      += np.dot(np.outer(y,pjb[(1+H):]),SM).T
            #this_muq_tmp         = np.zeros_like(my_muq[cand])
            #this_muq_tmp[:,cand] = np.dot(pjb[(1+H):] * SM.T,SM)
            #this_muq[cand]      += this_muq_tmp
            #this_mus[cand]      += np.inner(SM.T,pjb[(1+H):])

            #denom = pjb.sum()
            #my_mup += this_mup / denom
            #my_muq += this_muq / denom
            #my_mus += this_mus / denom

        ## Calculate updated mu
        #if 'mu' in self.to_learn:
            #tracing.tracepoint("M_step:update mu")
            #mup = np.empty_like(my_mup)
            #muq = np.empty_like(my_muq)
            #mus = np.empty_like(my_mus)
            #all_data_sum = np.empty_like(data_sum)
            #comm.Allreduce( [my_mup, MPI.DOUBLE], [mup, MPI.DOUBLE] )
            #comm.Allreduce( [my_muq, MPI.DOUBLE], [muq, MPI.DOUBLE] )
            #comm.Allreduce( [my_mus, MPI.DOUBLE], [mus, MPI.DOUBLE] )
            #comm.Allreduce( [data_sum, MPI.DOUBLE], [all_data_sum, MPI.DOUBLE] )
            #mu_numer = all_data_sum - np.dot(mus,np.dot(np.linalg.inv(muq), mup))
            #mu_denom = my_N - np.dot(mus,np.dot(np.linalg.inv(muq), mus))
            #mu_new  = mu_numer/ mu_denom
        #else:
            #mu_new = mu

        # Iterate over all datapoints
        tracing.tracepoint("M_step:iterating")
        for n in xrange(my_N):
            y     = my_y[n,:]-mu                  # length D
            cand  = candidates[n,:] # length Hprime
            pjb   = pjb_all[n, :]

            this_Wp = np.zeros_like(my_Wp)    # numerator for current datapoint   (H, D)
            this_Wq = np.zeros_like(my_Wq)    # denominator for current datapoint (H, H)
            this_pi = 0.0                     # numerator for pi update (current datapoint)

            # Zero active hidden cause (do nothing for the W and pi case) 
            # this_Wp += 0.     # nothing to do
            # this_Wq += 0.     # nothing to do
            # this_pi += 0.     # nothing to do

            # One active hidden cause
            this_Wp = np.outer(pjb[1:(H+1)],y)
            this_Wq = pjb[1:(H+1)] * np.identity(H)
            this_pi = pjb[1:(H+1)].sum()
            this_mus = pjb[1:(H+1)].copy()

            # Handle hidden states with more than 1 active cause
            this_Wp[cand]      += np.dot(np.outer(y,pjb[(1+H):]),SM).T
            this_Wq_tmp         = np.zeros_like(my_Wq[cand])
            this_Wq_tmp[:,cand] = np.dot(pjb[(1+H):] * SM.T,SM)
            this_Wq[cand]      += this_Wq_tmp
            this_pi            += np.inner(pjb[(1+H):], SM.sum(axis=1))
            this_mus[cand]     += np.inner(SM.T,pjb[(1+H):])

            denom = pjb.sum()
            my_Wp += this_Wp / denom
            my_Wq += this_Wq / denom
            my_pi += this_pi / denom
            my_mus += this_mus / denom

        # Calculate updated W
        if 'W' in self.to_learn:
            tracing.tracepoint("M_step:update W")
            Wp = np.empty_like(my_Wp)
            Wq = np.empty_like(my_Wq)
            comm.Allreduce( [my_Wp, MPI.DOUBLE], [Wp, MPI.DOUBLE] )
            comm.Allreduce( [my_Wq, MPI.DOUBLE], [Wq, MPI.DOUBLE] )
            #W_new  = np.dot(np.linalg.inv(Wq), Wp)
            #W_new  = np.linalg.solve(Wq, Wp)      # TODO check and switch to this one
            W_new  = np.linalg.lstsq(Wq, Wp)[0]    # TODO check and switch to this one
        else:
            W_new = W

        # Calculate updated pi
        if 'pi' in self.to_learn:
            tracing.tracepoint("M_step:update pi")
            pi_new = E_pi_gamma * comm.allreduce(my_pi) / H / N_use
        else:
            pi_new = pies

        # Calculate updated sigma
        if 'sigma' in self.to_learn:
            tracing.tracepoint("M_step:update sigma")
            # Loop for sigma update:
            for n in xrange(my_N):
                y     = my_y[n,:]-mu           # length D
                cand  = candidates[n,:]     # length Hprime
                logpj = logpj_all[n,:]      # length no_states
                corr  = logpj.max()         # scalar
                pjb   = np.exp(logpj - corr)

                # Zero active hidden causes
                this_sigma = pjb[0] * (y**2).sum()

                # Hidden states with one active cause
                this_sigma += (pjb[1:(H+1)] * ((W-y)**2).sum(axis=1)).sum()

                # Handle hidden states with more than 1 active cause
                SM = self.state_matrix                 # is (no_states, Hprime)
                W_ = W[cand]                           # is (Hprime x D)
                Wbar = np.dot(SM,W_)
                this_sigma += (pjb[(H+1):] * ((Wbar-y)**2).sum(axis=1)).sum()

                denom = pjb.sum()
                my_sigma += this_sigma/ denom

            sigma_new = np.sqrt(comm.allreduce(my_sigma) / D / N_use)
        else:
            sigma_new = sigma
        
        # Calculate updated mu:
        if 'mu' in self.to_learn:
            tracing.tracepoint("M_step:update mu")
            mus = np.empty_like(my_mus)
            all_data_sum = np.empty_like(data_sum)
            comm.Allreduce( [my_mus, MPI.DOUBLE], [mus, MPI.DOUBLE] )
            comm.Allreduce( [data_sum, MPI.DOUBLE], [all_data_sum, MPI.DOUBLE] )
            mu_new = all_data_sum/my_N - np.inner(W_new.T/my_N,mus)
        else:
            mu_new = mu

        for param in anneal.crit_params:
            exec('this_param = ' + param)
            anneal.dyn_param(param, this_param)

        dlog.append('N_use', N_use)

        return { 'W': W_new.T, 'pi': pi_new, 'sigma': sigma_new, 'mu': mu_new }

