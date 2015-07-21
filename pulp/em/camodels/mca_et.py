#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

import numpy as np
from math import pi
from scipy.misc import comb
from mpi4py import MPI

import pulp.em as em
import pulp.utils.parallel as parallel
import pulp.utils.tracing as tracing

from pulp.utils.datalog import dlog
from pulp.em.camodels import CAModel


class MCA_ET(CAModel):
    def __init__(self, D, H, Hprime, gamma, to_learn=['W', 'pi', 'sigma'], comm=MPI.COMM_WORLD):
        """ MCA-ET init method.

        Takes data dimension *D*, number of hidden causes *H*, 
        and ET approximation parameters *Hprime* and *gamma*. Optional
        list of parameters *to_learn* and MPI *comm* object.    
        """
        CAModel.__init__(self, D, H, Hprime, gamma, to_learn, comm)
            
        # 
        self.rho_temp_bound = 1.05    # for rho: never use a T smaller than this
        self.W_tol = 1e-4             # for W: ensure W[W<W_tol] = W_tol

        # Noise Policy
        W_tol = self.W_tol
        self.noise_policy = {
            'W'    : ( W_tol,   +np.inf, True            ),
            'pi'   : ( W_tol,   1-W_tol, False           ),
            'sigma': ( W_tol,   +np.inf, False           )
        }

    @tracing.traced
    def check_params(self, model_params):
        """
        Sanity-check the given model parameters. Raises an exception if something 
        is severely wrong.
        """
        # XXX
        #model_params = CAModel.check_params(self, model_params)

        # Obey W_tol
        model_params['W'] = np.maximum(model_params['W'], self.W_tol)  

        return model_params

    @tracing.traced
    def generate_data(self, model_params, my_N):
        """ Generate data according to the MCA model.

        This method does _not_ obey gamma: The generated data may have more
        than gamma active causes for a given datapoint.
        """
        H, D = self.H, self.D

        W     = model_params['W']
        pies  = model_params['pi']
        sigma = model_params['sigma']

        # Create output arrays, y is data, s is ground-truth
        y = np.zeros( (my_N, D) )
        s = np.zeros( (my_N, H), dtype=np.bool )

        for n in xrange(my_N):
            p = np.random.random(H)        # Create latent vector
            s[n] = p < pies                # Translate into boolean latent vector
            for h in xrange(H):            # Combine according to max-rule
                if s[n,h]:
                    y[n] = np.maximum(y[n], W[h])

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
        comm      = self.comm
        my_y      = data['y']
        my_N, _   = my_y.shape
        H, Hprime = self.H, self.Hprime
        W         = model_params['W']

        # Allocate return structure
        candidates = np.zeros( (my_N, Hprime), dtype=np.int )
        
        #TODO: When using different pies this should be changed!
        for n in xrange(my_N):
            W_interm = np.maximum(W, my_y[n])
            sim = np.abs(W_interm-my_y[n]).sum(axis=1)
            candidates[n] = np.argsort(sim)[0:Hprime]

        data['candidates'] = candidates

        return data

    @tracing.traced
    def E_step(self, anneal, model_params, my_data):
        """ MCA E_step

        my_data variables used:
            
            my_data['y']           Datapoints
            my_data['can']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing AND softmax
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """
        comm      = self.comm
        my_y      = my_data['y']
        my_cand   = my_data['candidates']
        my_N, D   = my_data['y'].shape
        H = self.H

        state_mtx = self.state_matrix        # shape: (no_states, Hprime)
        state_abs = self.state_abs           # shape: (no_states,)
        no_states = len(state_abs)

        W         = model_params['W']
        pies      = model_params['pi']
        sigma     = model_params['sigma']

        # Precompute 
        T        = anneal['T'] 
        T_rho    = np.maximum(T, self.rho_temp_bound)
        rho      = 1./(1.-1./T_rho)
        beta     = 1./T
        pre1     = -1./2./sigma/sigma
        pil_bar  = np.log( pies/(1.-pies) )
        Wl       = np.log(W)
        Wrho     = np.exp(rho * Wl)

        # Allocate return structures
        F = np.empty( [my_N, 1+H+no_states] )

        # Iterate over all datapoints
        for n in xrange(my_N):
            tracing.tracepoint("E_step:iterating")
            y    = my_y[n,:]
            cand = my_cand[n,:]

            # Zero active hidden causes
            log_prod_joint = pre1 * (y**2).sum()
            F[n,0] = log_prod_joint

            # Hidden states with one active cause
            log_prod_joint = pil_bar + pre1 * ((W-y)**2).sum(axis=1)
            F[n,1:H+1] = log_prod_joint

            # Handle hidden states with more than 1 active cause
            log_prior = pil_bar * state_abs        # is (no_states,)
            Wrho_ = Wrho[cand]                     # is (Hprime x D)

            Wbar = np.exp(np.log(np.dot(state_mtx, Wrho_))/rho)
            log_prod_joint = log_prior + pre1 * ((Wbar-y)**2).sum(axis=1)
            F[n,1+H:] = log_prod_joint

        assert np.isfinite(F).all()

        return { 'logpj': F }

    @tracing.traced
    def M_step(self, anneal, model_params, my_suff_stat, my_data):
        """ MCA M_step

        my_data variables used:
            
            my_data['y']           Datapoints
            my_data['candidates']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing AND softmax
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """
        comm      = self.comm
        H, Hprime = self.H, self.Hprime
        gamma     = self.gamma
        W         = model_params['W']
        pies      = model_params['pi']
        sigma     = model_params['sigma']

        # Read in data:
        my_y      = my_data['y']
        my_cand   = my_data['candidates']
        my_logpj  = my_suff_stat['logpj']
        my_N, D   = my_y.shape
        N         = comm.allreduce(my_N)

        state_mtx = self.state_matrix        # shape: (no_states, Hprime)
        state_abs = self.state_abs           # shape: (no_states,)
        no_states = len(state_abs)


        # To compute et_loglike:
        my_ldenom_sum = 0.0
        ldenom_sum = 0.0

        # Precompute 
        T        = anneal['T'] 
        T_rho    = np.maximum(T, self.rho_temp_bound)
        rho      = 1./(1.-1./T_rho)
        beta     = 1./T
        pre0     = (1.-rho)/rho
        pre1     = -1./2./sigma/sigma
        pil_bar  = np.log( pies/(1.-pies) )
        Wl       = np.log(W)
        Wrho     = np.exp(rho * Wl)
        Wsquared = W*W

        # Some asserts
        assert np.isfinite(pil_bar).all()
        assert np.isfinite(Wl).all()
        assert np.isfinite(Wrho).all()
        assert (Wrho > 1e-86).all()

        my_corr  = beta*((my_logpj).max(axis=1))            # shape: (my_N,)
        my_pjb   = np.exp(beta*my_logpj - my_corr[:, None]) # shape: (my_N, no_states)

        # Precompute factor for pi/gamma update
        A_pi_gamma = 0.; B_pi_gamma = 0.
        for gp in xrange(0, self.gamma+1):
            a = comb(H, gp, exact=1) * pies**gp * (1.-pies)**(H-gp)
            A_pi_gamma += a
            B_pi_gamma += gp * a

        # Truncate data
        if anneal['Ncut_factor'] > 0.0:
            tracing.tracepoint("M_step:truncating")
            my_denoms = np.log(my_pjb.sum(axis=1)) + my_corr
            N_use = int(N * (1 - (1 - A_pi_gamma) * anneal['Ncut_factor']))

            cut_denom = parallel.allsort(my_denoms)[-N_use]
            which     = np.array(my_denoms >= cut_denom)
            
            my_y     = my_y[which]
            my_cand  = my_cand[which]
            my_logpj = my_logpj[which]
            my_pjb   = my_pjb[which]
            my_corr  = my_corr[which]
            my_N, D  = my_y.shape
            N_use    = comm.allreduce(my_N)
        else:
            N_use = N
        dlog.append('N_use', N_use)
            
        # Allocate suff-stat arrays
        my_Wp    = np.zeros_like(W)  # shape (H, D)
        my_Wq    = np.zeros_like(W)  # shape (H, D)
        my_pi    = 0.0               #
        my_sigma = 0.0               #

        # Iterate over all datapoints
        for n in xrange(my_N):
            tracing.tracepoint("M_step:iterating")
            y     = my_y[n,:]             # shape (D,)
            cand  = my_cand[n,:]          # shape (Hprime,)
            logpj = my_logpj[n,:]         # shape (no_states,)
            pjb   = my_pjb[n,:]           # shape (no_states,)
            corr  = my_corr[n]            # scalar

            this_Wp = np.zeros_like(W)    # numerator for W (current datapoint)   (H, D)
            this_Wq = np.zeros_like(W)    # denominator for W (current datapoint) (H, D)
            this_pi = 0.0                 # numerator for pi update (current datapoint)
            this_sigma = 0.0              # numerator for gamma update (current datapoint)

            # Zero active hidden causes
            # this_Wp += 0.     # nothing to do
            # this_Wq += 0.     # nothing to do
            # this_pi += 0.     # nothing to do
            this_sigma += pjb[0] * (y**2).sum()

            # One active hidden cause
            this_Wp    += (pjb[1:(H+1),None] * Wsquared[:,:]) * y[None, :]
            this_Wq    += (pjb[1:(H+1),None] * Wsquared[:,:])
            this_pi    += pjb[1:(H+1)].sum()
            this_sigma += (pjb[1:(H+1)] * ((W-y)**2).sum(axis=1)).sum()

            # Handle hidden states with more than 1 active cause
            W_    = W[cand]                                    # is (Hprime, D)
            Wl_   = Wl[cand]                                   # is (   "    ")
            Wrho_ = Wrho[cand]                                 # is (   "    ")

            Wlrhom1 = (rho-1)*Wl_                              # is (Hprime, D)
            Wlbar   = np.log(np.dot(state_mtx,Wrho_)) / rho    # is (no_states, D)
            Wbar    = np.exp(Wlbar)                            # is (no_states, D)
            blpj    = beta*logpj[1+H:] - corr                  # is (no_states,)

            Aid  = (state_mtx[:,:, None] * np.exp(blpj[:,None,None] + (1-rho)*Wlbar[:, None, :] + Wlrhom1[None, :, :])).sum(axis=0)

            assert np.isfinite(Wlbar).all()
            assert np.isfinite(Wbar).all()
            assert np.isfinite(pjb).all()
            assert np.isfinite(Aid).all()

            this_Wp[cand] += Aid * y[None, :]                     
            this_Wq[cand] += Aid
            this_pi       += (pjb[1+H:] * state_abs).sum()
            this_sigma    += (pjb[1+H:] * ((Wbar-y)**2).sum(axis=1)).sum()

            denom     = pjb.sum()
            my_Wp    += this_Wp / denom
            my_Wq    += this_Wq / denom
            my_pi    += this_pi / denom
            my_sigma += this_sigma / denom

            my_ldenom_sum += np.log(np.sum(np.exp(logpj))) #For loglike computation


        # Calculate updated W
        if 'W' in self.to_learn:
            tracing.tracepoint("M_step:update W")

            Wp = np.empty_like(my_Wp)
            Wq = np.empty_like(my_Wq)

            assert np.isfinite(my_Wp).all()
            assert np.isfinite(my_Wq).all()

            comm.Allreduce( [my_Wp, MPI.DOUBLE], [Wp, MPI.DOUBLE] )
            comm.Allreduce( [my_Wq, MPI.DOUBLE], [Wq, MPI.DOUBLE] )

            # Make sure wo do not devide by zero
            tiny = np.finfo(Wq.dtype).tiny
            Wp[Wq < tiny] = 0.
            Wq[Wq < tiny] = tiny

            W_new = Wp / Wq
        else:
            W_new = W

        # Calculate updated pi
        if 'pi' in self.to_learn:
            tracing.tracepoint("M_step:update pi")

            assert np.isfinite(my_pi).all()
            pi_new = A_pi_gamma / B_pi_gamma * pies * comm.allreduce(my_pi) / N_use
        else:
            pi_new = pies

        # Calculate updated sigma
        if 'sigma' in self.to_learn:               # TODO: XXX see LinCA XXX (merge!)
            tracing.tracepoint("M_step:update sigma")

            assert np.isfinite(my_sigma).all()
            sigma_new = np.sqrt(comm.allreduce(my_sigma) / D / N_use)
        else:
            sigma_new = sigma

        #Put all together and compute (always) et_approx_likelihood
        ldenom_sum = comm.allreduce(my_ldenom_sum)
        lAi = (H * np.log(1. - pi_new)) - ((D/2) * np.log(2*pi)) -( D * np.log(sigma_new))

        #For practical and et approx reasons we use: sum of restected respons=1
        loglike_et = (lAi * N_use) + ldenom_sum

        return { 'W': W_new, 'pi': pi_new, 'sigma': sigma_new , 'Q':loglike_et}


    def calculate_respons(self, anneal, model_params, data):
        data['candidates'].sort(axis=1) #(we do this to set the order back=outside)
        F_JB = self.E_step(anneal, model_params, data)['logpj']
        #Transform into responsabilities
        corr = np.max(F_JB, axis=1)       
        exp_F_JB_corr = np.exp(F_JB - corr[:, None])
        respons = exp_F_JB_corr/(np.sum(exp_F_JB_corr, axis=1).reshape(-1, 1))
        return respons


