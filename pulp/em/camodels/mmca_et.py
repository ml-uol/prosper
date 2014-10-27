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
import pulp.utils.accel as accel

from pulp.utils.datalog import dlog
from pulp.em.camodels import CAModel

from pulp.utils.autotable import AutoTable



class MMCA_ET(CAModel):
    def __init__(self, D, H, Hprime, gamma, to_learn=['W', 'pi', 'sigma'], comm=MPI.COMM_WORLD):
        """ MMCA-ET init method.

        Takes data dimension *D*, number of hidden causes *H*, 
        and ET approximation parameters *Hprime* and *gamma*. Optional
        list of parameters *to_learn* and MPI *comm* object.    
        """
        CAModel.__init__(self, D, H, Hprime, gamma, to_learn, comm)
            
        # 
        self.rho_T_bound = 1.20       # for rho: never use a T smaller than this
        self.rho_lbound = 1           # for rho: never use a rho smaller than this
        self.rho_ubound = 35          # for rho: never use a rho larger than this
        self.tol = 1e-4               # for W: ensure W[W<tol] = tol

        self.rev_corr = False

        # Noise Policy
        tol = self.tol
        self.noise_policy = {
            'W'    : ( -np.inf,   +np.inf, False           ),
            'pi'   : (     tol,     1-tol, False           ),
            'sigma': (     tol,   +np.inf, False           )
        }

        # XXX debugging XXX
        #self.tbl = AutoTable("mmca-debug/mmca-debug-%04d.h5" % comm.rank)
        #self.last_candidates = None

    @tracing.traced
    def check_params(self, model_params):
        """
        Sanity-check the given model parameters. Raises an exception if something 
        is severely wrong.
        """
        # XXX
        #model_params = CAModel.check_params(self, model_params)

        tol = self.tol
        W = model_params['W']

        # Ensure |W| >= tol
        W[np.logical_and(W >= 0., W < +tol)] = +tol
        W[np.logical_and(W <= 0., W > -tol)] = -tol

        return model_params

    @tracing.traced
    def generate_from_hidden(self, model_params, my_hdata):
        """ 
            Generate data according to the MCA model while the latents are 
            given in my_hdata['s'].
        """
        W     = model_params['W']
        pies  = model_params['pi']
        sigma = model_params['sigma']
        H, D  = W.shape

        s       = my_hdata['s']
        my_N, _ = s.shape
        
        # Create output arrays, y is data
        y = np.zeros( (my_N, D) )

        for n in xrange(my_N):
            # Combine accoring do magnitude-max rule
            t0 = s[n, :, None] * W              # (H, D)  "stacked" version of a datapoint
            idx = np.argmax(np.abs(t0), axis=0) # Find maximum magnitude in stack 
            y[n] = t0[idx].diagonal()           # Collaps it

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

        #if self.last_candidates is not None:
        #    print "Reusing candidates"
        #    data['candidates'] = self.last_candidates
        #    return data

        # Allocate return structure
        candidates = np.zeros( (my_N, Hprime), dtype=np.int )
        
        for n in xrange(my_N):
            #W_interm = np.maximum(W, my_y[n])
            #sim = np.abs(W_interm-my_y[n]).sum(axis=1)
            sim = ((W-my_y[n])**2).sum(axis=1)
            candidates[n] = np.argsort(sim)[0:Hprime]

        data['candidates'] = candidates
        #self.last_candidates = candidates

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

        # Disable some warnings
        old_seterr = np.seterr(divide='ignore', under='ignore')

        # Precompute 
        T        = anneal['T'] 
        T_rho    = np.maximum(T, self.rho_T_bound)
        rho      = 1./(1.-1./T_rho)
        rho      = np.maximum(np.minimum(rho, self.rho_ubound), self.rho_lbound)
        beta     = 1./T
        pre1     = -1./2./sigma/sigma
        pil_bar  = np.log( pies/(1.-pies) )
        Wl       = accel.log(np.abs(W))
        Wrho     = accel.exp(rho * Wl)
        Wrhos    = np.sign(W) * Wrho

        # Allocate return structures
        F = np.empty( [my_N, 1+H+no_states] )

        # Iterate over all datapoints
        tracing.tracepoint("E_step:iterating...")
        for n in xrange(my_N):
            y    = my_y[n,:]
            cand = my_cand[n,:]

            # Zero active hidden causes
            log_prod_joint = pre1 * (y**2).sum()
            F[n,0] = log_prod_joint

            # Hidden states with one active cause
            log_prod_joint = pil_bar + pre1 * ((W-y)**2).sum(axis=1)
            F[n,1:H+1] = log_prod_joint

            # Handle hidden states with more than 1 active cause
            log_prior = pil_bar * state_abs             # is (no_states,)
            Wrhos_ = Wrhos[cand]                        # is (Hprime, D)

            t0 = np.dot(state_mtx, Wrhos_)
            Wbar = np.sign(t0) * accel.exp(accel.log(np.abs(t0))/rho)
            log_prod_joint = log_prior + pre1 * ((Wbar-y)**2).sum(axis=1)
            F[n,1+H:] = log_prod_joint


        assert np.isfinite(F).all()

        # Restore np.seterr
        np.seterr(**old_seterr)

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

        # Disable some warnings
        old_seterr = np.seterr(divide='ignore', under='ignore')

        # To compute et_loglike:
        my_ldenom_sum = 0.0
        ldenom_sum = 0.0

        # Precompute 
        T        = anneal['T'] 
        T_rho    = np.maximum(T, self.rho_T_bound)
        rho      = 1./(1.-1./T_rho)
        rho      = np.maximum(np.minimum(rho, self.rho_ubound), self.rho_lbound)
        beta     = 1./T
        pre1     = -1./2./sigma/sigma
        pil_bar  = np.log( pies/(1.-pies) )
        Wl       = accel.log(np.abs(W))
        Wrho     = accel.exp(rho * Wl)
        Wrhos    = np.sign(W) * Wrho
        Wsquared = W*W

        # Some asserts
        assert np.isfinite(pil_bar).all()
        assert np.isfinite(Wl).all()
        assert np.isfinite(Wrho).all()
        assert (Wrho > 1e-86).all()

        my_corr   = beta*((my_logpj).max(axis=1))            # shape: (my_N,)
        my_logpjb = beta*my_logpj - my_corr[:, None]         # shape: (my_N, no_states)
        my_pj     = accel.exp(my_logpj)                         # shape: (my_N, no_states)
        my_pjb    = accel.exp(my_logpjb)                        # shape: (my_N, no_states)

        # Precompute factor for pi update and ET cutting
        A_pi_gamma = 0.; B_pi_gamma = 0.
        for gp in xrange(0, self.gamma+1):
            a = comb(H, gp) * pies**gp * (1.-pies)**(H-gp)
            A_pi_gamma += a
            B_pi_gamma += gp * a

        # Truncate data
        if anneal['Ncut_factor'] > 0.0:
            tracing.tracepoint("M_step:truncating")
            my_logdenoms = accel.log(my_pjb.sum(axis=1)) + my_corr
            N_use = int(N * (1 - (1 - A_pi_gamma) * anneal['Ncut_factor']))

            cut_denom = parallel.allsort(my_logdenoms)[-N_use]
            my_sel,   = np.where(my_logdenoms >= cut_denom)
            my_N,     = my_sel.shape
            N_use     = comm.allreduce(my_N)
        else:
            my_N,_ = my_y.shape
            my_sel = np.arange(my_N)
            N_use  = N
            
        # Allocate suff-stat arrays
        my_Wp    = np.zeros_like(W)  # shape (H, D)
        my_Wq    = np.zeros_like(W)  # shape (H, D)
        my_pi    = 0.0               #
        my_sigma = 0.0               #

        # Do reverse correlation if requested
        if self.rev_corr:
            my_y_rc   = my_data['y_rc']
            D_rev_corr  = my_y_rc.shape[1]
            my_rev_corr = np.zeros( (H,D_rev_corr) )
            my_rev_corr_count = np.zeros(H)

        # Iterate over all datapoints
        tracing.tracepoint("M_step:iterating...")
        dlog.append('N_use', N_use)
        for n in my_sel:
            y      = my_y[n,:]             # shape (D,)
            cand   = my_cand[n,:]          # shape (Hprime,)
            logpj  = my_logpj[n,:]         # shape (no_states,)
            logpjb = my_logpjb[n,:]        # shape (no_states,)
            pj     = my_pj[n,:]            # shape (no_states,)
            pjb    = my_pjb[n,:]           # shape (no_states,)

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
            this_Wp    += (pjb[1:(H+1),None]) * y[None, :]
            this_Wq    += (pjb[1:(H+1),None])
            this_pi    +=  pjb[1:(H+1)].sum()
            this_sigma += (pjb[1:(H+1)] * ((W-y)**2).sum(axis=1)).sum()

            # Handle hidden states with more than 1 active cause
            W_     = W[cand]                                    # is (Hprime, D)
            Wl_    = Wl[cand]                                   # is (   "    ")
            Wrho_  = Wrho[cand]                                 # is (   "    ")
            Wrhos_ = Wrhos[cand]                                # is (   "    ")

            #Wbar   = calc_Wbar(state_mtx, W_)
            #Wlbar  = np.log(np.abs(Wbar))

            t0 = np.dot(state_mtx, Wrhos_)
            Wlbar   = accel.log(np.abs(t0)) / rho    # is (no_states, D)
            #Wlbar   = np.maximum(Wlbar, -9.21)
            Wbar    = np.sign(t0)*accel.exp(Wlbar)   # is (no_states, D)

            t = Wlbar[:, None, :]-Wl_[None, :, :]
            t = np.maximum(t, 0.)
            Aid = state_mtx[:,:, None] * accel.exp(logpjb[H+1:,None,None] - (rho-1)*t)
            Aid = Aid.sum(axis=0)

            #Aid = calc_Aid(logpjb[H+1:], W_, Wl_, state_mtx, Wbar, Wlbar, rho)

            #assert np.isfinite(Wlbar).all()
            #assert np.isfinite(Wbar).all()
            #assert np.isfinite(pjb).all()
            #assert np.isfinite(Aid).all()

            this_Wp[cand] += Aid * y[None, :]                     
            this_Wq[cand] += Aid
            this_pi       += (pjb[1+H:] * state_abs).sum()
            this_sigma    += (pjb[1+H:] * ((Wbar-y)**2).sum(axis=1)).sum()

            denom     = pjb.sum()
            my_Wp    += this_Wp / denom
            my_Wq    += this_Wq / denom
            my_pi    += this_pi / denom
            my_sigma += this_sigma / denom

            #self.tbl.append("logpj", logpj)
            #self.tbl.append("corr", my_corr[n])
            #self.tbl.append("denom", denom)
            #self.tbl.append("cand", cand)
            #self.tbl.append("Aid", Aid)

            my_ldenom_sum += accel.log(np.sum(accel.exp(logpj))) #For loglike computation

            # Estimate reverse correlation
            if self.rev_corr:
                pys = pjb / denom
                if np.isfinite(pys).all():
                    my_rev_corr       += pys[1:H+1, None]*my_y_rc[n,None,:]
                    my_rev_corr_count += pys[1:H+1]
                    my_rev_corr[cand]       += np.sum(state_mtx[:,:,None]*pys[H+1:,None,None]*my_y_rc[n,None,:], axis=0)
                    my_rev_corr_count[cand] += np.sum(state_mtx[:,:]*pys[H+1,None], axis=0)
                else:
                    print "Not all finite rev_corr %d" % n
 


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
            tiny = self.tol
            Wq[Wq < tiny] = tiny

            # Calculate updated W
            W_new = Wp / Wq

            # Add inertia depending on Wq
            alpha = 2.5
            inertia = np.maximum(1. - accel.exp(-Wq / alpha), 0.2)
            W_new = inertia*W_new + (1-inertia)*W
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

        # Put all together and compute (always) et_approx_likelihood
        ldenom_sum = comm.allreduce(my_ldenom_sum)
        lAi = (H * np.log(1. - pi_new)) - ((D/2) * np.log(2*pi)) -( D * np.log(sigma_new))

        # For practical and et approx reasons we use: sum of restected respons=1
        loglike_et = (lAi * N_use) + ldenom_sum

        if self.rev_corr:
            rev_corr       = np.empty_like(my_rev_corr)
            rev_corr_count = np.empty_like(my_rev_corr_count)
            comm.Allreduce( [my_rev_corr,       MPI.DOUBLE], [rev_corr,       MPI.DOUBLE])
            comm.Allreduce( [my_rev_corr_count, MPI.DOUBLE], [rev_corr_count, MPI.DOUBLE])
            rev_corr /= (1e-16+rev_corr_count[:,None])
        else:
            rev_corr = np.zeros( (H, D) )


        # Restore np.seterr
        np.seterr(**old_seterr)

        return { 'W': W_new, 'pi': pi_new, 'sigma': sigma_new , 'rev_corr': rev_corr, 'Q':loglike_et}

