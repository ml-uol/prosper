# -*- coding: utf-8 -*-



import numpy as np
from mpi4py import MPI
import itertools as itls
from scipy.misc import comb


import prosper.utils.parallel as parallel
# from prosper.utils.parallel import pprint as pp #use for debugging 
import prosper.utils.tracing as tracing

from prosper.utils.datalog import dlog
from prosper.em.camodels import CAModel
from prosper.em import Model


def generate_state_matrix(Hprime, gamma, H, states):
    """Ternary state space.

    :param Hprime: Vector length
    :type Hprime: int
    :param gamma: Maximum number of ones
    :type gamma: int
    :param H: Dimensionality of latent space
    :type H: int
    :param states: Ternary states
    :type states: np.array

    """

    l=len(states)
    icond=True    
    for i in range(0,l):
        if (states[i]==0):
            continue
        if icond:
            icond=False
            ss=np.eye( H,dtype=np.int8)*states[i]
            continue
        temp=np.eye(H,dtype=np.int8)*states[i]
        ss=np.concatenate((ss,temp))
        
    fullSM=ss[np.sum(np.abs(ss),1)==1]                                # For ternary 2*HxH
    s=np.empty((l**Hprime,Hprime),dtype=np.int8)
    c=0
    ar=np.array(states)
    for i in itls.product(ar,repeat=Hprime):
        s[c]=i
        c+=1
    states_abs=np.empty((l,l**Hprime))
    for i in range(l):
        states_abs[i,:]=(s==states[i]).sum(axis=1)
    
    state_matrix = s[np.sum(np.abs(s),axis=1)<=gamma]
    no_states=s.shape[0]        

    return fullSM, state_matrix, no_states, states_abs


class Ternary_ET(CAModel):
    @tracing.traced
    def __init__(self, D, H, Hprime, gamma,states=np.array([-1.,0.,1.]), to_learn=['W', 'pi', 'sigma'], comm=MPI.COMM_WORLD):
        Model.__init__(self, comm)
        self.to_learn = to_learn
        self.states=states
        # Model meta-parameters
        self.gamma=gamma
        self.D = D
        self.H = H
        self.Hprime=Hprime
        self.fullSM, self.state_matrix, self.no_states, self.state_abs = generate_state_matrix(Hprime, gamma, H, states)

        # Noise Policy
        tol = 1e-5
        self.noise_policy = {
            'W'    : (-np.inf, +np.inf, False ),
            'pi'   : (    tol,  1.-tol, False ),
            'sigma': (     0., +np.inf, False )
        }

    @tracing.traced
    def select_Hprimes(self, model_params, data,):
        """
        Return a new data-dictionary which has been annotated with
        a data['candidates'] dataset. A set of self.Hprime candidates
        will be selected.
        """
        my_N, D   = data['y'].shape
        H         = self.H
        SM        = self.fullSM
        l1,l2     = SM.shape                                          #H=self.H
        
        candidates= np.zeros((my_N, self.Hprime), dtype=np.int)
        W         = model_params['W'].T
        pi        = model_params['pi']
        sigma     = model_params['sigma']
        states    = self.states

        # Precompute 
        pre1     = -1./2./sigma/sigma
        l=len(states)
        pi_matrix=np.empty((l1,H))
        for i in range(l):
            if states[i]==-1:
                pi_matrix[SM[::]==states[i]]=pi/2
            elif states[i]==1:
                pi_matrix[SM[::]==states[i]]=pi/2
            else:
                pi_matrix[SM[::]==states[i]]=1-pi
        # Allocate return structures
        pil_bar=np.log(pi_matrix).sum(axis=1)
        F = np.empty( [my_N, l1] )
        pre_F = np.empty( [my_N, l1] )
        for n in range(my_N):
            tracing.tracepoint("E_step:iterating")
            y    = data['y'][n,:]
            # Handle hidden states with more than 1 active cause
            pre_F[n] = pil_bar                   # is (no_states,)
            Wbar =np.dot(SM,W)
            log_prod_joint = pre1 * (((Wbar-y)**2).sum(axis=1))
            F[n] = log_prod_joint
            # corr = (pre_F[n,:]+F[n,:]).max()
            F__=pre_F[n]+F[n]
            tmp=np.argsort(F__)[-self.Hprime:]
            tmp2=np.nonzero(SM[tmp])[1]            
            candidates[n]=tmp2
        data['candidates']=candidates
        return data
        
    @tracing.traced
    def generate_data(self, model_params, my_N):
        D = self.D
        H = self.H
        pi = model_params['pi']
        W  = model_params['W'].T
        sigma = model_params['sigma']
        # Create output arrays, y is data, s is ground-truth
        y = np.zeros( (my_N, D) )
        s = np.zeros( (my_N, H), dtype=np.int8)
        for n in range(my_N):
                p = np.random.random(H)        # create latent vector
                for i in range(H):
                        if (p[i]<(pi/2)):
                            s[n,i] = -1
                            y[n] += s[n,i]*W[i]
                        elif (p[i]<pi):
                            s[n,i] = 1
                            y[n] += s[n,i]*W[i]
                        else:
                            s[n,i] = 0
                            y[n] += s[n,i]*W[i]
                        
        # Add noise according to the model parameters
        y += np.random.normal( scale=sigma, size=(my_N, D) )
        # Build return structure
        return { 'y': y, 's': s }

    #@tracing.traced
    def E_step(self, anneal, model_params, my_data):
        """ TSC E_step

        my_data variables used:

            my_data['y']           Datapoints
            my_data['can']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """
        my_N, D =   my_data['y'].shape
        SM      =   self.state_matrix
        l1,l2   =   SM.shape
        W       =   model_params['W'].T
        pi      =   model_params['pi']
        sigma   =   model_params['sigma']
        states  =   self.states
        # Precompute 
        beta    =   1./anneal['T']
        pre1    =   -1./2./sigma/sigma
        l       =   len(states)
        pi_matrix=  np.empty((l1,self.Hprime))
        for i in range(l):
            if states[i]==-1:
                pi_matrix[SM[::]==states[i]]=pi/2
            elif states[i]==1:
                pi_matrix[SM[::]==states[i]]=pi/2
            else:
                pi_matrix[SM[::]==states[i]]=1-pi
        # Allocate return structures
        pil_bar =   np.log(pi_matrix).sum(axis=1)
        F       =   np.empty( [my_N, l1] )
        pre_F   =   np.empty( [my_N, l1] )
        # Iterate over all datapoints
        for n in range(my_N):
            #tracing.tracepoint("E_step:iterating")
            y    = my_data['y'][n,:]
            cand = my_data['candidates'][n,:]
            # print "cand  ", cand
            # Handle hidden states with more than 1 active cause
            pre_F[n,:]  =   pil_bar                   # is (no_states,)
            W_   = W[cand]                          # is (Hprime x D)
            
            Wbar = np.dot(SM,W_)
            log_prod_joint = pre1 * (((Wbar-y)**2).sum(axis=1))
            F[n,:] = log_prod_joint#+pil_bar
        
        if anneal['anneal_prior']:
            F += pre_F
            F *= beta
        else:
            F *=beta
            F += pre_F
        return { 'logpj': F}#, 'denoms': denoms}

    #@tracing.traced
    def M_step(self, anneal, model_params, my_suff_stat, my_data):
        """ TSC M_step

        my_data variables used:
            
            my_data['y']           Datapoints
            my_data['candidates']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """        
        comm      = self.comm
        H         = self.H
        gamma     = self.gamma
        W         = model_params['W'].T
        pi        = model_params['pi']
        sigma     = model_params['sigma']

        # Read in data:
        my_y       = my_data['y'].copy()
        candidates = my_data['candidates']
        logpj_all  = my_suff_stat['logpj']
        all_denoms = np.exp(logpj_all).sum(axis=1)
        my_N, D    = my_y.shape
        N          = comm.allreduce(my_N)
        

        SM         = self.state_matrix#[SM_bool]        # shape: (no_states, Hprime)
        state_abs  = np.abs(SM).sum(axis=1)
        
        
        # Precompute factor for pi update
        A_pi_gamma = 0.0
        B_pi_gamma = 0.0
        
        for gam1 in range(gamma+1):
            for gam2 in range(gamma-gam1+1):
                cmb=comb(gam1,gam1)*comb(gam1+gam2,gam2)*comb(H,H-gam1-gam2)
                A_pi_gamma += cmb * ((pi/2)**(gam1+gam2))*((1-pi)**(H-gam1-gam2))
                B_pi_gamma += (gam1+gam2) * cmb * ((pi/2)**(gam1+gam2))*((1-pi)**(H-gam1-gam2))
        E_pi_gamma = pi * H * A_pi_gamma / B_pi_gamma
        
        #Truncate data
        if anneal['Ncut_factor'] > 0.0:
            #tracing.tracepoint("M_step:truncating")
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

        #Log-Likelihood:
        L =  - 0.5 * D * np.log(2*np.pi*sigma**2)-np.log(A_pi_gamma)

        Fs = np.log(np.exp(logpj_all).sum(axis=1)).sum()
        L += comm.allreduce(Fs)/N_use
        dlog.append('L',L)



        # Precompute
        corr_all  = logpj_all.max(axis=1)                 # shape: (my_N,)
        pjb_all   = np.exp(logpj_all - corr_all[:, None])  # shape: (my_N, no_states)
        # Allocate 
        my_Wp     = np.zeros_like(W)   # shape (H, D)
        my_Wq     = np.zeros((H,H))    # shape (H, H)
        my_pi     = 0.0                #
        my_sigma  = 0.0             #


        # Iterate over all datapoints
        for n in range(my_N):
            #tracing.tracepoint("M_step:iterating")
            y     = my_y[n,:]                  # length D
            cand  = candidates[n,:] # length Hprime
            pjb = pjb_all[n, :]
            this_Wp = np.zeros_like(my_Wp)    # numerator for current datapoint   (H, D)
            this_Wq = np.zeros_like(my_Wq)    # denominator for current datapoint (H, H)
            this_pi = np.zeros_like(pi)       # numerator for pi update (current datapoint)
            # Handle hidden states with more than 1 active cause
            this_Wp[cand]           += np.dot(np.outer(y,pjb),SM).T
            this_Wq_tmp             = np.zeros_like(my_Wq[cand])
            this_Wq_tmp[:,cand]     = np.dot(pjb * SM.T,SM)
            this_Wq[cand]           += this_Wq_tmp
            this_pi += np.inner(pjb, state_abs)

            denom = pjb.sum()
            my_Wp += this_Wp / denom
            
            my_Wq += this_Wq / denom
            my_pi += this_pi / denom
        #Calculate updated W
        if 'W' in self.to_learn:
            #tracing.tracepoint("M_step:update W")
            Wp = np.empty_like(my_Wp)
            Wq = np.empty_like(my_Wq)
            comm.Allreduce( [my_Wp, MPI.DOUBLE], [Wp, MPI.DOUBLE] )
            comm.Allreduce( [my_Wq, MPI.DOUBLE], [Wq, MPI.DOUBLE] )
            W_new  = np.dot(np.linalg.pinv(Wq), Wp)
        else:
            W_new = W

        # Calculate updated pi
        pi_new=np.empty_like(pi)
        if 'pi' in self.to_learn:
            #tracing.tracepoint("M_step:update pi")
            pi_new = E_pi_gamma * comm.allreduce(my_pi) / H / N_use
        else:
            pi_new = pi
        # Calculate updated sigma
        if 'sigma' in self.to_learn:
            #tracing.tracepoint("M_step:update sigma")
            # Loop for sigma update:
            for n in range(my_N):
                #tracing.tracepoint("M_step:update sigma iteration")
                y     = my_y[n,:]           # length D
                cand  = candidates[n,:]     # length Hprime
                logpj = logpj_all[n,:]      # length no_states
                corr  = logpj.max()         # scalar
                pjb   = np.exp(logpj - corr)

                # Zero active hidden causes
                #this_sigma = pjb[0] * (y**2).sum()

                # Hidden states with one active cause
                #this_sigma += (pjb[1:(H+1)] * ((W-y)**2).sum(axis=1)).sum()

                # Handle hidden states with more than 1 active cause
                #SM = self.state_matrix                 # is (no_states, Hprime)
                W_ = W[cand]                           # is (Hprime x D)

                Wbar = np.dot(SM,W_)
                this_sigma = (pjb * ((Wbar-y)**2).sum(axis=1)).sum()

                denom = pjb.sum()
                my_sigma += this_sigma/ denom

            sigma_new = np.sqrt(comm.allreduce(my_sigma) / D / N_use)
        else:
            sigma_new = sigma
        
        for param in anneal.crit_params:
            exec('this_param = ' + param)
            anneal.dyn_param(param, this_param)
        
        dlog.append('N_use', N_use)

        return { 'W': W_new.transpose(), 'pi': pi_new, 'sigma': sigma_new, 'Q': 0.}


    @tracing.traced
    def inference(self, anneal, model_params, test_data, topK=10, logprob=False, adaptive=True, abs_marginal=True):
        """
        Perform inference with the learned model on test data and return the top K configurations with their posterior probabilities. 
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
        :param abs_marginal: Return marginal of states at absolut value
        :type abs_marginal: boolean
        """

        assert 'y' in test_data, "Key 'y' in test_data dict not defined."
        
        my_y = test_data['y']        
        my_N, D = my_y.shape
        H = self.H

        # Prepare return structure
        if topK==-1:
            topK=self.state_matrix.shape[0]
        res = {
            's': np.zeros( (my_N, topK, H), dtype=np.int8),
            'm': np.zeros( (my_N, H) ),
            'am' : np.zeros( (my_N, H) ),
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
                        res['p'][n_,m] = my_pjc[n, this_idx] / my_denomc[n]                    
                    s_prime = self.state_matrix[this_idx]
                    res['s'][n_,m,my_cand[n,:]] = s_prime
                    
                res['m'][n_,my_cand[n]] = (my_pjc[n][:,None]*self.state_matrix/my_denomc[n]).sum(0)                                
                if abs_marginal:
                    res['am'][n_,my_cand[n]] = (my_pjc[n][:,None]*np.abs(self.state_matrix)/my_denomc[n]).sum(0)

            if not adaptive:
                break

            which = ((res['s'][:,0,:].astype(bool)!=0).sum(-1)==self.gamma) # shape: (my_N,)
            if not which.any():
                break
            test_data_tmp['y']=my_y[which]
            my_N=np.sum(which)            
            print("For %i data points MAP state has activity equal to gamma." % my_N)
            del test_data_tmp['candidates']            

            if self.Hprime == self.H:
                pass
            else:
                self.Hprime+=1

            if self.gamma == self.H:
                continue
            else:
                self.gamma+=1

            print("Updating state matrix and running again.")
            self.fullSM, self.state_matrix, self.no_states, self.state_abs = generate_state_matrix(self.Hprime, self.gamma, self.H, self.states)

        if logprob:
            res['m'] = np.log(res['m'])
            res['am'] = np.log(res['am'])

        return res
