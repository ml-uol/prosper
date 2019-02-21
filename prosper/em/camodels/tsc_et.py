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
        self._update_state_matrix()
        # l=len(states)
        # icond=True
        # for i in xrange(0,l):
        #     if (states[i]==0):
        #         continue
        #     if icond:
        #         icond=False
        #         ss=np.eye( self.H,dtype=np.int8)*states[i]
        #         continue
        #     temp=np.eye(self.H,dtype=np.int8)*states[i]
        #     ss=np.concatenate((ss,temp))
            
        # self.fullSM=ss[np.sum(np.abs(ss),1)==1]                                # For ternary 2*HxH
        # s=np.empty((l**Hprime,Hprime),dtype=np.int8)
        # c=0
        # ar=np.array(states)
        # for i in itls.product(ar,repeat=Hprime):
        #     s[c]=i
        #     c+=1
        # states_abs=np.empty((l,l**Hprime))
        # for i in range(l):
        #     states_abs[i,:]=(s==states[i]).sum(axis=1)
        # # Noise Policy
        # tol = 1e-5
        # self.noise_policy = {
        #     'W'    : (-np.inf, +np.inf, False ),
        #     'pi'   : (    tol,  1.-tol, False ),
        #     'sigma': (     0., +np.inf, False )
        # }
        # # Generate state-space list
        # self.state_matrix = s[np.sum(np.abs(s),axis=1)<=gamma]
        # self.no_states=s.shape[0]
        # self.state_abs=states_abs

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
    def inference(self, anneal, model_params, my_data, no_maps=10,abs_marginal=True):
        W = model_params['W'].T
        my_y = my_data['y']
        H, D = W.shape
        my_N, D = my_y.shape

        # Prepare return structure
        if no_maps==-1:
            no_maps=self.state_matrix.shape[0]
        res = {
            's': np.zeros( (my_N, no_maps, H), dtype=np.int8),
            'm': np.zeros( (my_N, H) ),
            'am' : np.zeros( (my_N, H) ),
            'p': np.zeros( (my_N, no_maps) )
        }

        if 'candidates' not in my_data:
            my_data = self.select_Hprimes(model_params, my_data)
        my_cand = my_data['candidates']


        my_suff_stat = self.E_step(anneal, model_params, my_data)
        my_logpj  = my_suff_stat['logpj']
        my_corr   = my_logpj.max(axis=1)           # shape: (my_N,)
        my_logpjc = my_logpj - my_corr[:, None]    # shape: (my_N, no_states)
        my_pjc    = np.exp(my_logpjc)              # shape: (my_N, no_states)
        my_denomc = my_pjc.sum(axis=1)             # shape: (my_N)
        
        idx = np.argsort(my_logpjc, axis=-1)[:, ::-1]

        for n in range(my_N):                                   # XXX Vectorize XXX
            for m in range(no_maps):
                this_idx = idx[n,m]
                res['p'][n,m] = my_pjc[n, this_idx] / my_denomc[n]
                s_prime = self.state_matrix[this_idx]
                res['s'][n,m,my_cand[n,:]] = s_prime
            res['m'][n,my_cand[n]] = (my_pjc[n][:,None]*self.state_matrix/my_denomc[n]).sum(0)
            if abs_marginal:
                res['am'][n,my_cand[n]] = (my_pjc[n][:,None]*np.abs(self.state_matrix)/my_denomc[n]).sum(0)


        which = ((res['s'][:,0,:]!=0).sum(-1)==self.gamma)
        my_small_data={}
        my_small_data['y']=my_data['y'][which]
        my_small_data['candidates']=my_data['candidates'][which]
        my_small_cand = my_small_data['candidates']
        gamma_tmp= self.gamma
        Hprime_tmp = self.Hprime
        while which.any():
            print("Data with activity gamma: {}".format(which.sum()))
            if self.gamma+1 == self.H:
                continue
            self.gamma+=1
            self.Hprime+=1
            self._update_state_matrix()
            my_small_data = self.select_Hprimes(model_params, my_small_data)
            my_small_cand = my_small_data['candidates']
            my_suff_stat = self.E_step(anneal, model_params, my_small_data)
            my_logpj  = my_suff_stat['logpj']
            my_corr   = my_logpj.max(axis=1)           # shape: (my_N,)
            my_logpjc = my_logpj - my_corr[:, None]    # shape: (my_N, no_states)
            my_pjc    = np.exp(my_logpjc)              # shape: (my_N, no_states)
            my_denomc = my_pjc.sum(axis=1)             # shape: (my_N)
            
            idx = np.argsort(my_logpjc, axis=-1)[:, ::-1]
            my_N=np.sum(which)
            for n in range(my_N):                                   # XXX Vectorize XXX
                for m in range(no_maps):
                    higam_inds = np.where(which)[0]
                    this_idx = idx[n,m]   #state matrix index of map m
                    res['p'][higam_inds[n],m] = my_pjc[n, this_idx] / my_denomc[n]
                    s_prime = self.state_matrix[this_idx]
                    res['s'][higam_inds[n],m,my_small_cand[n,:]] = s_prime
                res['m'][higam_inds[n],my_small_cand[n,:]] = (my_pjc[n][:,None]*self.state_matrix/my_denomc[n]).sum(0)
                if abs_marginal:
                    res['am'][higam_inds[n],my_small_cand[n,:]] = (my_pjc[n][:,None]*np.abs(self.state_matrix)/my_denomc[n]).sum(0)
            which = ((res['s'][:,0,:]!=0).sum(-1)==self.gamma)
            my_small_data['y']=my_data['y'][which]
            my_small_data['candidates']=my_data['candidates'][which]

        self.gamma = gamma_tmp
        self.Hprime = Hprime_tmp
        self._update_state_matrix()
        return res

    def _update_state_matrix(self):
        l=len(self.states)
        icond=True
        Hprime = self.Hprime
        for i in range(0,l):
            if (self.states[i]==0):
                continue
            if icond:
                icond=False
                ss=np.eye( self.H,dtype=np.int8)*self.states[i]
                continue
            temp=np.eye(self.H,dtype=np.int8)*self.states[i]
            ss=np.concatenate((ss,temp))
            
        self.fullSM=ss[np.sum(np.abs(ss),1)==1]                                # For ternary 2*HxH
        s=np.empty((l**Hprime,Hprime),dtype=np.int8)
        c=0
        ar=np.array(self.states)
        for i in itls.product(ar,repeat=Hprime):
            s[c]=i
            c+=1
        states_abs=np.empty((l,l**Hprime))
        for i in range(l):
            states_abs[i,:]=(s==self.states[i]).sum(axis=1)
        # Noise Policy
        tol = 1e-5
        self.noise_policy = {
            'W'    : (-np.inf, +np.inf, False ),
            'pi'   : (    tol,  1.-tol, False ),
            'sigma': (     0., +np.inf, False )
        }
        # Generate state-space list
        self.state_matrix = s[np.sum(np.abs(s),axis=1)<=self.gamma]
        self.no_states=s.shape[0]
        self.state_abs=states_abs
        print("state matrix updated")