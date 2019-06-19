#
#  Lincense: Academic Free License (AFL) v3.0
#



import numpy as np
from mpi4py import MPI
import itertools as itls
from scipy.special import logsumexp

import prosper.em as em
import prosper.utils.parallel as parallel
import prosper.utils.tracing as tracing

from prosper.utils.datalog import dlog
from prosper.em.camodels import CAModel


from scipy.special import gammaln


def multinom2(a,b):
    """
    :param a: array
    :param b: array
    :return:np.exp(gammaln(a)-gammaln(b).sum())
    """
    return np.exp(gammaln(a+1)-gammaln(b+1).sum())

def np_product(iterable,repeat=2):
    return np.array(iterable)[np.moveaxis(
        np.indices((len(iterable),) * repeat), 0, repeat )
        .reshape(-1, repeat)]

def get_states(states, Hprime, gamma):
    tmp = np.array(list(itls.product(states, repeat=Hprime)))
    c1 = (np.sum(tmp != 0, 1) <= gamma) * (np.sum(tmp != 0, 1) > 1)
    return tmp[c1]

def get_states_np(states, Hprime, gamma):
    tmp = np_product(states, repeat=Hprime)
    c1 = (np.sum(tmp != 0, 1) <= gamma) * (np.sum(tmp != 0, 1) > 1)
    return tmp[c1]

def get_states2(states, Hprime, gamma):
    # assert type(states)==np.array
    assert len(states.shape)==1
    pd = itls.product(states, repeat=Hprime)
    sl = [np.array(c) for c in pd if (np.sum(np.array(c)!=0)<=gamma and np.sum(np.array(c)!=0)>1)]
    tmp = np.array(sl)
    # c1 = (np.sum(tmp != 0, 1) <= gamma) * (np.sum(tmp != 0, 1) > 1)
    return tmp


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


class DSC_ET(CAModel):
    def __init__(self, D, H, Hprime, gamma,states=np.array([-1.,0.,1.]), to_learn=['W', 'pi', 'sigma'], comm=MPI.COMM_WORLD):
        CAModel.__init__(self, D, H, Hprime, gamma, to_learn, comm)
        self.comm = comm
        if not type(states) == np.ndarray:
            raise TypeError("DSC: states must be of type numpy.ndarray")
        if Hprime > H:
            raise Exception("Hprime must be less or equal to H")
        if gamma > Hprime:
            raise Exception("gamma must be less or equal to Hprime")
        self.to_learn = to_learn

        # Model meta-parameters
        self.D = D
        self.H = H
        self.Hprime = Hprime
        self.gamma = gamma
        self.states = states
        self.K = self.states.shape[0]
        self.K_0 = int(np.argwhere(states == 0.))

        # some sanity checks
        assert Hprime <= H
        assert gamma <= Hprime

        # Noise Policy
        tol = 1e-5
        self.noise_policy = {
            'W': (-np.inf, +np.inf, False),
            'pi': (tol,  1. - tol, False),
            'sigma': (0., +np.inf, False)
        }

        # Generate state-space list
        ss = np.empty((0, self.H), dtype=np.int8)
        for i in range(self.K):
            if (i == self.K_0):
                continue
            temp = np.eye(self.H, dtype=np.int8) * states[i]
            ss = np.concatenate((ss, temp))

        # all hidden vectors with a single active cause - for ternary 2*HxH
        self.single_state_matrix = ss[np.sum(np.abs(np.sign(ss)), 1) == 1]

        # all hidden vectors with more than one active cause
        self.state_matrix = get_states(
            self.states, self.Hprime, self.gamma)

        # number of states with more than one active cause
        self.no_states = self.state_matrix.shape[0]

        #
        self.state_abs = np.empty((self.K, self.no_states))
        for i in range(self.K):
            self.state_abs[i, :] = (
                self.state_matrix == self.states[i]).sum(axis=1)
        self.state_abs[self.K_0, :] = self.H - \
            self.state_abs.sum(0) + self.state_abs[self.K_0, :]


    def check_params(self, model_params):
        """ Sanity check.

        Sanity-check the given model parameters. Raises an exception if
        something is severely wrong.
        """
        W     = model_params['W']
        pies  = model_params['pi']
        sigma = model_params['sigma']

        assert np.isfinite(W).all()      # check W

        assert np.isfinite(pies).all()   # check pies
        # assert pies.sum()<=1.
        # assert (pies>=0).all()
        #assert pies <= 1.

        assert np.isfinite(sigma).all()  # check sigma
        assert sigma >= 0.

        return model_params

    def generate_data(self, model_params, my_N, noise_on=True, gs=None, gp=None):
        D = self.D
        H = self.H
        states = self.states
        pi = model_params['pi']
        W = model_params['W'].T
        sigma = model_params['sigma']
        # Create output arrays, y is data, s is ground-truth
        y = np.zeros((my_N, D))
        s = np.zeros((my_N, H), dtype=np.int8)
        # print(states)
        # print(pi)
        for n in range(my_N):
            if gs is None:
                s[n] = np.random.choice(states, size=H, replace=True, p=pi)
            else:
                assert gs.shape[0]==my_N
                if gp is None:
                    assert len(gs.shape)==2
                    s[n]=gs[n]
                else:
                    assert gp.shape[0]==my_N
                    assert gp.shape[1]==gs.shape[1]
                    s[n]=(gs[n]*gp[n]).sum(0)
                
            y[n] = np.dot(s[n], W)
        # Add noise according to the model parameters
        if noise_on:
            y += np.random.normal(scale=sigma, size=(my_N, D))
        # Build return structure
        return {'y': y, 's': s}

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
        for key, val in list(my_data.items()):
            my_pdata[key] = val[sel]

        return my_pdata

    def select_Hprimes(self, model_params, data,):
        """
        Return a new data-dictionary which has been annotated with
        a data['candidates'] dataset. A set of self.Hprime candidates
        will be selected.
        """
        my_N, D   = data['y'].shape
        H         = self.H
        SSM        = self.single_state_matrix
        nss = SSM.shape[0]
        candidates= np.zeros((my_N, self.Hprime), dtype=np.int)
        W         = model_params['W'].T
        pi        = model_params['pi']
        sigma     = model_params['sigma']

        # Precompute
        pre1     = -1./2./sigma/sigma
        l_pis=np.zeros((self.H*(self.K-1)))
        c=0
        for i in range(self.K):
            if i ==self.K_0:
                continue
            l_pis[c*H:(c+1)*H]+=np.log(pi[i]) + (self.H-1)*np.log(pi[self.K_0])
            c+=1
            # l_pis+=self.state_abs[i]*pi[i]
        # Allocate return structures
        F = np.empty( [my_N, nss] )
        for n in range(my_N):
            y    = data['y'][n,:]

            Wbar = np.dot(SSM,W)
            log_prod_joint = pre1 * (((Wbar-y)**2).sum(axis=1))
            F[n] = log_prod_joint
            F__= F[n]+l_pis
            sort_prob_ind=np.mod(np.argsort(F__),H)[::-1]
            Fu,Si = np.unique(sort_prob_ind,return_index=True)
            cand = Fu[np.argsort(Si)][:self.Hprime]
            candidates[n]=cand
        data['candidates']=candidates
        return data

    def noisify_params(self, model_params, anneal):
        """ Noisify model params.

        Noisify the given model parameters according to self.noise_policy
        and the annealing object provided. The noise_policy of some model
        parameter PARAM will only be applied if the annealing object
        provides a noise strength via PARAM_noise.

        """
        #H, D = self.H, self.D
        normal = np.random.normal
        uniform = np.random.uniform
        comm = self.comm

        for param, policy in list(self.noise_policy.items()):
            pvalue = model_params[param]
            if (not param+'_noise'=='pi_noise') and anneal[param+"_noise"] != 0.0:
                if np.isscalar(pvalue):         # Param to be noisified is scalar
                    new_pvalue = 0
                    if comm.rank == 0:
                        scale = anneal[param+"_noise"]
                        new_pvalue = pvalue + normal(scale=scale)
                        if new_pvalue < policy[0]:
                            new_pvalue = policy[0]
                        if new_pvalue >= policy[1]:
                            new_pvalue = policy[1]
                        if policy[2]:
                            new_pvalue = np.abs(new_pvalue)
                    pvalue = comm.bcast(new_pvalue)
                else:                                  # Param to be noisified is an ndarray
                    if comm.rank == 0:
                        scale = anneal[param+"_noise"]
                        shape = pvalue.shape
                        new_pvalue = pvalue + normal(scale=scale, size=shape)
                        low_bound, up_bound, absify = policy
                        new_pvalue = np.maximum(low_bound, new_pvalue)
                        new_pvalue = np.minimum( up_bound, new_pvalue)
                        if absify:
                            new_pvalue = np.abs(new_pvalue)
                        pvalue = new_pvalue
                    comm.Bcast([pvalue, MPI.DOUBLE])
            elif param+'_noise'=='pi_noise' and anneal["pi_noise"] != 0.0:
                if comm.rank == 0:
                    scale = anneal[param+"_noise"]
                    shape = pvalue.shape
                    new_pvalue = pvalue + np.random.rand(*shape)*scale
                    new_pvalue = new_pvalue/new_pvalue.sum()
                    pvalue  =   new_pvalue
                comm.Bcast([pvalue, MPI.DOUBLE])

            model_params[param] = pvalue

        return model_params

    def E_step(self, anneal, model_params, my_data):
        """ LinCA E_step

        my_data variables used:

            my_data['y']           Datapoints
            my_data['can']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """
        my_N, D =   my_data['y'].shape
        H       =   self.H
        SM      =   self.state_matrix
        SSM      =   self.single_state_matrix
        W       =   model_params['W'].T
        pi      =   model_params['pi']
        sigma   =   model_params['sigma']
        states  =   self.states
        # Precompute
        beta    =   1./anneal['T']
        pre1    =   -1./2./sigma/sigma
        l       =   len(states)

        l_pis=np.zeros((self.no_states))
        for i in range(l):
            l_pis+=self.state_abs[i]*np.log(pi[i])
        # Allocate return structures
        F       =   np.empty( [my_N, 1 + (self.K-1)*H + self.no_states] )
        pre_F   =   np.empty( [1 + (self.K-1)*H + self.no_states] )
        # Iterate over all datapoints

        ################ Identify Inference Latent vectors##############
        ###########################################################################
        pre_F[0]  =   self.H * np.log(pi[self.K_0])
        c=0
        for state in range(self.K):
            if state == self.K_0:
                continue
            pre_F[c*H+1:(c+1)*H+1]  =   np.log(pi[state]) + ((self.H-1)*np.log(pi[self.K_0]))
            c+=1
        pre_F[(self.K-1)*H+1:]  =   l_pis
        for n in range(my_N):
            y    = my_data['y'][n,:]
            cand = my_data['candidates'][n,:]

            # Handle hidden states with zero active hidden causes
            log_prod_joint = pre1 * (y**2).sum()
            F[n,0] = log_prod_joint

            # Handle hidden states with 1 active cause
            # import ipdb;ipdb.set_trace()
            log_prod_joint = pre1 * ((np.dot(SSM,W)-y)**2).sum(axis=1)
            F[n,1:(self.K-1)*H+1] = log_prod_joint

            if self.gamma>1:
                # Handle hidden states with more than 1 active cause
                W_   = W[cand]                          # is (Hprime x D)

                Wbar = np.dot(SM,W_)
                log_prod_joint = pre1 * (((Wbar-y)**2).sum(axis=1))
                F[n,(self.K-1)*H+1:] = log_prod_joint#+l_pis

        if anneal['anneal_prior']:
            F[:,:] += pre_F[None,:]
            F[:,:] *= beta
        else:
            F[:,:] *=beta
            F[:,:] += pre_F[None,:]
        return { 'logpj': F}#, 'denoms': denoms}

    def M_step(self, anneal, model_params, my_suff_stat, my_data):
        """ LinCA M_step

        my_data variables used:

            my_data['y']           Datapoints
            my_data['candidates']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """
        comm      = self.comm
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

        A_pi_gamma = self.get_scaling_factors(model_params['pi'])
        dlog.append("prior_mass",A_pi_gamma)
        # _, A_pi_gamma, _=self.get_scaling_factors(model_params['pi'])

        #Truncate data
        N_use, my_y,candidates,logpj_all = self._get_sorted_data(N, anneal, A_pi_gamma, all_denoms, candidates,logpj_all,my_y)
        my_N, D = my_y.shape # update my_N

        # Precompute
        corr_all  = logpj_all.max(axis=1)                 # shape: (my_N,)
        pjb_all   = np.exp(logpj_all - corr_all[:, None])  # shape: (my_N, no_states)

        #Log-Likelihood:
        L = self.get_likelihood(D,sigma,A_pi_gamma,logpj_all,N_use)
        dlog.append('L',L)
        # Allocate
        my_Wp     = np.zeros_like(W)   # shape (H, D)
        my_Wq     = np.zeros((self.H,self.H))    # shape (H, H)
        my_pi     = np.zeros_like(pi)  # shape (K)
        my_sigma  = 0.0             #
        SM = self.state_matrix
        SSM = self.single_state_matrix

        # Iterate over all datapoints
        for n in range(my_N):
            y = my_y[n, :]                  # length D
            cand = candidates[n, :]  # length Hprime
            pjb = pjb_all[n, :]
            this_Wp = np.zeros_like(my_Wp)    # numerator for current datapoint (H, D)
            this_Wq = np.zeros_like(my_Wq)    # denominator for current datapoint (H, H)
            this_pi = np.zeros_like(pi)       # numerator for pi update (current datapoint)

            # Handle hidden states with 0 active causes
            this_pi[self.K_0] = self.H*pjb[0]
            this_sigma = pjb[0] * (y**2).sum()

            # Handle hidden states with 1 active cause
            #FIX: I am sure I need to multiply with pi somewhere here
            c=0
            # import ipdb;ipdb.set_trace()
            for state in range(self.K):
                if state == self.K_0:
                    continue
                sspjb = pjb[c*self.H+1:(c+1)*self.H+1]
                # this_Wp  += np.outer(sspjb,y.T)
                # this_Wq  += sspjb[:,None] * SSM[c*self.H:(c+1)*self.H]
                
                this_pi[state]  += sspjb.sum()

                recons = self.states[state]*W
                sqe = ((recons-y)**2).sum(1)
                this_sigma += (sspjb * sqe).sum()

                c+=1
            this_pi[self.K_0]  += ((self.H-1) * pjb[1:(self.K-1)*self.H+1]).sum()
            this_Wp         += np.dot(np.outer(y,pjb[1:(self.K-1)*self.H+1]),SSM).T
            # this_Wq_tmp           = np.zeros_like(my_Wq[cand])
            # this_Wq_tmp[:,cand]   = np.dot(pjb[(self.K-1)*self.H+1:] * SM.T,SM)
            this_Wq         += np.dot(pjb[1:(self.K-1)*self.H+1] * SSM.T, SSM)



            if self.gamma>1:
                # Handle hidden states with more than 1 active cause
                this_Wp[cand]         += np.dot(np.outer(y,pjb[(self.K-1)*self.H+1:]),SM).T
                this_Wq_tmp           = np.zeros_like(my_Wq[cand])
                this_Wq_tmp[:,cand]   = np.dot(pjb[(self.K-1)*self.H+1:] * SM.T,SM)
                this_Wq[cand]         += this_Wq_tmp

                this_pi += np.inner(pjb[(self.K-1)*self.H+1:], self.state_abs)

                W_ = W[cand]                           # is (Hprime x D)
                Wbar = np.dot(SM,W_)
                this_sigma += (pjb[(self.K-1)*self.H+1:] * ((Wbar-y)**2).sum(axis=1)).sum()
            #Scale down
            denom = pjb.sum()
            my_Wp += this_Wp / denom
            my_Wq += this_Wq / denom

            my_pi += this_pi / denom

            my_sigma += this_sigma/ denom/D

        #Calculate updated W
        Wp = np.empty_like(my_Wp)
        Wq = np.empty_like(my_Wq)
        comm.Allreduce( [my_Wp, MPI.DOUBLE], [Wp, MPI.DOUBLE] )
        comm.Allreduce( [my_Wq, MPI.DOUBLE], [Wq, MPI.DOUBLE] )
        # W_new  = np.dot(np.linalg.pinv(Wq), Wp)
        rcond = -1
        if float(np.__version__[2:]) >= 14.0:
            rcond = None
        W_new  = np.linalg.lstsq(Wq, Wp, rcond=rcond)[0]    # TODO check and switch to this one

        # Calculate updated pi
        pi_new=np.empty_like(pi)
        # pi_new = E_pi_gamma * comm.allreduce(my_pi) / H / N_use
        for i in range(self.K):
            pi_new[i]  = comm.allreduce(my_pi[i])/comm.allreduce(my_pi.sum())

        eps = 1e-6
        if np.any(pi_new<eps):
            which_lo = pi_new<eps
            which_hi = pi_new>=eps
            pi_new[which_lo] += eps - pi_new[which_lo]
            pi_new[which_hi] -= (eps*np.sum(which_lo))/np.sum(which_hi)

        if 'penalty' in list(self.__dict__.keys()):
            self.penalty
            if self.penalty>pi_new[self.K_0]:
                r = (1-self.penalty)/(1-pi_new[self.K_0])
                pi_new[pi_new!=0] = pi_new[pi_new!=0]*r
                pi_new[self.K_0] = self.penalty
                pi_new/=pi_new.sum()

        # Calculate updated sigma
        sigma_new = np.sqrt(comm.allreduce(my_sigma) /  N_use)

        if 'W' not in self.to_learn:
            W_new = W
        if 'pi' not in self.to_learn:
            pi_new = pi
        if 'sigma' not in self.to_learn:
            sigma_new = sigma

        for param in anneal.crit_params:
            exec('this_param = ' + param)
            anneal.dyn_param(param, this_param)

        dlog.append('N_use', N_use)

        return { 'W': W_new.transpose(), 'pi': pi_new, 'sigma': sigma_new, 'Q': 0.}

    def calculate_respons(self, anneal, model_params, data):
        data['candidates'].sort(axis=1) #(we do this to set the order back=outside)
        F_JB = self.E_step(anneal, model_params, data)['logpj']
        #Transform into responsibilities

        corr = np.max(F_JB, axis=1)
        exp_F_JB_corr = np.exp(F_JB - corr[:, None])
        respons = exp_F_JB_corr/(np.sum(exp_F_JB_corr, axis=1).reshape(-1, 1))
        return respons

    def free_energy(self, model_params, my_data):

        return 0.0

    def gain(self, old_parameters, new_parameters):

        return 0.0

    def get_scaling_factors(self,pi):
        # Precompute factor for pi update
        A_pi_gamma = 0.0
        # B_pi_gamma = np.zeros_like(pi)

        ar_gamma=np.arange(self.gamma+1)
        # iterator over all possible values of gamma_{k}
        p=itls.product(ar_gamma,repeat=len(self.states)-1)

        for gp in p:
            # gp: (tuple) holds (g_1,g_2,...,g_k)
            ngp=np.array(gp)
            if ngp.sum()>self.gamma:
                continue
            num0 = self.H-ngp.sum() #number of zeros for combinations gp, i.e. H-g_1-g_2-...-g_k
            abs_array = np.insert(ngp,self.K_0, num0)
            if not abs_array.sum() == self.H :
                raise Exception("wrong number of elements counted")
            cmb = multinom2(abs_array.sum(),abs_array)
            pm = np.prod(pi**abs_array)
            # pm2 = np.exp(np.sum(np.log(pi)*abs_array))
            A_pi_gamma += cmb * pm
            # B_pi_gamma += np.prod(abs_array * cmb * (pi**abs_array))
        # E_pi_gamma = pi * self.H * A_pi_gamma / B_pi_gamma
        # return E_pi_gamma, A_pi_gamma, B_pi_gamma
        return A_pi_gamma

    def _get_sorted_data(self,N, anneal, A_pi_gamma, all_denoms, candidates,logpj_all,my_y):
        comm = self.comm
        # if False:
        if anneal['Ncut_factor'] > 0.0:

            N_use = int(N * (1 - (1 - A_pi_gamma) * anneal['Ncut_factor']))
            cut_denom = parallel.allsort(all_denoms)[-N_use]
            which   = np.array(all_denoms > cut_denom)
            # which   = np.array(all_denoms >= cut_denom)
            candidates = candidates[which]
            logpj_all = logpj_all[which]
            my_y    = my_y[which]
            my_N, D = my_y.shape
            N_use = comm.allreduce(my_N)
            # N_use = N
        else:
            N_use = N

        return N_use, my_y,candidates,logpj_all

    def get_likelihood(self,D,sigma,A,logpj_all,N):
        comm = self.comm
        L =  - 0.5 * D * np.log(2*np.pi*sigma**2)#-np.log(A)
        Fs = logsumexp(logpj_all,1).sum()
        L += comm.allreduce(Fs)/N
        return L

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
        W_mean = parallel.allmean(my_y, axis=0, comm=comm)     # shape: (D, )

        # Calculate data variance
        # import ipdb;ipdb.set_trace()  ######### Break Point ###########

        sigma_sq = parallel.allmean((my_y - W_mean)**2, axis=0, comm=comm)  # shape: (D, )
        sigma_init = np.sqrt(sigma_sq).sum()/D                         # scalar

        # Initial W
        noise = sigma_init / 4.
        # noise = sigma_init * 4.
        # shape: (H, D)`
        # W_init = W_mean[:, None] + np.random.laplace(scale=noise, size=[D, H])
        W_init = W_mean[:, None] + np.random.normal(scale=noise, size=[D, H])

        sparsity = 1. - (1./H) 
        pi_init = np.random.rand(self.K - 1)
        pi_init = (1 - sparsity) * pi_init / pi_init.sum()
        pi_init = np.insert(pi_init, self.K_0, sparsity)

        # Create and set Model Parameters, W columns have the same average!
        model_params = {
            'W': W_init,
            'pi': pi_init,
            'sigma': sigma_init
        }

        return model_params

    def inference(self, anneal, model_params, test_data, topK=10, logprob=False, adaptive=True,
        Hprime_max=None, gamma_max=None):
        W = model_params['W']
        my_y = my_data['y']
        D, H = W.shape
        my_N, D = my_y.shape

        # Prepare return structure
        res = {
            's': np.zeros( (my_N, topK, H), dtype=np.int8),
            'm': np.zeros( (my_N, H) ),
            'p': np.zeros( (my_N, topK) ),
            'gamma': np.zeros( (my_N,) ),
            'Hprime': np.zeros( (my_N,) )
        }

        if 'candidates' not in my_data:
            my_data = self.select_Hprimes(model_params, my_data)
            my_cand = my_data['candidates']

        my_suff_stat = self.E_step(anneal, model_params, my_data)
        my_logpj = my_suff_stat['logpj']
        my_corr = my_logpj.max(axis=1)           # shape: (my_N,)
        my_logpjc = my_logpj - my_corr[:, None]    # shape: (my_N, no_states)
        my_pjc = np.exp(my_logpjc)              # shape: (my_N, no_states)
        my_denomc = my_pjc.sum(axis=1)             # shape: (my_N)

        idx = np.argsort(my_logpjc, axis=-1)[:, ::-1]
        for n in range(my_N):                                   # XXX Vectorize XXX
            for m in range(topK):
                this_idx = idx[n, m]
                # res['p'][n, m] = my_pjc[n, this_idx] / my_denomc[n]
                if logprob:
                    res['p'][n_,m] = my_logpjc[n, this_idx] - np.log(my_denomc[n])
                else:
                    res['p'][n_,m] = my_pjc[n, this_idx] / my_denomc[n]
                if this_idx == 0:
                    pass
                elif this_idx < ((self.K-1)*H + 1):
                    s_prime = self.single_state_matrix[this_idx-1,:]
                    res['s'][n, m, :] = s_prime
                else:
                    s_prime = self.state_matrix[this_idx - (self.K-1)*H - 1,:]
                    res['s'][n, m, my_cand[n, :]] = s_prime

        if not logprob:
            res['m'] = np.exp(res['m'])

        return res
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
                    elif this_idx < ((self.K-1)*H + 1):
                        s_prime = self.single_state_matrix[this_idx-1,:]

                        res['s'][n_,m,(this_idx-1)%H] = s_prime[(this_idx-1)%H]
                    else:
                        s_prime = self.state_matrix[this_idx - (self.K-1)*H - 1]                        
                        res['s'][n_,m,my_cand[n,:]] = s_prime

                for h in range(H):
                    if h in my_cand[n,:]:
                        idx_ = np.where(my_cand[n]==h)[0][0]
                        logp = np.hstack([my_logpjc[n,h+1],my_logpjc[n,(self.K-1)*H+1:][self.state_matrix[:,idx_]==1]])
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

        return res

