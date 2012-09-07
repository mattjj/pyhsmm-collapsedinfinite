from __future__ import division
import numpy as np
na = np.newaxis
import collections, itertools
import abc

from pyhsmm.util.stats import sample_discrete, sample_discrete_from_log, combinedata
from pyhsmm.util.general import rle as rle

# NOTE: assumes censoring. can make no censoring by adding to score of last
# segment

SAMPLING = -1 # special constant for indicating a state or state range that is being resampled
NEW = -2 # special constant indicating a potentially new label
ABIGNUMBER = 10000 # state labels are sampled uniformly from 0 to abignumber exclusive

####################
#  States Classes  #
####################

# TODO an array class that maintains its own rle
# must override set methods
# type(x).__setitem__(x,i) classmethod
# also has members norep and lens (or something)
# that are either read-only or also override setters
# for now, i'll just make sure outside that anything that sets self.stateseq
# also sets self.stateseq_norep and self.durations
# it should also call beta updates...

class collapsed_states(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def resample(self):
        pass

    @abc.abstractmethod
    def _counts_from(self,k):
        pass

    @abc.abstractmethod
    def _counts_to(self,k):
        pass

    @abc.abstractmethod
    def _counts_fromto(self,k):
        pass

    def _new_label(self,ks):
        assert SAMPLING not in ks
        newlabel = np.random.randint(ABIGNUMBER)
        while newlabel in ks:
            newlabel = np.random.randint(ABIGNUMBER)
        newweight = self.beta.betavec[newlabel] # instantiate, needed if new state at beginning of seq
        return newlabel

    def _data_withlabel(self,k):
        assert k != SAMPLING
        return self.data[self.stateseq == k]

    def _occupied(self):
        return set(self.stateseq) - set((SAMPLING,))

    def plot(self,colors_dict):
        from matplotlib import pyplot as plt
        stateseq_norep, durations = rle(self.stateseq)
        X,Y = np.meshgrid(np.hstack((0,durations.cumsum())),(0,1))

        if colors_dict is not None:
            C = np.array([[colors_dict[state] for state in stateseq_norep]])
        else:
            C = stateseq_norep[na,:]

        plt.pcolor(X,Y,C,vmin=0,vmax=1)
        plt.ylim((0,1))
        plt.xlim((0,len(self.stateseq)))
        plt.yticks([])


class collapsed_stickyhdphmm_states(collapsed_states):
    def __init__(self,model,beta,alpha_0,kappa,obs,data=None,T=None,stateseq=None):
        self.alpha_0 = alpha_0
        self.kappa = kappa

        self.model = model
        self.beta = beta
        self.obs = obs

        self.data = data

        if (data,stateseq) == (None,None):
            # generating
            assert T is not None, 'must pass in T when generating'
            self._generate(T)
        elif data is None:
            self.T = stateseq.shape[0]
            self.stateseq = stateseq
        elif stateseq is None:
            self.data = data
            self._generate(data.shape[0])
        else:
            assert data.shape[0] == stateseq.shape[0]
            self.stateseq = stateseq
            self.data = data
            self.T = data.shape[0]

    def _generate(self,T):
        self.T = T
        alpha, kappa = self.alpha_0, self.kappa
        betavec = self.beta.betavec
        stateseq = np.zeros(T,dtype=np.int)
        model = self.model
        self.stateseq = stateseq[:0]

        # NOTE: we have a choice of what state to start in; it's just a
        # definition choice that isn't specified in the HDP-HMM
        # Here, we choose just to sample from beta. Note that if this is the
        # first chain being sampled in this model, this will always sample
        # zero, since no states will be occupied.
        ks = list(model._occupied()) + [None]
        firststate = sample_discrete(np.arange(len(ks)))
        if firststate == len(ks)-1:
            stateseq[0] = self._new_label(ks)
        else:
            stateseq[0] = ks[firststate]

        # runs a CRF with fixed weights beta forwards
        for t in range(1,T):
            self.stateseq = stateseq[:t]
            ks = list(model._occupied() | self._occupied())
            betarest = 1-sum(betavec[k] for k in ks)
            # get the counts of new states coming out of our current state
            # going to all other states
            fromto_counts = np.array([model._counts_fromto(stateseq[t-1],k)
                                            + self._counts_fromto(stateseq[t-1],k)
                                            for k in ks])
            # for those states plus a new one, sample proportional to
            scores = np.array([(alpha*betavec[k] + (kappa if k == stateseq[t+1] else 0) + ft)
                    for k,ft in zip(ks,fromto_counts)] + [alpha*betarest])
            nextstateidx = sample_discrete(scores)
            if nextstateidx == scores.shape[0]-1:
                stateseq[t] = self._new_label(ks)
            else:
                stateseq[t] = ks[nextstateidx]
        self.stateseq = stateseq

    def resample(self):
        model = self.model

        for t in np.random.permutation(self.T):
            # throw out old value
            self.stateseq[t] = SAMPLING
            ks = list(model._occupied())
            self.beta.housekeeping(ks)

            # form the scores and sample from them
            scores = np.array([self._score(k,t) for k in ks]+[self._new_score(ks,t)])
            idx = sample_discrete_from_log(scores)

            # set the state
            if idx == scores.shape[0]-1:
                self.stateseq[t] = self._new_label(ks)
            else:
                self.stateseq[t] = ks[idx]

    def _score(self,k,t):
        alpha, kappa = self.alpha_0, self.kappa
        betavec, model, o = self.beta.betavec, self.model, self.obs
        data, stateseq = self.data, self.stateseq

        score = 0

        # left transition score
        if t > 0:
            score += np.log( (alpha*betavec[k] + (kappa if k == stateseq[t-1] else 0)
                                + model._counts_fromto(stateseq[t-1],k))
                                / (alpha+kappa+model._counts_from(stateseq[t-1])) )

        # right transition score
        if t < self.T - 1:
            # indicators since we may need to include the left transition in
            # counts (since we are scoring exchangeably, not independently)
            another_from = 1 if t > 0 and stateseq[t-1] == k else 0
            another_fromto = 1 if (t > 0 and stateseq[t-1] == k and stateseq[t+1] == k) else 0

            score += np.log( (alpha*betavec[stateseq[t+1]] + (kappa if k == stateseq[t+1] else 0)
                                + model._counts_fromto(k,stateseq[t+1]) + another_fromto)
                                / (alpha+kappa+model._counts_from(k) + another_from) )

        # observation score
        score += o.log_predictive(data[t],model._data_withlabel(k))

        return score

    def _new_score(self,ks,t):
        alpha, kappa = self.alpha_0, self.kappa
        betavec, model, o = self.beta.betavec, self.model, self.obs
        data, stateseq = self.data, self.stateseq

        score = 0

        # left transition score
        if t > 0:
            betarest = 1-sum(betavec[k] for k in ks)
            score += np.log(alpha*betarest/(alpha+kappa+model._counts_from(stateseq[t-1])))

        # right transition score
        if t < self.T-1:
            score += np.log(betavec[stateseq[t+1]])

        # observation score
        score += o.log_marginal_likelihood(data[t])

        return score

    def _counts_from(self,k):
        assert k != SAMPLING
        assert np.sum(self.stateseq == SAMPLING) in (0,1)
        temp = np.sum(self.stateseq[:-1] == k)
        if SAMPLING in self.stateseq[1:] and \
                self.stateseq[np.where(self.stateseq == SAMPLING)[0]-1] == k:
            temp -= 1
        return temp

    def _counts_to(self,k):
        assert k != SAMPLING
        assert np.sum(self.stateseq == SAMPLING) in (0,1)
        temp = np.sum(self.stateseq[1:] == k)
        if SAMPLING in self.stateseq[:-1] and \
                self.stateseq[np.where(self.stateseq == SAMPLING)[0]+1] == k:
            temp -= 1
        return temp

    def _counts_fromto(self,k1,k2):
        assert k1 != SAMPLING and k2 != SAMPLING
        if k1 not in self.stateseq or k2 not in self.stateseq:
            return 0
        else:
            from_indices, = np.where(self.stateseq[:-1] == k1) # EXCEPT last
            return np.sum(self.stateseq[from_indices+1] == k2)


class collapsed_hdphsmm_states(collapsed_states):
    def __init__(self,model,beta,alpha_0,obs,dur,data=None,T=None,stateseq=None):
        self.alpha_0 = alpha_0

        self.model = model
        self.beta = beta
        self.obs = obs
        self.dur = dur

        self.data = data

        if (data,stateseq) == (None,None):
            # generating
            assert T is not None, 'must pass in T when generating'
            self._generate(T)
        elif data is None:
            self.T = stateseq.shape[0]
            self.stateseq = stateseq
        elif stateseq is None:
            self.data = data
            # self._generate(data.shape[0]) # initialized from the prior
            # self.stateseq = self.stateseq[:self.T]
            self.stateseq = np.random.randint(25,size=data.shape[0])
            self.T = data.shape[0]
        else:
            assert data.shape[0] == stateseq.shape[0]
            self.stateseq = stateseq
            self.stateseq_norep, self.durations = rle(stateseq)
            self.data = data
            self.T = data.shape[0]

    def _generate(self,T):
        alpha = self.alpha_0
        betavec = self.beta.betavec
        model = self.model
        self.stateseq = np.array([])

        ks = list(model._occupied()) + [None]
        firststateidx = sample_discrete(np.arange(len(ks)))
        if firststateidx == len(ks)-1:
            firststate = self._new_label(ks)
        else:
            firststate = ks[firststateidx]

        self.dur.resample(combinedata((model._durs_withlabel(firststate),self._durs_withlabel(firststate))))
        firststate_dur = self.dur.rvs()

        self.stateseq = np.ones(firststate_dur,dtype=int)*firststate
        t = firststate_dur

        # run a family-CRF (CRF with durations) forwards
        while t < T:
            ks = list(model._occupied() | self._occupied())
            betarest = 1-sum(betavec[k] for k in ks)
            fromto_counts = np.array([model._counts_fromto(self.stateseq[t-1],k)
                                            + self._counts_fromto(self.stateseq[t-1],k)
                                            for k in ks])
            scores = np.array([(alpha*betavec[k] + ft if k != self.stateseq[t-1] else 0)
                    for k,ft in zip(ks,fromto_counts)]
                    + [alpha*(1-betavec[self.stateseq[t-1]])*betarest])
            nextstateidx = sample_discrete(scores)
            if nextstateidx == scores.shape[0]-1:
                nextstate = self._new_label(ks)
            else:
                nextstate = ks[nextstateidx]

            # now get the duration of nextstate!
            self.dur.resample(combinedata((model._durs_withlabel(nextstate),self._durs_withlabel(nextstate))))
            nextstate_dur = self.dur.rvs()

            self.stateseq = np.concatenate((self.stateseq,np.ones(nextstate_dur,dtype=int)*nextstate))

            t += nextstate_dur

        self.T = len(self.stateseq)

    def resample(self):
        self.resample_label_version()

    def _durs_withlabel(self,k):
        assert k != SAMPLING
        if len(self.stateseq) > 0:
            stateseq_norep, durations = rle(self.stateseq)
            return durations[stateseq_norep == k]
        else:
            return []

    def _counts_fromto(self,k1,k2):
        assert k1 != SAMPLING and k2 != SAMPLING
        if k1 not in self.stateseq or k2 not in self.stateseq or k1 == k2:
            return 0
        else:
            stateseq_norep, _ = rle(self.stateseq)
            from_indices, = np.where(stateseq_norep[:-1] == k1) # EXCEPT last
            return np.sum(stateseq_norep[from_indices+1] == k2)

    def _counts_from(self,k):
        assert k != SAMPLING
        stateseq_norep, _ = rle(self.stateseq)
        temp = np.sum(stateseq_norep[:-1] == k)
        if SAMPLING in stateseq_norep[1:] and \
                stateseq_norep[np.where(stateseq_norep == SAMPLING)[0]-1] == k:
            temp -= 1
        return temp

    def _counts_to(self,k):
        assert k != SAMPLING
        stateseq_norep, _ = rle(self.stateseq)
        temp = np.sum(stateseq_norep[1:] == k)
        if SAMPLING in stateseq_norep[:-1] and \
                stateseq_norep[np.where(stateseq_norep == SAMPLING)[0]+1] == k:
            temp -= 1
        return temp

    ### label sampler stuff

    def resample_label_version(self):
        # NOTE never changes first label: we assume the initial state
        # distribution is a delta at that label
        for t in (np.random.permutation(self.T-1)+1):
            self.stateseq[t] = SAMPLING
            ks = self.model._occupied()
            self.beta.housekeeping(ks)
            ks = list(ks)

            # sample a new value
            scores = np.array([self._label_score(t,k) for k in ks] + [self._new_label_score(t,ks)])
            newlabelidx = sample_discrete_from_log(scores)
            if newlabelidx == scores.shape[0]-1:
                self.stateseq[t] = self._new_label(ks)
            else:
                self.stateseq[t] = ks[newlabelidx]

    def _label_score(self,t,k):
        assert t > 0

        score = 0.

        # unpack variables
        model = self.model
        alpha = self.alpha_0
        beta = self.beta.betavec
        stateseq = self.stateseq
        obs, durs = self.obs, self.dur

        # left transition (if there is one)
        if stateseq[t-1] != k:
            score += np.log(alpha * beta[k] + model._counts_fromto(stateseq[t-1],k)) \
                    - np.log(alpha * (1-beta[stateseq[t-1]]) + model._counts_from(stateseq[t-1]))

        # right transition (if there is one)
        if t < self.T-1 and stateseq[t+1] != k:
            score += np.log(alpha * beta[stateseq[t+1]] + model._counts_fromto(k,stateseq[t+1])) \
                    - np.log(alpha * (1-beta[k]) + model._counts_from(k))

        # predictive likelihoods
        for (data,otherdata), (dur,otherdurs) in self._local_group(t,k):
            score += obs.log_predictive(data,otherdata) + durs.log_predictive(dur,otherdurs)

        return score

    def _new_label_score(self,t,ks):
        assert t > 0

        score = 0.

        # unpack
        model = self.model
        alpha = self.alpha_0
        beta = self.beta.betavec
        stateseq = self.stateseq
        obs, durs = self.obs, self.dur

        # left transition (only from counts, no to counts)
        score += np.log(alpha) - np.log(alpha*(1.-beta[stateseq[t-1]])
                        + model._counts_from(stateseq[t-1]))

        # add in right transition (no counts)
        if t < self.T-1:
            score += np.log(beta[stateseq[t+1]])

        # add in sum over k factor
        if t < self.T-1:
            betas = np.random.beta(1,self.beta.gamma_0,size=200)
            betas[1:] *= (1-betas[:-1]).cumprod()
            score += np.log(self.beta.remaining*(betas/(1-betas)).sum())
        else:
            score += np.log(self.beta.remaining)

        # add in obs/dur scores of local pieces
        for (data,otherdata), (dur,otherdurs) in self._local_group(t,NEW):
            score += obs.log_predictive(data,otherdata) + durs.log_predictive(dur,otherdurs)

        return score

    def _local_group(self,t,k):
        '''
        returns a sequence of length between 1 and 3, where each sequence element is
        ((data,otherdata), (dur,otherdurs))
        '''
        # temporarily modifies members, like self.stateseq and maybe self.data
        assert self.stateseq[t] == SAMPLING
        orig_stateseq = self.stateseq.copy()

        # temporarily set stateseq to hypothetical stateseq
        # so that we can get the indicator sequence
        # TODO if i write the special stateseq class, this will need fixing
        self.stateseq[t] = k
        wholegroup, pieces = self._local_slices(self.stateseq,t)
        self.stateseq[t] = SAMPLING

        # build local group of statistics
        localgroup = []
        self.stateseq[wholegroup] = SAMPLING
        for piece, val in pieces:
            # get all the other data
            otherdata, otherdurs = self.model._data_withlabel(val), self.model._durs_withlabel(val)

            # add a piece to our localgroup
            localgroup.append(((self.data[piece],otherdata),(piece.stop-piece.start,otherdurs)))

            # remove the used piece from the exclusion
            self.stateseq[piece] = orig_stateseq[piece]

        # restore original views
        self.stateseq = orig_stateseq

        # return
        return localgroup

    @classmethod
    def _local_slices(cls,stateseq,t):
        '''
        returns slices: wholegroup, (piece1, ...)
        '''
        A,B = fill(stateseq,t-1), fill(stateseq,t+1)
        if A == B:
            return A, ((A,stateseq[A.start]),)
        elif A.start <= t < A.stop or B.start <= t < B.stop:
            return slice(A.start,B.stop), [(x,stateseq[x.start]) for x in (A,B) if x.stop - x.start > 0]
        else:
            It = slice(t,t+1)
            return slice(A.start,B.stop), [(x,stateseq[x.start]) for x in (A,It,B) if x.stop - x.start > 0]


#######################
#  Utility Functions  #
#######################

def fill(seq,t):
    if t < 0:
        return slice(0,0)
    elif t > seq.shape[0]-1:
        return slice(seq.shape[0],seq.shape[0])
    else:
        endindices, = np.where(np.diff(seq) != 0) # internal end indices (not incl -1 and T-1)
        startindices = np.concatenate(((0,),endindices+1,(seq.shape[0],))) # incl 0 and T
        idx = np.where(startindices <= t)[0][-1]
        return slice(startindices[idx],startindices[idx+1])

def canonize(seq):
    seq = seq.copy()
    canondict = collections.defaultdict(itertools.count().next)
    for idx,s in enumerate(seq):
        seq[idx] = canondict[s]
    reversedict = {}
    for k,v in canondict.iteritems():
        reversedict[v] = k
    return seq, canondict, reversedict

class dummytrans(object):
    def __init__(self,A):
        self.A = A

    def resample(self,*args,**kwargs):
        pass
