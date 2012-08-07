from __future__ import division
import numpy as np
na = np.newaxis
import numpy.ma as ma

from pyhsmm.util.stats import sample_discrete, sample_discrete_from_log
from pyhsmm.util.general import rle

####################
#  States Classes  #
####################

class collapsed_stickyhdphmm_states(object):
    # TODO can reuse all this from (non-sticky) hdphmm class if i set
    # self.alpha to be alpha+kappa, though would need to wrap self-transitions
    # somehow...
    def __init__(self,model,betavec,alpha_0,kappa,obs,data=None,T=None,stateseq=None):
        self.alpha_0 = alpha_0

        self.model = model
        self.betavec = betavec
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
        self.stateseq = stateseq = np.empty(T,dtype=np.int)
        model = self.model

        # NOTE: we have a choice of what state to start in; it's just a
        # definition choice that isn't specified in the HDP-HMM
        # Here, we choose just to sample from beta. Note that if this is the
        # first chain being sampled in this model, this will always sample
        # zero, since no states will be occupied.
        ks = model._get_occupied()
        betarest = 1-sum(betavec[k] for k in ks)
        scores = np.array([betavec[k] for k in ks] + [betarest])
        firststate = sample_discrete(scores)
        if firststate == scores.shape[0]-1:
            stateseq[0] = self._new_label(ks)
        else:
            stateseq[0] = ks[firststate]

        # runs a CRF with fixed weights beta forwards
        for t in range(1,T):
            state = stateseq[t-1]
            ks = model._get_occupied()
            betarest = 1-sum(betavec[k] for k in ks)
            # get the counts of new states coming out of our current state
            # going to all other states
            fromto_counts = np.array([model._get_counts_fromto(state,k) for k in ks])
            total_from = fromto_counts.sum()
            # for those states plus a new one, sample proportional to
            # ((alpha+kappa)*beta + fromto) / (alpha+kappa+totalfrom)
            scores = np.array([((alpha+kappa)*betavec[k] + ft)/(alpha+kappa+total_from)
                    for ft in fromto_counts] + [((alpha+kappa)*betarest)/(alpha+kappa+total_from)])
            nextstate = sample_discrete(scores)
            if nextstate == scores.shape[0]-1:
                stateseq[t] = self._new_label(ks)
            else:
                stateseq[t] = ks[nextstate]

    def resample(self):
        model, alpha, betavec = self.model, self.alpha_0, self.betavec
        self.z = ma.masked_array(self.z,mask=np.zeros(self.z.shape))

        for t in np.random.permutation(self.T):
            # throw out old value
            self.z.mask[t] = True

            # form the scores and sample from them
            ks = list(model._get_occupied())
            scores = np.array([self._get_score(k,t) for k in ks]+[self._get_new_score(ks,t)])
            idx = sample_discrete_from_log(scores)

            # set the state
            if idx == scores.shape[0]-1:
                self.z[t] = self._new_label(ks)
            else:
                self.z[t] = ks[idx] # resets the mask

    def _get_score(self,k,t):
        alpha, kappa = self.alpha_0, self.kappa
        betavec, model, o = self.betavec, self.model, self.obs
        data, stateseq = self.data, self.stateseq

        score = 0

        # left transition score
        if t > 0:
            b = betavec[k]
            if stateseq[t-1] == k:
                b += kappa
            score += np.log( ((alpha+kappa)*b + model._get_counts_fromto(stateseq[t-1],k)) \
                    / (alpha+kappa+model._get_counts_from(stateseq[t-1])))

        # right transition score
        if t < self.T - 1:
            b = betavec[stateseq[t+1]]
            if stateseq[t+1] == k:
                b += kappa

            # indicators since we may need to include the left transition in
            # counts (since we are scoring exchangeably, not independently)
            if t > 0:
                another_from = 1 if stateseq[t-1] == k else 0
                another_fromto = 1 if stateseq[t-1] == k and stateseq[t+1] == k else 0

            score += np.log( ((alpha+kappa)*b + model._get_counts_fromto(k,stateseq[t+1]) + another_fromto) \
                    / (alpha+kappa+model._counts_from(k) + another_from))

        # observation score
        score += o.log_predictive(data[t],model._get_data_withlabel(k))

        return score

    def _get_new_score(self,ks,t):
        alpha, kappa = self.alpha_0, self.kappa
        betavec, model, o = self.betavec, self.model, self.obs
        data, stateseq = self.data, self.stateseq

        score = 0

        # left transition score
        if t > 0:
            betarest = 1-sum(betavec[k] for k in ks)
            score += np.log((alpha+kappa)*betarest
                    /(alpha+kappa+model._get_counts_from(stateseq[t-1])))

        # right transition score
        if t < self.T-1:
            score += np.log(betavec[stateseq[t+1]])

        # observation score
        score += o.log_marginal_likelihood(data[t])

    def _new_label(self,ks):
        # return a label that isn't already used
        newlabel = np.random.randint(low=0,high=5*max(ks))
        while newlabel in ks:
            newlabel = np.random.randint(low=0,high=5*max(ks))
        return newlabel

    # masking self.z in resample() makes the _get methods work

    def _get_counts_from(self,k):
        return np.sum(self.stateseq[:-1] == k) # except last!

    def _get_counts_fromto(self,k1,k2):
        if k1 not in self.stateseq:
            return 0
        else:
            from_indices, = np.where(self.stateseq[:-1] == k1) # EXCEPT last
            return np.sum(self.stateseq[from_indices+1] == k2)

    def _get_data_withlabel(self,k):
        return self.data[self.stateseq == k]

    def _get_occupied(self):
        return set(self.stateseq)


# TODO TODO below here

class collapsed_hdphsmm_states(object): # TODO this class is broken!
    def __init__(self,model,betavec,alpha,obsclass,durclass,data=None,T=None,stateseq=None):
        raise NotImplementedError, 'this class is currently borked'
        self.alpha = alpha

        self.model = model
        self.betavec = betavec # infinite vector
        self.obsclass = obsclass
        self.durclass = durclass

        self.data = data

        if (data,stateseq) == (None,None):
            # generating
            assert T is not None, 'must pass in T when generating'
            self.T = T
            self.generate()
        elif data is None:
            self.T = stateseq.shape[0]
            self.stateseq = stateseq
            self.generate_obs()
        elif stateseq is None:
            self.T = data.shape[0]
            self.data = data
            self.generate_states()
        else:
            assert data.shape[0] == stateseq.shape[0]
            self.stateseq = stateseq
            self.data = data
            self.T = data.shape[0]

    def resample(self):
        self.resample_segment_version()

    def _get_counts_from(self,k):
        return self.__get_counts_from(self.stateseq,k)

    def _get_counts_fromto(self,k1,k2):
        return self.__get_counts_fromto(self.stateseq,k1,k2)

    def _get_data_withlabel(self,k):
        return self.__get_data_withlabel(self.stateseq,self.data,k)

    def _get_durs_withlabel(self,k):
        return self.__get_durs_withlabel(self.stateseq,k)

    @classmethod
    def __get_counts_from(cls,stateseq,k):
        stateseq_norep = get_norep(stateseq)[:-1] # EXCEPT last
        rawcount = np.sum(stateseq_norep == k)
        if SAMPLING in stateseq_norep:
            sampling_index, = np.where(stateseq_norep == SAMPLING)
            if sampling_index > 0 and stateseq[sampling_index-1] == k:
                return rawcount - 1
        return rawcount

    @classmethod
    def __get_counts_fromto(cls,stateseq,k1,k2):
        if k1 not in stateseq:
            return 0
        else:
            stateseq_norep = get_norep(stateseq)
            from_indices, = np.where(stateseq_norep[:-1] == k1) # EXCEPT last
            return np.sum(stateseq_norep[from_indices + 1] == k2)

    @classmethod
    def __get_data_withlabel(cls,stateseq,data,k):
        return data[stateseq == k]

    @classmethod
    def __get_durs_withlabel(cls,stateseq,k):
        stateseq_norep, durs = rle(stateseq)
        return durs[stateseq_norep == k]

    def _get_occupied(self,stateseq):
        return set(self.stateseq)

    def _new_label(self,ks): # TODO this should be in transitions
        # sample beta conditioned on finding a new label
        beta = self.betavec

        betanew = 1.-sum(beta[k] for k in ks)

        i = -1 # incremented at least once
        prob = np.random.rand() * betanew
        while prob > 0:
            if i not in ks:
                prob -= self.betavec[i]
            i += 1

        return i

    ### label sampler stuff

    def resample_label_version(self):
        for t in np.random.permutation(self.T):
            # throw out old value (flag used in count methods)
            self.stateseq[t] = SAMPLING

            # sample a new value
            ks = self.model._get_occupied()
            scores = [self._label_score(t,k) for k in ks] + [self._new_label_score(t,ks)]
            labels = ks + [NEW]
            newlabel = labels[sample_discrete(scores)]
            if newlabel is not NEW:
                self.stateseq[t] = newlabel
            else:
                self.stateseq[t] = self._new_label(ks)


    def _label_score(self,t,k):
        score = 1.

        # unpack variables
        model = self.model
        alpha = self.alpha
        beta = self.betavec
        stateseq = self.stateseq
        obs, durs = self.obsclass, self.durclass

        # left transition
        if t > 0 and stateseq[t-1] != k:
            score *= (alpha * beta[k] + model.counts_fromto(stateseq[t-1],k)) / \
                    (alpha * (1-beta[stateseq[t-1]]) + model.counts_from(stateseq[t-1]))

        # right transition
        if t < self.T-1 and stateseq[t+1] != k:
            score *= (alpha * beta[stateseq[t+1]] + model.counts_fromto(k,stateseq[t+1])) / \
                    (alpha * (1-beta[k]) + model.counts_from(k))

        # predictive likelihoods
        for (data,otherdata), (dur,otherdurs) in self.__get_local_group(t,k):
            score *= obs.predictive(data,otherdata) * durs.predictive(dur,otherdurs)

        return score

    def _new_label_score(self,t,ks):
        # we know there won't be any merges
        score = 1.

        # unpack
        model = self.model
        alpha = self.alpha
        beta = self.betavec
        stateseq = self.stateseq
        obs, durs = self.obsclass, self.durclass

        # compute betanew (aka betarest), this line is the main reason this is a
        # separate method from _label_score
        betanew = 1.-sum(beta[k] for k in ks)

        # left transition (only from counts)
        if t > 0:
            score *= alpha * betanew / \
                    (alpha * (1.-beta[stateseq[t-1]]) + model.counts_from(stateseq[t-1]))

        # add in right transition (no counts)
        if t < self.T-1:
            score *= beta[stateseq[t+1]] / (1.-betanew)

        # add in obs/dur scores of local pieces
        for (data,otherdata), (dur,otherdurs) in self._get_local_group(t,NEW):
            score *= obs.predictive(data,otherdata) * durs.predictive(dur,otherdurs)

        return score

    def _get_local_group(self,t,k):
        '''
        returns a sequence of length between 1 and 3, where each sequence element is
        ((data,otherdata), (dur,otherdurs))
        '''
        # all the ugly logic is in this method
        # temporarily modifies members, like self.stateseq and maybe self.data
        assert self.stateseq[t] == SAMPLING

        # save original views
        orig_data = self.data
        orig_stateseq = self.stateseq

        # temporarily set stateseq to hypothetical stateseq
        # so that we can get the indicator sequence
        orig_stateseq[t] = k
        wholegroup, pieces = self._get_local_slices(orig_stateseq,t)
        orig_stateseq[t] = SAMPLING

        # build local group of statistics, messing with self.data and self.stateseq views
        localgroup = []
        exclusion = np.zeros(self.T,dtype=bool)
        exclusion[wholegroup] = True
        exclusion[t] = False # include t to break pieces; its label is SAMPLING
        for piece, val in pieces:
            # hide the stuff we don't want to count
            self.data = ma.masked_array(orig_data,exclusion)
            self.stateseq = ma.masked_array(orig_stateseq,exclusion)

            # get all the other data (using our handy exclusion)
            otherdata, otherdurs = self.model._get_data_withlabel(val), self.model._get_durs_withlabel(val)

            # add a piece to our localgroup
            localgroup.append(((orig_data[piece],otherdata),(piece.stop-piece.start,otherdurs)))

            # remove the used piece from the exclusion
            exclusion[piece] = False

        # restore original views
        self.data = orig_data
        self.stateseq = orig_stateseq

        # return
        return localgroup

    @classmethod
    def __get_local_slices(cls,stateseq,t):
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

    ### super-state sampler stuff

    def resample_superstate_version(self):
        raise NotImplementedError


#######################
#  Utility Functions  #
#######################

# maybe factor these out into a cython util file

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

def get_norep(s):
    return rle(s)[0]
