from __future__ import division
import numpy as np
na = np.newaxis

from pyhsmm.util.states import sample_discrete
from pyhsmm.util.general import rle

NEW = None
SAMPLING = -1

class collapsed_hdphsmm_states(object):
    def __init__(self,model,betavec,alpha,obsclass,durclass,data=None,T=None,stateseq=None):
        self.alpha = alpha

        self.model = model
        self.betavec = betavec # infinite vector
        self.obsclass = obsclass
        self.durclass = durclass

        self.data = data

        if data is None:
            assert T is not None
            assert stateseq is None, 'not implemented yet!'
            self.T = T
            self.generate()
        else:
            self.data = data
            self.T = data.shape[0]
            self.generate_states()

    def resample(self):
        self.resample_segment_version()

    def get_counts_from(self,k):
        raise NotImplementedError

    def get_counts_fromto(self,k1,k2):
        raise NotImplementedError

    def get_data_withlabel(self,k):
        raise NotImplementedError

    def get_durs_withlabel(self,k):
        raise NotImplementedError

    def get_occupied(self):
        raise NotImplementedError

    def _new_label(self,ks):
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
            ks = self.model.get_occupied()
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
        for (data,otherdata), (dur,otherdurs) in self._get_local_group(t,k):
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
        raise NotImplementedError
        # lots of calls with same t and rest of stateseq in a row... could cache

    ### super-state sampler stuff

    # TODO

    def resample_superstate_version(self):
        raise NotImplementedError

class collapsed_stickyhdphmm_states(object):
    pass

