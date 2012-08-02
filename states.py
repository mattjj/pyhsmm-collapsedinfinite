from __future__ import division
import numpy as np
na = np.newaxis
import numpy.ma as ma
import operator

from pyhsmm.util.stats import sample_discrete
from pyhsmm.util.general import rle

NEW = -1
SAMPLING = -2

# TODO add caching so that, e.g., lots of calls with the same t in a row are faster

# TODO test all these gorram count methods

####################
#  States Classes  #
####################

class collapsed_hdphsmm_states(object):
    def __init__(self,model,betavec,alpha,obsclass,durclass,data=None,T=None,stateseq=None):
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
            self.stateseq = stateseq
            self.generate_obs()
        elif stateseq is None:
            self.data = data
            self.generate_states()
        else:
            self.generate()

    def resample(self):
        self.resample_segment_version()

    def get_counts_from(self,k):
        return self._get_counts_from(self.stateseq,k)

    def get_counts_fromto(self,k1,k2):
        return self._get_counts_fromto(self.stateseq,k1,k2)

    def get_data_withlabel(self,k):
        return self._get_data_withlabel(self.stateseq,self.data,k)

    def get_durs_withlabel(self,k):
        return self._get_durs_withlabel(self.stateseq,k)

    @classmethod
    def _get_counts_from(cls,stateseq,k):
        stateseq_norep = get_norep(stateseq)[:-1] # EXCEPT last
        rawcount = np.sum(stateseq_norep == k)
        if SAMPLING in stateseq_norep:
            sampling_index, = np.where(stateseq_norep == SAMPLING)
            if sampling_index > 0 and stateseq[sampling_index-1] == k:
                return rawcount - 1
        return rawcount

    @classmethod
    def _get_counts_fromto(cls,stateseq,k1,k2):
        stateseq_norep = get_norep(stateseq)
        if k1 not in stateseq:
            return 0
        else:
            from_indices, = np.where(stateseq_norep[:-1] == k1) # EXCEPT last
            return np.sum(stateseq_norep[from_indices + 1] == k2)

    @classmethod
    def _get_data_withlabel(cls,stateseq,data,k):
        return data[stateseq == k]

    @classmethod
    def _get_durs_withlabel(cls,stateseq,k):
        stateseq_norep, durs = rle(stateseq)
        return durs[stateseq_norep == k]

    @classmethod
    def _get_occupied(cls,stateseq):
        return set(stateseq)

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
        '''
        returns a sequence of length between 1 and 3, where each sequence element is
        ((data,otherdata), (dur,otherdurs))
        '''
        # all the ugly logic is in this method
        # temporarily modifies members, like self.stateseq and maybe self.data

        # save original views
        orig_data = self.data
        orig_stateseq = self.stateseq

        # temporarily set stateseq to hypothetical stateseq
        # so that we can get the indicator sequence
        orig_stateseq[t] = k
        wholegroup, pieces = self._get_local_slices(orig_stateseq,t)
        orig_stateseq[t] = SAMPLING

        # build local group, messing with self.data and self.stateseq
        exclusion = wholegroup; exclusion[t] = False # keep the SAMPLING index in there to break the stateseq
        localgroup = []
        for piece in pieces:
            # hide the stuff we don't want to count
            self.data = ma.masked_array(orig_data,exclusion)
            self.stateseq = ma.masked_array(orig_stateseq,exclusion)

            # get all the other data (using our handy exclusion)
            otherdata, otherdurs = self.model.get_data_withlabel(k), self.model.get_durs_withlabel(k)

            # add a piece to our localgroup
            localgroup.append(((orig_data[piece],otherdata),(np.sum(piece),otherdurs)))

            # remove the used piece from the exclusion
            exclusion &= ~piece

        # restore original views
        self.data = orig_data
        self.statseq = orig_stateseq

        # return
        return localgroup

    @classmethod
    def _get_local_slices(stateseq,t):
        '''
        returns slices: wholegroup, (piece1, ...)
        '''
        A,B = fill(stateseq,t-1), fill(stateseq,t+1)
        if A == B:
            return A, (A,)
        elif A.start <= t < A.stop or B.start <= t < B.stop:
            return A|B, [x for x in (A,B) if x.stop - x.start > 0]
        else:
            It = slice(t,t+1)
            return A|B|It, [x for x in (A,It,B) if x.stop - x.start > 0]

    ### super-state sampler stuff

    # TODO

    def resample_superstate_version(self):
        raise NotImplementedError


class collapsed_stickyhdphmm_states(object):
    pass


#######################
#  Utility Functions  #
#######################

def fill(seq,t):
    # TODO implement in C to make this fast
    if t < 0 or t > seq.shape[0]:
        return slice(0,0) # empty
    else:
        endindices, = np.where(np.diff(seq) != 0) # internal end indices (not incl -1 and T-1)
        startindices = np.concatenate(((0,),endindices+1,(seq.shape[0],))) # incl 0 and T
        idx = np.where(startindices <= t)[0][-1]
        return slice(startindices[idx],startindices[idx+1])

def get_norep(s):
    # TODO can make faster
    return rle(s)[0]

