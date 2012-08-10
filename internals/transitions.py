from __future__ import division
import numpy as np
from collections import defaultdict


class beta(object):
    def __init__(self,gamma_0,model):
        self.gamma_0 = gamma_0
        self.model = model
        self.remaining = 1.
        self.betavec = defaultdict(self._stickbreaking(gamma_0).next)

    def resample(self):
        ks = list(self.model._occupied())
        n = self._get_countmatrix(ks)

        # m auxiliary variables (from original HDP paper, sec 5.3)
        msum = np.zeros(len(ks))
        for (i,j), nij in np.ndenumerate(n):
            for x in range(nij):
                msum[j] += np.random.random() < self.gamma_0 * self.betavec[ks[j]] \
                            / (x + self.gamma_0 * self.betavec[ks[j]])

        weights = np.random.dirichlet(
                np.concatenate((self.gamma_0 + msum, (self.gamma_0,))))
        self.betavec = defaultdict(self._stickbreaking(self.gamma_0).next)
        self.betavec.update(zip(ks,weights[:-1]))
        self.remaining = weights[-1]

    def _get_countmatrix(self,ks):
        counts = np.zeros((len(ks),len(ks)))
        for i in ks:
            for j in ks:
                counts[i,j] = self.model._counts_fromto(i,j)
        return counts

    def housekeeping(self,ks):
        ks = set(ks)
        toremove = set(self.betavec.keys()) - ks
        if toremove != set():
            deletionmass = sum(self.betavec[k] for k in toremove)
            for k in toremove:
                del self.betavec[k]
            self.remaining += deletionmass

        assert all(v > 0 for v in self.betavec.values()) and \
                np.allclose(self.remaining + sum(self.betavec.values()),1)

    def _stickbreaking(self,gamma):
        while True:
            p = np.random.beta(1,gamma)
            piece, self.remaining = p*self.remaining, (1-p)*self.remaining
            yield piece

class censored_beta(beta):
    def _get_countmatrix(self,ks):
        counts = super(censored_beta,self)._get_countmatrix(ks)
        assert np.all(np.diag(counts) == 0)
        counts_from = counts.sum(1)
        for i in range(counts.shape[0]):
            pi_ii = np.random.beta(1,self.gamma)
            counts[i,i] = np.random.geometric(1-pi_ii,size=counts_from[i]).sum()
        return counts
