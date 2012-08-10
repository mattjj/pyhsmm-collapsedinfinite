from __future__ import division
import numpy as np
from collections import defaultdict

infinite_vector = defaultdict

class beta(object):
    def __init__(self,gamma_0):
        self.gamma_0 = gamma_0
        self.remaining = 1.
        self.betavec = infinite_vector(self._stickbreaking(gamma_0).next)

    def resample(self,label_tocount_pairs):
        ks, counts = zip(*label_tocount_pairs)
        weights = np.random.dirichlet(
                np.concatenate((self.gamma_0 + np.array(counts), (self.gamma_0,))))
        self.betavec = infinite_vector(self._stickbreaking(self.gamma_0).next)
        self.betavec.update(zip(ks,weights[:-1]))
        self.remaining = weights[-1]

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
