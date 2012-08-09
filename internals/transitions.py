from __future__ import division
import numpy as np
from collections import defaultdict

# TODO could implement slicing in my 'infinite vector': don't use a raw
# defaultdict. also list indexing.
infinite_vector = defaultdict

# TODO is keys() iter order guaranteed to be the same as values() iteration
# order? i assume so :D

class beta(object):
    def __init__(self,gamma_0):
        self.gamma_0 = gamma_0
        self.remaining = 1
        self.betavec = infinite_vector(self.stickbreaking(gamma_0).next)

    def resample(self,label_tocount_pairs):
        ks, counts = zip(*label_tocount_pairs)
        counts = np.array(counts)

        weights = np.random.dirichlet(np.concatenate((self.gamma_0 + counts, (self.gamma_0,))))
        newbetavec = infinite_vector(self.stickbreaking(self.gamma_0).next)
        newbetavec.update(zip(ks,weights[:-1]))
        self.remaining = weights[-1]

    def rvs(self,size=[]):
        # TODO natural number samples from beta
        raise NotImplementedError

    def housekeeping(self,ks):
        ks = set(ks)

        rescale_factor = 1.
        deletions = []
        for i in self.betavec:
            if i not in ks:
                rescale_factor *= (1.-self.betavec[i])
                deletions.append(i)
        for d in deletions:
            del self.betavec[d]

        tot = 0.
        for i in self.betavec:
            self.betavec[i] /= rescale_factor
            tot += self.betavec[i]
        self.remaining = 1.-tot
        assert self.remaining >= 0 or self.remaining > -1e-4
        if self.remaining < 0:
            self.remaining = 1e-5
            for k in self.betavec.iterkeys():
                self.betavec[k] *= (1.-1e-5)

    def stickbreaking(self,gamma):
        while True:
            proportion = np.random.beta(1,gamma)
            piece, self.remaining = proportion*self.remaining, (1-proportion)*self.remaining
            yield piece
