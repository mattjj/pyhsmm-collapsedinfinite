from __future__ import division
import numpy as np
from collections import defaultdict

# TODO could implement slicing in my 'infinite vector': don't use a raw
# defaultdict. also list indexing.
infinite_vector = defaultdict

# TODO is keys() iter order guaranteed to be the same as values() iteration
# order? i assume so :D

class beta(object):
    def __init__(self,gamma):
        self.gamma = gamma
        self.betavec = infinite_vector(stickbreaking(gamma).next)

    def resample(self,nonzerocountsdict):
        ks, counts = nonzerocountsdict.keys(), np.array(nonzerocountsdict.values())
        weights = np.random.dirichlet(np.concatenate(self.gamma + counts, self.gamma))
        newbetavec = infinite_vector(stickbreaking(self.gamma,total=weights[-1]).next)
        newbetavec.update(zip(ks,weights[:-1]))

def stickbreaking(gamma,total=1.):
    while True:
        proportion = np.random.beta(1,gamma)
        piece, total = proportion*total, (1-proportion)*total
        yield piece
