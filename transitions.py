from __future__ import division
import numpy as np
from collections import defaultdict

# TODO could implement slicing in my 'infinite vector': don't use a raw
# defaultdict

class beta(object):
    def __init__(self,gamma):
        self.gamma = gamma
        self.betavec = defaultdict(stickbreaking(gamma).next)

    def resample(self,nonzerocountsdict):
        raise NotImplementedError

def stickbreaking(gamma):
    remaining = 1.
    while True:
        proportion = np.random.beta(1,gamma)
        piece, remaining = proportion*remaining, (1-proportion)*remaining
        yield piece
