from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import diskmemo

#################
#  Experiments  #
#################


def hsmm_vs_stickyhmm():
    pass

def wl_is_faster():
    pass

def wl_gives_same_answers():
    pass

####################
#  Sample-getting  #
####################

# the cached versions won't re-run samples that have already been collected

@diskmemo.memoize
def get_hmm_samples():
    # sticky hdp-hmm samples and hdp-hsmm label samples
    pass

@diskmemo.memoize
def get_hsmm_samples():
    # hdp-hsmm label samples and weak limit samples
    pass
