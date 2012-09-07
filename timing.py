from __future__ import division
from timeit import timeit

# functions in this file can be called like
# timing.get_wl_timing(alpha=6,gamma=6,L=10,
#  obsdistnstring='pyhsmm.basic.distributions.ScalarGaussianNIX(mu_0=0.,kappa_0=0.1,sigmasq_0=1,nu_0=10)',
#  durdistnstring='pyhsmm.basic.distributions.PoissonDuration(2*30,2)')

NUM_RUNS = 5
NITER = 50

wl_setup = \
'''
import numpy as np
import pyhsmm
model = pyhsmm.models.HSMM({alpha},{gamma},
    [{obsdistnstring} for itr in range({L})],
    [{durdistnstring} for itr in range({L})])
model.add_data(np.{data!r})
'''

# NOTE: the stuff below means code must be run from the directory containing the
# collapsedinfinite models.py file
da_setup = \
'''
import numpy as np
import models, pyhsmm
model = models.collapsed_hdphsmm({alpha_0},{gamma_0},
        {obsclassstring},
        {durclassstring})
model.add_data(np.{data!r})
'''

stmt = \
'''
for itr in range(%d):
    model.resample_model()
''' % NITER

def get_wl_timing(**kwargs):
    return timeit(stmt=stmt,setup=wl_setup.format(**kwargs),number=NUM_RUNS)/NUM_RUNS/NITER

def get_da_timing(**kwargs):
    return timeit(stmt=stmt,setup=da_setup.format(**kwargs),number=NUM_RUNS)/NUM_RUNS/NITER
