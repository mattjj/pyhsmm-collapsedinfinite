from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt
plt.interactive(True)
import os, cPickle

import models, pybasicbayes
from pymattutil.text import progprint_xrange

obs_hypparams = (0.,0.1,1.,10)
dur_hypparams = (1,20)

if os.path.isfile('data'):
    with open('data','r') as infile:
        thetuple = cPickle.load(infile)
    if len(thetuple) == 2:
        data, truestates, initialization = thetuple + (None,)
    else:
        data, truestates, initialization = thetuple
else:
    while True:
        blah = models.collapsed_hdphsmm(2,2,obs=pybasicbayes.distributions.ScalarGaussianNIX(*obs_hypparams),dur=pybasicbayes.distributions.Poisson(5*5,5))
        data, truestates = blah.generate(40)

        plt.close('all')
        plt.plot(data,'bx')
        plt.title('data')

        plt.matshow(np.tile(truestates,(10,1)))
        plt.title('true states')

        if 'y' == raw_input('proceed? ').lower():
            with open('data','w') as outfile:
                cPickle.dump((data,truestates),outfile,protocol=2)
            break

### sticky hdphmm sampler

# stickystateseqs = []
# for superitr in range(5):
#     blah = models.collapsed_stickyhdphmm(2,2,50,obs=pybasicbayes.distributions.ScalarGaussianNIX(*obs_hypparams))
#     blah.add_data(data)

#     hi = []
#     for itr in progprint_xrange(5000):
#         blah.resample_model()
#         hi.append(blah.states_list[0].stateseq.copy())
#     stickystateseqs.append(hi)

# with open('stickystateseqs.samples','w') as outfile:
#     cPickle.dump(stickystateseqs,outfile,protocol=2)

### hdphsmm superstate sampler


superstatestateseqs = []
for superitr in range(5):
    blah = models.collapsed_hdphsmm(2,2,obs=pybasicbayes.distributions.ScalarGaussianNIX(*obs_hypparams),dur=pybasicbayes.distributions.GeometricDuration(*dur_hypparams))
    blah.add_data(data)

    hi = []
    for itr in progprint_xrange(5000):
        blah.resample_model_superstates()
        hi.append(blah.states_list[0].stateseq.copy())
    superstatestateseqs.append(hi)

with open('superstatestateseqs.samples','w') as outfile:
    cPickle.dump(superstatestateseqs,outfile,protocol=2)

### hdphsmm label sampler


labelstateseqs = []
for superitr in range(5):
    blah = models.collapsed_hdphsmm(2,2,obs=pybasicbayes.distributions.ScalarGaussianNIX(*obs_hypparams),dur=pybasicbayes.distributions.GeometricDuration(*dur_hypparams))
    blah.add_data(data)

    hi = []
    for itr in progprint_xrange(5000):
        blah.resample_model_labels()
        hi.append(blah.states_list[0].stateseq.copy())
    labelstateseqs.append(hi)

with open('labelstateseqs.samples','w') as outfile:
    cPickle.dump(labelstateseqs,outfile,protocol=2)

