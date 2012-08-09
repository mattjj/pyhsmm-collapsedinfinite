from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt
plt.interactive(True)
import os, cPickle

import models, pybasicbayes
from pymattutil.text import progprint_xrange

obs_hypparams = (0.,0.1,1.,10)
dur_hypparams = (20*5,20)

plt.figure()

if os.path.isfile('data'):
    with open('data','r') as infile:
        thetuple = cPickle.load(infile)
    if len(thetuple) == 2:
        data, truestates, initialization = thetuple + (None,)
    else:
        data, truestates, initialization = thetuple
else:
    while True:
        blah = models.collapsed_hdphsmm(4,10,obs=pybasicbayes.distributions.ScalarGaussianNIX(*obs_hypparams),dur=pybasicbayes.distributions.PoissonDuration(*dur_hypparams))
        data, truestates = blah.generate(40)

        plt.close('all')
        plt.plot(data,'bx')
        plt.title('data')

        plt.matshow(np.tile(truestates,(10,1)))
        plt.title('true states')

        if 'y' == raw_input('proceed? ').lower():
            break

blah = models.collapsed_hdphsmm(4,10,obs=pybasicbayes.distributions.ScalarGaussianNIX(*obs_hypparams),dur=pybasicbayes.distributions.PoissonDuration(*dur_hypparams))
blah.add_data(data,stateseq=initialization)

for itr in progprint_xrange(20):
    blah.resample_model()

plt.matshow(np.tile(blah.states_list[-1].stateseq,(10,1)))
plt.title('after laborious resampling')

plt.show()
