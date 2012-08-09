from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import models, pybasicbayes
from pymattutil.text import progprint_xrange

obs_hypparams = (0.,0.01,1.,10)

blah = models.collapsed_stickyhdphmm(3,10,20,pybasicbayes.distributions.ScalarGaussianNIX(*obs_hypparams))
data, truestates = blah.generate(100)

plt.figure()
plt.plot(data)
plt.title('data')

plt.matshow(np.tile(truestates,(10,1)))
plt.title('true states')

blah = models.collapsed_stickyhdphmm(3,10,20,pybasicbayes.distributions.ScalarGaussianNIX(*obs_hypparams))
blah.add_data(data)

for itr in progprint_xrange(500):
    blah.resample_model()

plt.matshow(np.tile(blah.states_list[-1].stateseq,(10,1)))
plt.title('after laborious resampling')

plt.show()
