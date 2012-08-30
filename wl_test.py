from __future__ import division

import pyhsmm
import models
from pyhsmm.util.text import progprint_xrange

obs_hypparams = (0.,0.1,1.,10)
dur_hypparams = (20,5)
Nmax = 30

# with open('data','r') as infile:
#     data,truestates = cPickle.load(infile)

truth = pyhsmm.models.HSMM(2,2,
        [pyhsmm.basic.distributions.ScalarGaussianNIX(*obs_hypparams) for s in range(Nmax)],
        [pyhsmm.basic.distributions.GeometricDuration(*dur_hypparams) for s in range(Nmax)])

data = truth.generate(5)[0]

### weak limit model

b = pyhsmm.models.HSMM(2,2,
        [pyhsmm.basic.distributions.ScalarGaussianNIX(*obs_hypparams) for s in range(Nmax)],
        [pyhsmm.basic.distributions.GeometricDuration(*dur_hypparams) for s in range(Nmax)])

b.add_data(data)

seqs = []
for itr in progprint_xrange(2000):
    b.resample_model()
    seqs.append(b.states_list[0].stateseq.copy())


### label sequence model
a = models.collapsed_hdphsmm(2,2,
        obs=pyhsmm.basic.distributions.ScalarGaussianNIX(*obs_hypparams),
        dur=pyhsmm.basic.distributions.GeometricDuration(*dur_hypparams))

a.add_data(data)

lseqs = []
for itr in progprint_xrange(2000):
    a.resample_model_labels()
    lseqs.append(a.states_list[0].stateseq.copy())

