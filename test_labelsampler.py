from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import models, pybasicbayes
from pymattutil.text import progprint_xrange

obs_hypparams = (0.,0.01,2.,3.)

dur_hypparams = (5*10,5)

blah = models.collapsed_hdphsmm(2,10,obs=pybasicbayes.distributions.ScalarGaussianNIX(*obs_hypparams),dur=pybasicbayes.distributions.PoissonDuration(*dur_hypparams))
data, truestates = blah.generate(50)
