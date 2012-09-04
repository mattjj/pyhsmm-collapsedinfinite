from __future__ import division
import numpy as np
import cPickle, os

import pyhsmm, models, util
import diskmemo

obs_hypparams = dict(mu_0=0.,kappa_0=0.05,sigmasq_0=1,nu_0=10)
dur_psn_hypparams = dict(alpha_0=2*10,beta_0=2)
dur_geo_hypparams = dict(alpha_0=4,beta_0=20)

#####################
#  Data generation  #
#####################

def generate_hsmm_data():
    return pyhsmm.models.HSMM(6,6,
            [pyhsmm.basic.distributions.ScalarGaussianNIX(**obs_hypparams) for s in range(10)],
            [pyhsmm.basic.distributions.PoissonDuration(**dur_psn_hypparams) for s in range(10)]).generate(50)

def generate_hmm_data():
    return pyhsmm.models.HSMM(6,6,
            [pyhsmm.basic.distributions.ScalarGaussianNIX(**obs_hypparams) for s in range(10)],
            [pyhsmm.basic.distributions.GeometricDuration(**dur_geo_hypparams) for s in range(10)]).generate(50)

if os.path.isfile('data'):
    with open('data','r') as infile:
        (hmm_data, hmm_labels), (hsmm_data, hsmm_labels) = cPickle.load(infile)
else:
    (hmm_data, hmm_labels), (hsmm_data, hsmm_labels) = thetuple = \
            generate_hmm_data(), generate_hsmm_data()
    with open('data','w') as outfile:
        cPickle.dump(thetuple,outfile,protocol=2)

#################
#  Experiments  #
#################

def wl_is_faster_autocorr():
    # TODO
    # show wl (average?) state autocorrelation is small,
    # while da state autocorrelation is big
    pass

def wl_is_faster_hamming():
    # TODO
    # show hamming error to true state sequence decreases faster with wl
    pass

def hsmm_vs_stickyhmm():
    # TODO
    # show convergence rates in #iter are same
    pass

def wl_gives_same_answers():
    # TODO
    # compare squared errors in posterior mean sequences
    # with wl samples and da samples
    pass

####################
#  Sample-getting  #
####################

@diskmemo.memoize
def get_hdphsmm_wl_poisson_samples(data,nruns=25,niter=200,L=10):
    return get_samples_parallel(hdphsmm_wl_poisson_sampler,nruns,niter=niter,data=hsmm_data,L=L,alpha_0=6,gamma_0=6)

@diskmemo.memoize
def get_hdphsmm_da_poisson_samples(data,nruns=25,niter=200):
    return get_samples_parallel(hdphsmm_da_poisson_sampler,nruns,niter=niter,data=hsmm_data,alpha_0=6,gamma_0=6)

@diskmemo.memoize
def get_shdphmm_da_samples(data,nruns=25,niter=200):
    return get_samples_parallel(hdphsmm_da_geo_sampler,nruns,niter=niter,data=hmm_data,alpha_0=6,gamma_0=6)

@diskmemo.memoize
def get_hdphsmm_da_geo_samples(data,nruns=25,niter=200):
    return get_samples_parallel(shdphmm_da_sampler,nruns,niter=niter,data=hmm_data,alpha_0=6,gamma_0=6)

####################
#  Sample-running  #
####################

def run_model(model,data,niter):
    model.add_data(data)

    seqs = np.empty((niter,data.shape[0]))
    seqs[0] = model.states_list[0].stateseq

    for itr in range(1,niter):
        model.resample_model()
        seqs[itr] = model.states_list[0].stateseq

    return seqs

def hdphsmm_wl_poisson_sampler(niter,data,L,alpha_0,gamma_0):
    model = pyhsmm.models.HSMM(alpha_0,gamma_0,
            [pyhsmm.basic.distributions.ScalarGaussianNIX(**obs_hypparams) for s in range(L)],
            [pyhsmm.basic.distributions.PoissonDuration(**dur_psn_hypparams) for s in range(L)])
    return run_model(model,data,niter)

def hdphsmm_da_poisson_sampler(niter,data,alpha_0,gamma_0):
    model = models.collapsed_hdphsmm(alpha_0,gamma_0,
            obs=pyhsmm.basic.distributions.ScalarGaussianNIX(**obs_hypparams),
            dur=pyhsmm.basic.distributions.PoissonDuration(**dur_psn_hypparams))
    return run_model(model,data,niter)

def hdphsmm_da_geo_sampler(niter,data,alpha_0,gamma_0):
    model = models.collapsed_hdphsmm(alpha_0,gamma_0,
            obs=pyhsmm.basic.distributions.ScalarGaussianNIX(**obs_hypparams),
            dur=pyhsmm.basic.distributions.GeometricDuration(**dur_geo_hypparams))
    return run_model(model,data,niter)

def shdphmm_da_sampler(niter,data,alpha_0,gamma_0,kappa):
    model = models.collapsed_stickyhdphmm(alpha_0,gamma_0,kappa,
            obs=pyhsmm.basic.distributions.ScalarGaussianNIX(**obs_hypparams))
    return run_model(model,data,niter)

def get_samples_parallel(sampler,nruns,**kwargs):
    def applier(tup):
        return apply(tup[0],(),tup[1])
    samples_list = dv.map_sync(applier, zip([sampler]*nruns,[kwargs]*nruns))

    dv.purge_results('all')
    return samples_list

#######################
#  Parallelism stuff  #
#######################

class dummy_directview(object):
    map_sync = map
    __len__ = lambda self: 1
    purge_results = lambda x,y: None
dv = dummy_directview()

def go_parallel():
    global dv, c
    from IPython.parallel import Client
    c = Client()
    dv = c[:]

