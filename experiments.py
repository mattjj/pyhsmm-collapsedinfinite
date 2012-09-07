from __future__ import division
import numpy as np
na = np.newaxis
import cPickle, os
from matplotlib import pyplot as plt

import pyhsmm, models, util, timing
import diskmemo

SAVING = False

obs_hypparams = dict(mu_0=0.,kappa_0=0.02,sigmasq_0=1,nu_0=10)
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

allfigfuncs = []

def compare_timing():
    wl_timing = timing.get_wl_timing(alpha=6,gamma=6,L=10,data=hsmm_data,obsdistnstring='pyhsmm.basic.distributions.ScalarGaussianNIX(mu_0=0.,kappa_0=0.02,sigmasq_0=1,nu_0=10)',durdistnstring='pyhsmm.basic.distributions.PoissonDuration(2*10,2)')
    da_timing = timing.get_da_timing(alpha_0=6,gamma_0=6,data=hsmm_data,obsclassstring='pyhsmm.basic.distributions.ScalarGaussianNIX(mu_0=0.,kappa_0=0.02,sigmasq_0=1,nu_0=10)',durclassstring='pyhsmm.basic.distributions.PoissonDuration(2*10,2)')

    print 'WL time per iteration: %0.4f' % wl_timing
    print 'DA time per iteration: % 0.4f' % da_timing

    return wl_timing, da_timing

def wl_is_faster_hamming():
    # show hamming error to true state sequence decreases faster with wl

    ### get samples
    wl_samples = get_hdphsmm_wl_poisson_samples(hsmm_data,nruns=100,niter=300,L=10)
    da_samples = get_hdphsmm_da_poisson_samples(hsmm_data,nruns=24,niter=150)

    ### get hamming errors for samples
    def f(tup):
        return util.stateseq_hamming_error(tup[0],tup[1])
    wl_errs = np.array(dv.map_sync(f,zip(wl_samples,[hsmm_labels]*len(wl_samples))))
    da_errs = np.array(dv.map_sync(f,zip(da_samples,[hsmm_labels]*len(da_samples))))

    ### plot
    plt.figure()

    for errs, samplername, color in zip([wl_errs, da_errs],['Weak Limit','Direct Assignment'],['b','g']):
        plt.plot(np.median(errs,axis=0),color+'-',label='%s Sampler' % samplername)
        plt.plot(util.scoreatpercentile(errs.copy(),per=25,axis=0),color+'--')
        plt.plot(util.scoreatpercentile(errs.copy(),per=75,axis=0),color+'--')

    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('Hamming error')

    save('figures/wl_is_faster_hamming.pdf')

    return wl_errs, da_errs
allfigfuncs.append(wl_is_faster_hamming)

def hsmm_vs_stickyhmm():
    # show convergence rates in #iter are same

    ### get samples
    hsmm_samples = get_hdphsmm_da_geo_samples(hmm_data,nruns=50,niter=100)
    shmm_samples = get_shdphmm_da_samples(hmm_data,nruns=50,niter=100)

    ### get hamming errors for samples
    def f(tup):
        return util.stateseq_hamming_error(tup[0],tup[1])
    hsmm_errs = np.array(dv.map_sync(f,zip(hsmm_samples,[hmm_labels]*len(hsmm_samples))))
    shmm_errs = np.array(dv.map_sync(f,zip(shmm_samples,[hmm_labels]*len(shmm_samples))))

    ### plot
    plt.figure()

    for errs, samplername, color in zip([hsmm_errs, shmm_errs],['Geo-HDP-HSMM DA','Sticky-HDP-HMM DA'],['b','g']):
        plt.plot(np.median(errs,axis=0),color+'-',label='%s Sampler' % samplername)
        plt.plot(util.scoreatpercentile(errs.copy(),per=25,axis=0),color+'--')
        plt.plot(util.scoreatpercentile(errs.copy(),per=75,axis=0),color+'--')

    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('Hamming error')

    save('figures/hsmm_vs_stickyhmm.pdf')

    return hsmm_errs, shmm_errs
allfigfuncs.append(hsmm_vs_stickyhmm)

####################
#  Sample-getting  #
####################

@diskmemo.memoize
def get_hdphsmm_wl_poisson_samples(data,nruns,niter,L):
    return get_samples_parallel(hdphsmm_wl_poisson_sampler,nruns,niter=niter,data=hsmm_data,L=L,alpha_0=6,gamma_0=6)

@diskmemo.memoize
def get_hdphsmm_da_poisson_samples(data,nruns,niter):
    return get_samples_parallel(hdphsmm_da_poisson_sampler,nruns,niter=niter,data=hsmm_data,alpha_0=6,gamma_0=6)

@diskmemo.memoize
def get_shdphmm_da_samples(data,nruns,niter):
    return get_samples_parallel(shdphmm_da_sampler,nruns,niter=niter,data=data,alpha_0=6,gamma_0=6,kappa=30)

@diskmemo.memoize
def get_hdphsmm_da_geo_samples(data,nruns,niter):
    return get_samples_parallel(hdphsmm_da_geo_sampler,nruns,niter=niter,data=data,alpha_0=6,gamma_0=6)

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

###################
#  Figure saving  #
###################

import os
def save(pathstr):
    filepath = os.path.abspath(pathstr)
    if SAVING:
        if (not os.path.isfile(pathstr)) or raw_input('save over %s? [y/N] ' % filepath).lower() == 'y':
            plt.savefig(filepath)
            print 'saved %s' % filepath
            return
    print 'not saved'

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

##########
#  Main  #
##########

def main():
    for f in allfigfuncs:
        f()

if __name__ == '__main__':
    main()
