from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt
import abc
from collections import defaultdict

from warnings import warn
import pdb

from pybasicbayes.abstractions import ModelGibbsSampling

from internals import transitions, states

class Collapsed(ModelGibbsSampling):
    __metaclass__ = abc.ABCMeta

    def resample_model(self):
        for s in self.states_list:
            s.resample()
        self.beta.resample([(k,self._counts_to(k)) for k in self._occupied()])

    # statistic gathering methods

    def _counts_from(self,k):
        # returns an integer
        return sum(s._counts_from(k) for s in self.states_list)

    def _counts_to(self,k):
        # returns an integer
        return sum(s._counts_to(k) for s in self.states_list)

    def _counts_fromto(self,k1,k2):
        # returns an integer
        return sum(s._counts_fromto(k1,k2) for s in self.states_list)

    def _data_withlabel(self,k):
        # returns a list of (masked) arrays
        return [s._data_withlabel(k) for s in self.states_list]

    def _occupied(self):
        # returns a set
        return reduce(set.union,(s._occupied() for s in self.states_list),set([]))

    ### optional methods

    def plot(self,*args,**kwargs):
        warn('using temporary implementation of plot') # TODO
        for s in self.states_list:
            plt.matshow(np.tile(s.stateseq,(s.stateseq//10,1)))


class collapsed_stickyhdphmm(Collapsed):
    def __init__(self,gamma_0,alpha_0,kappa,obs):
        self.gamma_0 = gamma_0
        self.alpha_0 = alpha_0
        self.kappa = kappa
        self.obs = obs

        self.beta = transitions.beta(gamma_0=gamma_0)

        self.states_list = []

    def add_data(self,data):
        self.states_list.append(states.collapsed_stickyhdphmm_states(
            model=self,beta=self.beta,alpha_0=self.alpha_0,
            kappa=self.kappa,obs=self.obs,data=data))

    def generate(self,T,keep=True):
        # TODO only works if there's no other data in the model; o/w need to add
        # existing data to obs resample. it should be an easy update.
        assert len(self.states_list) == 0

        tempstates = states.collapsed_stickyhdphmm_states(
                T=T,model=self,beta=self.beta,alpha_0=self.alpha_0,
                kappa=self.kappa,obs=self.obs)

        used_states = np.bincount(tempstates.stateseq)

        allobs = []
        for state, count in enumerate(used_states):
            self.obs.resample()
            allobs.append([self.obs.rvs(1) for itr in range(count)])

        obs = []
        for state in tempstates.stateseq:
            obs.append(allobs[state].pop())
        obs = np.concatenate(obs)

        if keep:
            tempstates.data = obs
            self.states_list.append(tempstates)

        return obs, tempstates.stateseq


class collapsed_hdphsmm(Collapsed):
    def __init__(self,gamma_0,alpha_0,obs,dur):
        self.gamma_0 = gamma_0
        self.alpha_0 = alpha_0
        self.obs = obs
        self.dur = dur

        self.beta = transitions.beta(gamma_0=gamma_0)

        self.states_list = []

    def add_data(self,data):
        self.states_list.append(states.collapsed_hdphsmm_states(
            model=self,beta=self.beta,alpha_0=self.alpha_0,
            obs=self.obs,dur=self.dur,data=data))

    def _durs_withlabel(self,k):
        # returns a list of (masked) arrays
        return [s._durs_withlabel(k) for s in self.states_list]

    def generate(self,T,keep=True):
        # TODO only works if there's no other data in the model
        assert len(self.states_list) == 0

        tempstates = states.collapsed_hdphsmm_states(
                T=T,model=self,beta=self.beta,alpha_0=self.alpha_0,
                obs=self.obs,dur=self.dur)

        used_states = defaultdict(lambda: 0)
        for state in tempstates.stateseq:
            used_states[state] += 1

        allobs = {}
        for state,count in used_states.items():
            self.obs.resample()
            allobs[state] = [self.obs.rvs(1) for itr in range(count)]

        obs = []
        for state in tempstates.stateseq:
            obs.append(allobs[state].pop())
        obs = np.concatenate(obs)

        if keep:
            tempstates.data = obs
            self.states_list.append(tempstates)

        return obs, tempstates.stateseq


# TODO methods to convert to/from weak limit representations
