from __future__ import division
import abc

from pybasicbayes.abstractions import ModelGibbsSampling

import transitions, states

# TODO methods to convert to/from weak limit representations

class Collapsed(ModelGibbsSampling):
    __metaclass__ = abc.ABCMeta

    def resample(self):
        for s in self.states_list:
            s.resample()
        self.beta.resample([(k,self._get_counts_from(k)) for k in self._get_occupied()])

    # statistic gathering methods

    def _get_counts_from(self,k):
        # returns an integer
        return sum(s._get_counts_from(k) for s in self.states_list)

    def _get_counts_fromto(self,k1,k2):
        # returns an integer
        return sum(s._get_counts_fromto(k1,k2) for s in self.states_list)

    def _get_data_withlabel(self,k):
        # returns a list of (masked) arrays
        return [s._get_data_withlabel(k) for s in self.states_list]

    def _get_occupied(self):
        # returns a set
        return reduce(set.union,(s._get_occupied() for s in self.states_list))

    ### optional methods

    def generate(self,T):
        raise NotImplementedError

    def plot(self,*args,**kwargs):
        raise NotImplementedError


class collapsed_hdphsmm(Collapsed):
    def __init__(self,gamma,alpha,obs,dur):
        self.gamma = gamma
        self.alpha = alpha
        self.obs = obs
        self.dur = dur

        self.beta = transitions.beta(gamma=gamma)

        self.states_list = []

    def add_data(self,data,**kwargs):
        self.states_list.append(states.collapsed_hdphsmm_states(
                betavec=self.beta.betavec,alpha=self.alpha,obs=self.obs,dur=self.dur,data=data,**kwargs))

    def _get_durs_withlabel(self,k):
        # returns a list of (masked) arrays
        return [s._get_durs_withlabel(k) for s in self.states_list]


class collapsed_stickyhdphmm(Collapsed):
    def __init__(self,gamma,alpha,kappa,obs):
        self.gamma = gamma
        self.alpha = alpha
        self.kappa = kappa
        self.obs = obs

        self.beta = transitions.beta(gamma=gamma)

        self.states_list = []

    def add_data(self,data,**kwargs):
        self.states_list.append(states.collapsed_stickyhdphmm_states(
            betavec=self.beta.betavec,alpha=self.alpha,kappa=self.kappa,obs=self.obs,data=data,**kwargs))

