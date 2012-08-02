from __future__ import division
import operator
import abc

import transitions, states

# TODO methods to convert to/from weak limit representations

def union(itr):
    return reduce(operator.or_, itr)

class collapsed(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self,*args,**kwargs):
        pass

    @abc.abstractmethod
    def add_data(self,data):
        pass

    ### generic methods

    def resample(self):
        self.states.resample()
        self.beta.resample([(k,self.get_counts_from(k)) for k in self.get_occupied()])

    # statistic gathering methods

    def get_counts_from(self,k):
        return sum(s.get_counts_from(k) for s in self.states_list)

    def get_counts_fromto(self,k1,k2):
        return sum(s.get_counts_fromto(k1,k2) for s in self.states_list)

    def get_data_withlabel(self,k):
        return union(s.get_data_withlabel(k) for s in self.states_list)

    def get_durs_withlabel(self,k):
        return union(s.get_durs_withlabel(k) for s in self.states_list)

    def get_occupied(self):
        return union(self.get_occupied() for s in self.states_list)

    ### optional methods

    def generate(self,T):
        raise NotImplementedError

    def plot(self,*args,**kwargs):
        raise NotImplementedError


class collapsed_hdphsmm(collapsed):
    def __init__(self,gamma,alpha,obs,dur):
        self.gamma = gamma
        self.alpha = alpha
        self.obs = obs
        self.dur = dur

        self.beta = transitions.beta(gamma=gamma)

        self.states_list = []

    def add_data(self,data):
        self.states_list.append(states.collapsed_hdphsmm_states(
                betavec=self.beta.betavec,alpha=self.alpha,obs=self.obs,dur=self.dur,data=data))


class collapsed_stickyhdphmm(collapsed):
    pass

