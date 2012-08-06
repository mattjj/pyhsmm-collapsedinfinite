from __future__ import division
import set
import abc

import transitions, states

# TODO methods to convert to/from weak limit representations

# TODO rename to DACollapsed, maybe?

class collapsed(object):
    __metaclass__ = abc.ABCMeta

    def resample(self):
        for s in self.states_list:
            s.resample()
        self.beta.resample([(k,self.get_counts_from(k)) for k in self.get_occupied()])

    # statistic gathering methods

    def get_counts_from(self,k):
        # returns an integer
        return sum(s.get_counts_from(k) for s in self.states_list)

    def get_counts_fromto(self,k1,k2):
        # returns an integer
        return sum(s.get_counts_fromto(k1,k2) for s in self.states_list)

    def get_data_withlabel(self,k):
        # returns a list of (masked) arrays
        return [s.get_data_withlabel(k) for s in self.states_list]

    def get_durs_withlabel(self,k):
        # returns a list of (masked) arrays
        return [s.get_durs_withlabel(k) for s in self.states_list]

    def get_occupied(self):
        # returns a set
        return reduce(set.union,(s.get_occupied() for s in self.states_list))

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

    def add_data(self,data,**kwargs):
        self.states_list.append(states.collapsed_hdphsmm_states(
                betavec=self.beta.betavec,alpha=self.alpha,obs=self.obs,dur=self.dur,data=data,**kwargs))


class collapsed_stickyhdphmm(collapsed):
    pass

