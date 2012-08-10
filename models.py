from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt
from matplotlib import cm
import abc
from collections import defaultdict

from warnings import warn

from pybasicbayes.abstractions import ModelGibbsSampling
from pymattutil.general import rle

from internals import transitions, states

class Collapsed(ModelGibbsSampling):
    __metaclass__ = abc.ABCMeta

    def resample_model(self):
        for s in self.states_list:
            s.resample()
        self.beta.resample()

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

    def plot(self,color=None):
        import itertools
        num_states = len(self._occupied())
        state_colors = {}
        idx = 0
        for state in itertools.chain(*[s.stateseq for s in self.states_list]):
            if state not in state_colors:
                state_colors[state] = idx/(num_states-1) if color is None else color
                idx += 1
        cmap = cm.get_cmap()

        for s in self.states_list:
            plt.figure()
            ### obs stuff
            # plt.subplot(2,1,1)
            # for state in rle(s.stateseq)[0]:
            #     self.obs.plot(color=cmap(state_colors[state]),
            #             data=s.data[s.stateseq==state] if s.data is not None else None,
            #             plot_params=False)
            # plt.subplot(2,1,2)

            ### states stuff
            s.plot(colors_dict=state_colors)

class collapsed_stickyhdphmm(Collapsed):
    def __init__(self,gamma_0,alpha_0,kappa,obs):
        self.gamma_0 = gamma_0
        self.alpha_0 = alpha_0
        self.kappa = kappa
        self.obs = obs

        self.beta = transitions.beta(model=self,gamma_0=gamma_0)

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

        self.beta = transitions.censored_beta(model=self,gamma_0=gamma_0)

        self.states_list = []

    def add_data(self,data,stateseq=None):
        self.states_list.append(states.collapsed_hdphsmm_states(
            model=self,beta=self.beta,alpha_0=self.alpha_0,
            obs=self.obs,dur=self.dur,data=data,stateseq=stateseq))

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

    def resample_model_superstates(self):
        for s in self.states_list:
            s.resample_superstate_version()
        self.beta.resample()

    def resample_model_labels(self):
        for s in self.states_list:
            s.resample_label_version()
        self.beta.resample()



# TODO methods to convert to/from weak limit representations
