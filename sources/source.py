''' Input Sources '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np


#####################
## Constant Source ##
#####################

class ConstantSource(object):
    ''' A source with constant amplitude '''
    def __init__(self, nw, amplitude=1, add_phase=False, add_phase_difference=False):
        ''' ConstantSource __init__

        Create a source with constant amplitude

        Parameters
        ----------
        nw : the network for which to create the source
        amplitude=1: the amplitude of the network. One can also specify an array of
                     amplitudes, corresponding to the different batches of the simulation
        add_phase: add time varying phase to source input
        add_phase_difference: add a phase difference between different sources.
        '''

        # shape of the virtual array represented by this source:
        self.shape = (2, nw.env.num_timesteps, nw.nmc, nw.env.num_batches)

        # shape of the virtual array returned by this source:
        self.source = nw.new_variable(nw.zeros((
                nw.nmc, # number of memory containing nodes in the network
                nw.env.num_batches, # number of parallel simulations
        ), cuda=nw.is_cuda))

        self.source[:nw.num_sources] = 1

        if isinstance(amplitude, np.ndarray):
            amplitude = nw.new_variable(torch.from_numpy(amplitude))
            if len(amplitude.size()) == 1:
                amplitude.unsqueeze_(0)

        self.source *= amplitude

        if add_phase_difference:
            pass # not yet implemented

    def size(self, idx=None):
        if idx is None:
            return self.shape
        return self.shape[idx]

    def __getitem__(self, key):
        return key[0]*self.source

