''' Simulation Environment Module '''

#############
## Imports ##
#############

import torch
import numpy as np
import warnings
from copy import deepcopy


#################
## Environment ##
#################

class Environment(object):
    ''' Simulation Environment '''
    def __init__(self, **kwargs):

        ''' Environment __init__

        Arguments
        ---------
        t_start (float) [s]: start time of the simulation
        t_end (float) [s]: end time of the simulation (not included)
        dt (float) [s]: timestep of the simulation
        wl (float) [m]: center wavelength of the simulation
        wls (array) [m]: all wavelengths of the simulation
        num_wl (int) : number of wavelengths in the simulation
        use_delays (bool) : if delays are necessary in the simulation.
                            You can set this to false in frequency domain simulations
                            with constant input source.
        num_batches (int) : number of parallel simulations to perform
        num_timesteps (int) : number of timesteps in the simulation (overrules t_end)
        t (array): time array (overrules t_start, t_end, dt and num_timesteps)
        cuda (True, False, None): whether to use cuda in the simulation.
                                  If `None`, the default of the network is used.
        name (str) : name of the environment
        '''

        # wavelength
        wls = kwargs.pop('wls',None)
        self.wl = kwargs.pop('wl', 1.55e-6)
        if wls is not None:
            self.wls = wls

        # time data:
        dt = kwargs.pop('dt', 1e-14)
        num_timesteps = kwargs.pop('num_timesteps', None)
        t_start = kwargs.pop('t_start', 0)
        t_end = kwargs.pop('t_end', 1e-12)

        self._dt = dt # dt has to be stored seperatly for the extreme case of len(t) < 1
        t = kwargs.pop('t', None)
        if t is not None:
            self.t = np.array(t)
            self._dt = t[1] - t[0]
        elif num_timesteps is None:
            self.t = np.arange(t_start, t_end, dt)
        else:
            self.t = np.arange(t_start, num_timesteps*dt, dt)

        # batches
        self.num_batches = kwargs.pop('num_batches', 1)

        # use delays
        # (set to False to speed up frequency calculations with constant source)
        self.use_delays = kwargs.pop('use_delays', True)

        # use CUDA or not
        self.cuda = kwargs.pop('cuda', None)
        if self.cuda and not torch.cuda.is_available():
            warnings.warn('CUDA requested, but is not available. Rollback to CPU')
            self.cuda = False

        # name
        self.name = kwargs.pop('name', 'env')

        # other keyword arguments are added to the attributes like so:
        for k, v in kwargs.items():
            self.__dict__[k] = v

    @property
    def wl(self):
        return np.mean(self.wls)
    @wl.setter
    def wl(self, value):
        self.wls = np.array([value])

    @property
    def num_wl(self):
        return self.wls.shape[0]

    @property
    def num_timesteps(self):
        ''' Number of timesteps in the simulation '''
        return self.t.shape[0]
    @num_timesteps.setter
    def num_timesteps(self, value):
        ''' Number of timesteps in the simulation '''
        self.t = np.arange(self.t_start, value*self.dt, self.dt)

    @property
    def dt(self):
        ''' Timestep of the simulation '''
        return self._dt
    @dt.setter
    def dt(self, value):
        ''' Timestep of the simulation '''
        self._dt = value
        self.t = np.arange(self.t_start, self.t_end, value)

    @property
    def t_start(self):
        ''' Starting time of the simulation '''
        return self.t[0]
    @t_start.setter
    def t_start(self, value):
        ''' Starting time of the simulation '''
        self.t = np.arange(value, self.t_end, self.dt)

    @property
    def t_end(self):
        ''' End time of the simulation '''
        return self.t[-1] + self.dt
    @t_end.setter
    def t_end(self, value):
        ''' End time of the simulation '''
        self.t = np.arange(self.t_start, value, self.dt)

    def copy(self, **kwargs):
        ''' Create a (deep) copy of the environment object '''
        new = deepcopy(self)
        for kw, val in kwargs.items():
            setattr(new, kw, val)
        if 'dt' in kwargs:
            new.dt = kwargs['dt']
        return new

    def __repr__(self):
        return self.name + '(wl=%.2e, dt=%.2e)'%(self.wl, self.dt)
