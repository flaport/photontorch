'''
# PhotonTorch Simulation Environment

The simulation environment module contains a single class: Environment

This class contains all the necessary parameters to initialize a network for a simulation.

'''

#############
## Imports ##
#############

# Standard Library
import warnings
from copy import deepcopy
from collections import OrderedDict

# Torch
import torch

# Other
import numpy as np


#################
## Environment ##
#################

class Environment(object):
    ''' Simulation Environment

    The simulation environment is a smart data class that contains all
    the necessary parameters to initialize a network for a simulation.

    It is able to initialize parameters that depend on each other correctly.
    Changing a parameter that depends on another, changes both.
    '''
    def __init__(self, **kwargs):

        ''' Environment

        Args:
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
    def c(self):
        ''' speed of light '''
        return 299792458.0 #[m/s]

    @property
    def wl(self):
        ''' Wavelength of the simulation '''
        return np.mean(self.wls)
    @wl.setter
    def wl(self, value):
        ''' Wavelength of the simulation '''
        self.wls = np.array([value])

    @property
    def num_wl(self):
        ''' Number of wavelengths in the simulation '''
        return self.wls.shape[0]
    @num_wl.setter
    def num_wl(self, value):
        ''' Number of wavelengths in the simulation '''
        if value == 1:
            self.wls = np.mean(self.wls, keepdims=True)
        else:
            self.wls = np.linspace(self.wls[0], self.wls[-1], value, endpoint=True)

    @property
    def num_timesteps(self):
        ''' Number of timesteps in the simulation '''
        return self.t.shape[0]
    @num_timesteps.setter
    def num_timesteps(self, value):
        ''' Number of timesteps in the simulation '''
        self.t = np.arange(self.t_start, value*self.dt-0.5*self.dt, self.dt)

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
        ''' String Representation of the environment '''
        env = OrderedDict()
        _env = deepcopy(self.__dict__)
        env['name'] = _env.pop('name',None)
        env['dt'] = '%.2e'%_env.pop('_dt', None)
        env['t_start'] = '%.2e'%self.t_start if self.t_start != 0 else 0
        env['t_end'] = '%.2e'%self.t_end if self.t_end != 0 else 0
        del _env['t']
        wls = _env.pop('wls', None)
        if wls is not None and len(self.wls) == 1:
            env['wl'] = '%.2e'%wls[0]
        else:
            env['wls'] = _env.pop('wls')
        env.update(_env)

        s =     'Simulation Environment:\n'
        s = s + '-----------------------\n'
        for k, v in env.items():
            s = s + '%s : %s\n'%(str(k), str(v))
        return s

    def __str__(self):
        ''' String Representation of the environment '''
        return repr(self)

