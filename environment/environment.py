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
            num_timesteps (int) : number of timesteps in the simulation (overrules t_end)
            t (array): time array (overrules t_start, t_end, dt and num_timesteps)
            cuda (True, False, None): whether to use cuda in the simulation.
                                    If `None`, the default of the network is used.
            name (str) : name of the environment
        '''

        # wavelength
        wl = kwargs.pop('wl', None)
        if wl is not None and np.array(wl).ndim > 0:
            kwargs['wls'] = wl
            wl = None
        dwl = kwargs.pop('dwl', None)
        num_wl = int(kwargs.pop('num_wl', 1))
        wl_start = float(kwargs.pop('wl_start', 1.5e-6))
        wl_end = float(kwargs.pop('wl_end', 1.6e-6))
        if wl is None and dwl is None and num_wl==1:
            wls = np.array([0.5*(wl_start+wl_end)])
        elif wl is None and dwl is None and num_wl>1:
            wls = np.linspace(wl_start, wl_end, num_wl, endpoint=True)
        elif wl is None and num_wl==1:
            wls = np.arange(wl_start, wl_end, float(dwl))
        elif dwl is None and num_wl==1:
            wls = np.array([float(wl)])
        wls = np.asarray(kwargs.pop('wls', wls), dtype=np.float64)
        if wls.ndim == 0: wls = wls[None]
        self.wls = wls

        # use delays
        # (set to False to speed up frequency calculations with constant source)
        self.use_delays = bool(kwargs.pop('use_delays', not kwargs.pop('no_delays', False)))

        # time data:
        dt = kwargs.pop('dt', 1e-14)
        num_timesteps = int(kwargs.pop('num_timesteps', 1 if not self.use_delays else 1000))
        t_start = float(kwargs.pop('t_start', 0))
        t_end = float(kwargs.pop('t_end', t_start + num_timesteps*dt))
        if num_timesteps == 1:
            self.t = np.array([dt])
        else:
            self.t = np.asarray(kwargs.pop('t', np.arange(t_start, t_end, dt)))

        # use CUDA or not
        self.cuda = kwargs.pop('cuda', None)
        if self.cuda and not torch.cuda.is_available():
            warnings.warn('CUDA requested, but is not available. Rollback to CPU', RuntimeWarning)
            self.cuda = False

        # name
        self.name = kwargs.pop('name', 'env')

        # other keyword arguments are added to the attributes like so:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def copy(self, **kwargs):
        ''' Create a (deep) copy of the environment object '''
        new = deepcopy(self)
        for kw, val in kwargs.items():
            setattr(new, kw, val)
        return new

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
        self.wls = np.array([float(value)])

    @property
    def dwl(self):
        ''' Wavelength step of the simulation '''
        if self.num_wl > 1:
            return self.wls[1] - self.wls[0]
    @dwl.setter
    def dwl(self, value):
        ''' Wavelength step of the simulation '''
        if self.num_wl > 1:
            self.wls = np.arange(self.wls[0], self.wls[-1] - 0.5*value, value)

    @property
    def wl_start(self):
        ''' Wavelength start of the simulation '''
        return self.wls[0]
    @wl_start.setter
    def wl_start(self, value):
        ''' Wavelength start of the simulation '''
        self.wls = self.wls[self.wls > value]
        self.wls -= (self.wls[0] - value)

    @property
    def wl_end(self):
        ''' Wavelength end of the simulation '''
        return self.wls[-1]
    @wl_end.setter
    def wl_end(self, value):
        ''' Wavelength end of the simulation '''
        self.wls = self.wls[self.wls<value]
        self.wls += (value - self.wls[-1])

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
    def t(self):
        ''' Time range of the simulation '''
        return self._t
    @t.setter
    def t(self, value):
        ''' Time range of the simulatino '''
        self._t = np.asarray(value)
        self._dt = self._t[0] if self._t.shape[0] == 1 else self._t[1]-self._t[0]

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
        self.t = self.t[self.t>value]
        self.t -= (self.t[0] - value)

    @property
    def t_end(self):
        ''' End time of the simulation '''
        return self.t[-1] + self.dt if self.t.shape[0] > 1 else self.t[-1]
    @t_end.setter
    def t_end(self, value):
        ''' End time of the simulation '''
        self.t = self.t[self.t<value]

    @property
    def num_timesteps(self):
        ''' Number of timesteps in the simulation '''
        return self.t.shape[0]
    @num_timesteps.setter
    def num_timesteps(self, value):
        ''' Number of timesteps in the simulation '''
        self.t = np.arange(self.t_start, self.t_start + value*self.dt-0.5*self.dt, self.dt)

    @property
    def no_delays(self):
        ''' To use delays during simulation or not '''
        return not self.use_delays
    @no_delays.setter
    def no_delays(self, value):
        ''' To use delays during simulation or not '''
        self.use_delays = bool(not value)

    def __to_str_dict(self):
        ''' Dictionary representation of the environment '''
        env = OrderedDict()
        env['name'] = repr(self.name)
        if self.use_delays:
            env['t_start'] = '%.2e'%self.t_start if self.t_start > self.dt else '0'
            env['t_end'] = '%.2e'%self.t_end
            env['dt'] = '%.2e'%self.dt
        if self.num_wl > 1:
            env['wl_start'] = '%.3e'%self.wls[0]
            env['wl_end'] = '%.3e'%self.wls[-1]
            env['num_wl'] = repr(self.num_wl)
        else:
            env['wl'] = '%.3e'%self.wl
        env.update({k:repr(v) for k, v in self.__dict__.items() if k[0] != '_'})
        del env['wls']
        return env

    def __repr__(self):
        ''' String Representation of the environment '''
        dic = self.__to_str_dict()
        return 'Environment('+', '.join('='.join([k, v]) for k, v in dic.items())+')'


    def __str__(self):
        ''' String Representation of the environment '''
        dic = self.__to_str_dict()
        s =     'Simulation Environment:\n'
        s = s + '-----------------------\n'
        for k, v in dic.items():
            s = s + '%s = %s\n'%(str(k), str(v))
        return s
