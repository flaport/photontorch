''' Simulation Environment Module '''

#############
## Imports ##
#############

import numpy as np


#################
## Environment ##
#################

class Environment(object):
    ''' Simulation Environment '''
    def __init__(
        self,
        t_start=0,
        t_end=1e-12,
        dt=1e-14,
        wl=1.55e-6,
        use_delays=True,
        num_timesteps=None,
        t = None,
        name='',
    ):
        ''' Environment __init__
        
        Arguments
        ---------
        t_start (float) [s]: start time of the simulation
        t_end (float) [s]: end time of the simulation (not included)
        dt (float) [s]: timestep of the simulation
        wl (float) [m]: wavelength of the simulation
        delays (bool) : if delays are necessary in the simulation
        num_timesteps (int) : number of timesteps in the simulation (overrules t_end)
        t (array): time array (overrules t_start, t_end, dt and num_timesteps)
        name (str) : name of the environment
        '''

        # time array:
        self._dt = dt # dt has to be stored seperatly for the extreme case of len(t) < 1
        if t is not None:
            self.t = np.array(t)
            self._dt = t[1] - t[0]
        elif num_timesteps is None:
            self.t = np.arange(t_start, t_end, dt)
        else:
            self.t = np.arange(t_start, num_timesteps*dt, dt)

        # wavelength
        self.wl = wl
        
        # use delays
        # (set to False to speed up frequency calculations with constant source)
        self.use_delays = use_delays

        # name
        self.name = name

    @property
    def num_timesteps(self):
        return self.t.shape[0]
    @num_timesteps.setter
    def num_timesteps(self, value):
        self.t = np.arange(self.t_start, value*self.dt, self.dt)
    
    @property
    def dt(self):
        return self._dt
    @dt.setter
    def dt(self, value):
        self._dt = value
        self.t = np.arange(self.t_start, self.t_end, value)
    
    @property
    def t_start(self):
        return self.t[0]
    @t_start.setter
    def t_start(self, value):
        self.t = np.arange(value, self.t_end, self.dt)
    
    @property
    def t_end(self):
        return self.t[-1] + self.dt
    @t_end.setter
    def t_end(self, value):
        self.t = np.arange(self.t_start, value, self.dt)
        