''' Simulation Environment Module '''


#################
## Environment ##
#################

class Environment(object):
    ''' Simulation Environment '''
    def __init__(self, dt=None, wl=None):

        # timestep
        self.dt = dt

        # wavelength
        self.wl = wl
