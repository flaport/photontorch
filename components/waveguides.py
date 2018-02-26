''' Waveguides Module '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .connection import Connection
from ..constants import pi, c

###############
## Waveguide ##
###############

class Waveguide(Connection):
    '''
    A waveguide is a Component where each of the two nodes
    introduces a delay corresponding to the length of the waveguide.

    A waveguide is not trainable (for now)

    Connections
    -----------
    waveguide['ij']:

    i ---- j

    '''

    def __init__(self,
                 length=1e-6,
                 neff=2.86,
                 loss=0,
                 length_bounds=None,
                 phase=None,
                 phase_bounds=(0,2*np.pi),
                 name=None):
        '''
        Waveguide Initialization

        Parameters
        ----------
        length : float. Length of the waveguide in meter.
        neff = 4.0 : float. Effective index of the waveguide
        loss = 0 : float. Loss in the waveguide [dB/m]
        length_bounds : tuple. bounds in which to optimize the length
        phase : float : if a phase is given, the phase introduced
                        by the wavelength becomes decoupled from the length.
                        This can be useful for training purposes.
        phase_bounds : tuple : bounds in which to optimize the phase.
        name : str. Name of the specific waveguide
        '''
        Connection.__init__(self, name=name)
        # Handle inputs
        self.neff = float(neff)
        self.loss = loss
        # as the phase is very sensitive on the length, we need double precision to
        # store the length of the waveguide:
        self.length = self.new_bounded_parameter(
            data=length,
            bounds=length_bounds,
            dtype='double',
            requires_grad=(length_bounds is not None) and (length_bounds[0]!=length_bounds[1]),
        )
        self.phase = None
        if phase is not None:
            self.phase = self.new_bounded_parameter(
                data=phase%(2*np.pi),
                bounds=phase_bounds,
                requires_grad=(phase_bounds is not None) and (phase_bounds[0] != phase_bounds[1]),
            )

    @property
    def delays(self):
        ''' The delay per node is given by the propagation time in the waveguide '''
        delay = self.neff*self.length/c
        return torch.cat([delay, delay]).float()

    @property
    def rS(self):
        ''' real part of the scattering matrix '''
        if self.phase is not None:
            cos_phase = self.phase.cos().double()
        else:
            wls = self.new_variable(self.env.wls, dtype='double')
            cos_phase = torch.cos(2*pi*self.neff*self.length/wls)
        re = 10**(-self.loss*self.length/10)*cos_phase
        # we can safely convert back to single precision now:
        re = re.float()
        S = self.new_variable([[[0, 1],
                                [1, 0]]])
        return re.view(-1,1,1)*S

    @property
    def iS(self):
        ''' imaginary part of the scattering matrix '''
        if self.phase is not None:
            sin_phase = self.phase.sin().double()
        else:
            wls = self.new_variable(self.env.wls, dtype='double')
            sin_phase = torch.sin(2*pi*self.neff*self.length/wls)
        ie = 10**(-self.loss*self.length/10)*sin_phase
        # we can safely convert back to single precision now:
        ie = ie.float()
        S = self.new_variable([[[0, 1],
                                [1, 0]]])
        return ie.view(-1,1,1)*S
