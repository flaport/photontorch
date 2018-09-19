
'''
# SOAs

'''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .component import Component
from ..torch_ext.nn import Parameter, Buffer
from ..constants.constants import pi, c, q, h


################
## Simple SOA ##
################

class LinearSoa(Component):
    ''' A Linear SOA is a memory-less component with one input and one output.

    It amplifies a signal instantaneously and linearly with the specified amplification factor

    A simple SOA has one trainable parameter: the amplification.

    Terms:

        0 ---- 1

    '''

    num_ports = 2

    def __init__(self, amplification=2, trainable=True, name=None):
        ''' SOA

        Args:
            amplification (float). Amplification of the soa
            trainable (bool): Whether the amplification is trainable
            name (str): name of this specific soa
        '''
        super(LinearSoa, self).__init__(name=name)

        parameter = Parameter if trainable else Buffer
        self.amplification = parameter(torch.tensor(float(amplification), device=self.device))

    def get_S(self):
        ''' Scattering matrix with shape: (2, # wavelengths, # ports, # ports) '''
        S = torch.zeros((2, self.env.num_wl, self.num_ports, self.num_ports), device=self.device)
        S[0,:,0,1] = self.amplification
        return S


class BaseSoa(Component):
    ''' A unidirectional Base Soa '''

    num_ports = 3

    def function(self, t, x_in, x_out):
        ''' Action of the soa on the input state, output state and its internal state '''
        a_in, h, a_out = x_in

        # input amplitude
        x_out[0] = a_in

        # internal state
        x_out[1] = h + self.env.dt*self.dhdt(t, h, a_in)

        # output amplitude
        x_out[2] = a_in*torch.exp(0.5*h)
        #x_out[2] = a_in*(1 + 0.5*h + 0.25*0.5*h**2 + 0.125*(1.0/6.0)*h**3)

    def dhdt(self, t, h, a):
        return 0.0

    def get_functions_at(self):
        return torch.ones(self.num_ports, device=self.device, dtype=torch.uint8)

    def get_S(self):
        ''' Scattering matrix with shape: (2, # wavelengths, # ports, # ports) '''
        S = torch.zeros((2, self.env.num_wl, self.num_ports, self.num_ports), device=self.device)
        S[0,:,0,0] = 1.0
        S[0,:,1,1] = 1.0
        S[0,:,2,2] = 1.0
        return S

    def get_C(self):
        ''' Scattering matrix with shape: (2, # wavelengths, # ports, # ports) '''
        S = torch.zeros((2, self.num_ports, self.num_ports), device=self.device)
        S[0,1,1] = 1.0 # the internal state should be connected onto itself to keep track of it.
        return S


class Soa(BaseSoa):
    ''' A unidirectional Soa with trainable amplification

    The Soa amplifies the signal with a certain amplification.
    The amplification is reached after a certain startup time.

    '''

    def __init__(self, amplification, startup_time = 100e-12, trainable=False, name=None):
        ''' Soa

        Args:
            amplification (float): the maximum amplification of the soa
            startup_time (float): how long it takes before the soa reaches max amplification
            trainable (bool): if the amplification is trainable
            name (str): name of the soa
        '''
        super(Soa, self).__init__(name=name)

        if amplification < 1:
            raise ValueError('Amplification should be bigger than 1.')

        self.startup_time = startup_time

        parameter = Parameter if trainable else Buffer
        self.amplification = parameter(torch.tensor(float(amplification), device=self.device))

    def dhdt(self, t, h, a):
        if t < self.startup_time:
            return (1.0/self.startup_time)*torch.log(self.amplification)
        return 0


class AgrawalSoa(BaseSoa):
    ''' An unidirectional SOA based on the Agrawal model:

    Note:
        An AgrawalSoa is not trainable for now.

    Ref:
        Agrawal, G.P. and Olsson, N.A., 1989. Self-phase modulation and spectral broadening of
        optical pulses in semiconductor laser amplifiers. IEEE Journal of Quantum Electronics,
        25(11), pp.2297-2306.
    '''

    def __init__(self,
        L = 500e-6, # length of soa
        W = 2e-6, # width of soa
        H = 0.2e-6, # height of soa
        N0 = 1e24, # transparency carrier density
        a = 2.7e-20, # differenctial gain coefficient
        I = 0.56/3.0, # current through soa
        tc = 300e-12, # lifetime of the carriers
        gamma = 0.3, # confinement factor
        alpha = 5.0, # linewidth enhancement
        neff = 2.86, # effective index used to calculate phase offset
        ng = 3.75, # group index used to calculate delay of soa
        wl = 1.55e-6, # wavelength of the simulation
        name=None):
        ''' Agrawal Soa

        Args (float):
            L = 500e-6: length of soa
            W = 2e-6: width of soa
            H = 0.2e-6: height of soa
            N0 = 1e24: transparency carrier density
            a = 2.7e-20: differenctial gain coefficient
            I = 0.56/3.0: current through soa
            tc = 300e-12: lifetime of the carriers
            gamma = 0.3: confinement factor
            alpha = 5.0: linewidth enhancement
            neff = 2.86: effective index used to calculate phase offset
            ng = 3.75: group index used to calculate delay of soa
            wl = 1.55e-6: wavelength of the simulation

        Args (other):
            name (str): name of the soa
        '''

        super(AgrawalSoa, self).__init__(name=name)

        ## base parameters
        self.L = L # length of soa
        self.W = W # width of soa
        self.H = H # height of soa
        self.N0 = N0 # transparency carrier density
        self.a = a # differenctial gain coefficient
        self.I = I # applied current through soa
        self.tc = tc # lifetime of the carriers
        self.gamma = gamma # confinement factor
        self.ng = ng # group index used to calculate delay of soa
        self.neff = neff # effective index used to calculate phase offset
        self.alpha = alpha # linewidth enhancement
        self.wl = wl # wavelength of the simulation

    def initialize(self, env):
        ''' Initialize component and calculate derived parameters '''
        ## derived parameters
        self.V = self.L*self.H*self.W # volume of soa
        self.Ith = q*self.V*self.N0/self.tc # current threshold
        self.g0 = self.gamma*self.a*self.N0*(self.I/self.Ith - 1) # small signal gain
        self.Psat = (h*c/self.wl)*self.W*self.H/(self.gamma*self.tc*self.a) # saturation power of the SOA
        self.delay = self.L*self.ng/c # delay introduced by the soa
        self.phase = self.L*self.neff*2*pi*env.wl # phase introduced by the soa

        super(AgrawalSoa, self).initialize(env)

    def get_delays(self):
        ''' Delays introduced by the SOA '''
        return torch.tensor([self.delay,0,0], device=self.device, dtype=torch.get_default_dtype())

    def dhdt(self, t, h, a):
        ''' Internal state equation '''

        # input power
        P = torch.sum(a**2, 0)

        # real part of internal state
        h_real = h[0]

        # real part of dhdt
        dhdt_real = (1.0/self.tc)*((self.g0*self.L - h_real) - (P/self.Psat)*(torch.exp(h_real) - 1))

        # full dhdt
        dhdt = torch.stack([dhdt_real, torch.zeros_like(dhdt_real)], 0)

        # return result
        return dhdt
