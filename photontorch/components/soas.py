""" Semiconductor Optical Amplifiers

SOAs amplify your signal.

"""

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
from ..environment import current_environment


################
## Simple SOA ##
################


class LinearSoa(Component):
    """ A Linear SOA is a memory-less component with one input and one output.

    It amplifies a signal instantaneously and linearly with the specified amplification factor

    A simple SOA has one trainable parameter: the amplification.

    Terms::

        0 ---- 1

    """

    num_ports = 2

    def __init__(self, amplification=2, trainable=True, name=None):
        """
        Args:
            amplification (float): Amplification of the soa
            trainable (bool): makes the amplification trainable
            name (str): the name of the component (default: lowercase classname)
        """
        super(LinearSoa, self).__init__(name=name)

        parameter = Parameter if trainable else Buffer
        self.amplification = parameter(
            torch.tensor(float(amplification), device=self.device)
        )

    def set_S(self, S):
        S[0, :, 0, 1] = self.amplification
        return S


class BaseSoa(Component):
    """ The BaseSoa is a memory-containing component with one input and one output.

    It amplifies a signal according to its internal state, which in turn is
    modified by its rate equation.

    Terms::

        0 ---- 1

    """

    num_ports = 3

    def action(self, t, x_in, x_out):
        """ Nonlinear action of the component on its active nodes

        Args:
            t (float): the current time in the simulation
            x_in (torch.Tensor[#active nodes, 2, #wavelengths, #batches]): the input tensor
                used to define the action
            x_out (torch.Tensor[#active nodes, 2, #wavelengths, #batches]): the output
                tensor. The result of the action should be stored in the
                elements of this tensor.

        """
        a_in, h, _ = x_in  # unpack the three active nodes

        # input amplitude
        x_out[0] = a_in  # nothing happens to the input active node

        # the internal state is modified by its state equation:
        x_out[1] = h + self.env.dt * self.dhdt(t, h, a_in)

        # the output amplitude is modified by the internal state
        x_out[2] = a_in * torch.exp(0.5 * h)
        # x_out[2] = a_in*(1 + 0.5*h + 0.25*0.5*h**2 + 0.125*(1.0/6.0)*h**3)

    def dhdt(self, t, h, a):
        """ Derivative of the internal state h with respect to time

        Args:
            t (float): the current time in the simulation
            h (Tensor[2, #wavelengths, #batches]): the current internal state
                of the SOA
            a (Tensor[2, #wavelengths, #batches]): the current input amplitude
                of the SOA

        Returns:
            Tensor[2, #wavelengths, #batches]: the rate of change of the
            internal state
        """

        return 0.0

    def set_actions_at(self, actions_at):
        actions_at[:] = 1

    def set_S(self, S):
        S[0, :, 0, 0] = 1.0
        S[0, :, 1, 1] = 1.0
        S[0, :, 2, 2] = 1.0
        return S

    def set_C(self, C):
        C[0, 1, 1] = 1.0 # the internal state should be connected onto itself.


class Soa(BaseSoa):
    """ The Soa is a memory-containing component with one input and one output.

    It amplifies a signal according to its internal state, which in turn is
    modified by its rate equation.

    Terms::

        0 ---- 1

    """

    def __init__(
        self, amplification=1.0, startup_time=100e-12, trainable=False, name=None
    ):
        """
        Args:
            amplification (float): the maximum amplification of the soa
            startup_time (float): how long it takes before the soa reaches max amplification
            trainable (bool): makes the amplification trainable
            name (str): the name of the component (default: lowercase classname)
        """
        super(Soa, self).__init__(name=name)

        if amplification < 1:
            raise ValueError("Amplification should be bigger than 1.")

        self.startup_time = startup_time

        parameter = Parameter if trainable else Buffer
        self.amplification = parameter(
            torch.tensor(float(amplification), device=self.device)
        )

    def dhdt(self, t, h, a):
        if t < self.startup_time:
            return (1.0 / self.startup_time) * torch.log(self.amplification)
        return 0


class AgrawalSoa(BaseSoa):
    """ The AgrawalSoa is a memory-containing component with one input and one output.

    It amplifies a signal according to its internal state, which in turn is modified by
    its rate equation defined by Agrawal et. al.

    Terms::

        0 ---- 1

    Reference:
        Agrawal, G.P. and Olsson, N.A., 1989. Self-phase modulation and spectral
        broadening of optical pulses in semiconductor laser amplifiers. IEEE Journal of
        Quantum Electronics, 25(11), pp.2297-2306.

    """

    def __init__(
        self,
        L=500e-6,  # length of soa
        W=2e-6,  # width of soa
        H=0.2e-6,  # height of soa
        N0=1e24,  # transparency carrier density
        a=2.7e-20,  # differenctial gain coefficient
        I=0.56 / 3.0,  # current through soa
        tc=300e-12,  # lifetime of the carriers
        gamma=0.3,  # confinement factor
        alpha=5.0,  # linewidth enhancement
        neff=2.34,  # effective index used to calculate phase offset
        ng=3.75,  # group index used to calculate delay of soa
        wl=1.55e-6,  # wavelength of the simulation
        name=None,
    ):
        """
        Args:
            L (loat): length of soa
            W (loat): width of soa
            H (loat): height of soa
            N0 (loat): transparency carrier density
            a (loat): differenctial gain coefficient
            I (loat): current through soa
            tc (loat): lifetime of the carriers
            gamma (float): confinement factor
            alpha (float): linewidth enhancement
            neff (float): effective index used to calculate phase offset
            ng (float): group index used to calculate delay of soa
            wl (float): wavelength of the simulation
            name (str): the name of the component (default: lowercase classname)
        """

        super(AgrawalSoa, self).__init__(name=name)

        ## base parameters
        self.L = L  # length of soa
        self.W = W  # width of soa
        self.H = H  # height of soa
        self.N0 = N0  # transparency carrier density
        self.a = a  # differenctial gain coefficient
        self.I = I  # applied current through soa
        self.tc = tc  # lifetime of the carriers
        self.gamma = gamma  # confinement factor
        self.ng = ng  # group index used to calculate delay of soa
        self.neff = neff  # effective index used to calculate phase offset
        self.alpha = alpha  # linewidth enhancement
        self.wl = wl  # wavelength of the simulation

    def initialize(self):
        self._env = env = current_environment()

        # elementary charge
        q = 1.6e-19
        # planck constant
        h = 6.626e-34
        ## derived parameters
        self.V = self.L * self.H * self.W  # volume of soa
        self.Ith = q * self.V * self.N0 / self.tc  # current threshold
        self.g0 = (
            self.gamma * self.a * self.N0 * (self.I / self.Ith - 1)
        )  # small signal gain
        self.Psat = (
            (h * env.c / self.wl) * self.W * self.H / (self.gamma * self.tc * self.a)
        )  # saturation power of the SOA
        self.delay = self.L * self.ng / env.c  # delay introduced by the soa
        self.phase = (
            self.L
            * self.neff
            * 2
            * np.pi
            * np.mean(env.wavelength)  # TODO: make this an array
        )  # phase introduced by the soa

        super(AgrawalSoa, self).initialize()
        return self

    def set_delays(self, delays):
        delays[0] = self.delay

    def dhdt(self, t, h, a):
        # input power
        P = torch.sum(a ** 2, 0)

        # real part of internal state
        h_real = h[0]

        # real part of dhdt
        dhdt_real = (1.0 / self.tc) * (
            (self.g0 * self.L - h_real) - (P / self.Psat) * (torch.exp(h_real) - 1)
        )

        # full dhdt
        dhdt = torch.stack([dhdt_real, torch.zeros_like(dhdt_real)], 0)

        # return result
        return dhdt
