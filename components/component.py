'''
# Base Component Submodule

The base component is a parent class meant for subclassing. It should not be used directly.

Each Component is generally defined by several key properties:

  * num_ports: The number of ports of the components.
  * C: The connection matrix for the component (usually all zero for base components)
  * rS: The real part of the Scattering matrix
  * iS: The imaginary part of the Scattering matrix
  * sources_at: Where there are sources in the component (usually all zero for base components)
  * detectors_at: Where there are detectors in the component (usually all zero for base components)
  * delays: delays introduced by the nodes of the component.

'''

#############
## Imports ##
#############

## Other
import numpy as np

## Relative
from ..torch_ext import where
from ..torch_ext.nn import Module


###############
## Component ##
###############

class Component(Module):
    ''' Generic base component.

        The base component is a parent class meant for subclassing.
        It should not be used directly.
    '''

    num_ports = 1 # Number of ports of the component

    def __init__(self, name=None):
        ''' Component

        Args:
            name (str): the name of the component (defaults to the classname in lowercase)
        '''
        Module.__init__(self)
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = name

    def initialize(self, env):
        ''' Simulation initialization for the component.

        Befor a component can be used for simulation, it should be initialized with a
        simulation environment.

        Args:
            env (Environment): simulation environment.

        Note:
            Just initializing the component with a simulation environment is not enough
            to be able to start a simulation. A connection matrix for which
            C.sum(0) == C.sum(1) == all ones is required.
            You can achieve this by connecting the component into a network with some Terms.
        '''
        self._env = env
        if env.cuda is not None:
            if env.cuda and not self.is_cuda:
                self.cuda()
            elif not env.cuda and self.is_cuda:
                self.cpu()
        self.zero_grad()
        if (self.sources_at & self.detectors_at).any():
            raise ValueError('Sources and Detectors cannot be combined in the same node.')

    @property
    def env(self):
        '''@property
        Returns:
            The environment for which the component is initialized.
        '''
        return self._env

    @property
    def rS(self):
        ''' Real Part of the Scattering matrix of the component.

        Returns:
            torch.FloatTensor with shape (# wavelengths, # ports, # ports)
        '''
        raise NotImplementedError('The real part of the scattering matrix '
                                  'of %s does not exist'%self.name)

    @property
    def iS(self):
        ''' Imaginary Part of the Scattering matrix of the component.

        Returns:
            torch.FloatTensor with shape (# wavelengths, # ports, # ports)
        '''
        raise NotImplementedError('The imaginary part of the scattering matrix '
                                  'of %s does not exist'%self.name)

    @property
    def C(self):
        ''' Connection matrix of the component.

        Returns:
            torch.FloatTensor of only ones and zeros with shape (# num ports, # num ports).
        '''
        return self.new_tensor(np.zeros((self.num_ports, self.num_ports)), 'float')

    @property
    def delays(self):
        ''' Delays introduced by the component.

        Returns:
            torch.FloatTensor with the delays in each node of the component. Shape: (# num ports, )
        '''
        return self.new_tensor(np.zeros(self.num_ports), 'float')

    @property
    def sources_at(self):
        ''' The locations of the sources in the component.

        Returns:
            torch.ByteTensor containing the locations of the sources in the component. shape: (# num ports, )
        '''
        return self.new_tensor(np.zeros(self.num_ports), 'byte')

    @property
    def detectors_at(self):
        ''' The locations of the detectors in the component.

        Returns:
            torch.ByteTensor containing the locations of the detectors in the component. shape: (# num ports, )
        '''
        return self.new_tensor(np.zeros(self.num_ports), 'byte')

    @property
    def free_idxs(self):
        ''' Free indices to make connections to '''
        C = self.C
        return where(((C.sum(0) > 0) | (C.sum(1) > 0)).ne(1).data)

    def __repr__(self):
        ''' String Representation of the component '''
        return self.name

    def __str__(self):
        ''' String Representation of the component '''
        return self.name

    def __getitem__(self, key):
        ''' Special __getitem__

        Each component contains a special __getitem__, which is solely used to connect
        components together. Indexing a component with a string index will create a
        connector with that same index. This connector will connect to any other connector
        where the same string index is used. Connected connectors can be made into a
        network. See Network for more invormation.

        Args:
            key (str): string key of the connection.

        '''
        from ..networks.connector import Connector
        return Connector(key, [self])
