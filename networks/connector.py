'''
# Connector

The connector module provides the Connector class, a helper class to connect Components
together into a Network.
'''

###############
## Connector ##
###############

class Connector(object):
    '''
    The connector object is an abstract object that is invoked
    while connecting components into a network.

    It is merely there as an aid for connecting components together
    in a simplified fashion.

    Example:
        ring = wg['ij'] * wg['jk'] * wg['ki']

        during the creating of the ring, three connector components are created:
        wg['ij'] = Conn('ij', wg)
        wg['jk'] = Conn('jk', wg)
        wg['ki'] = Conn('ki', wg)

        When connectors are multiplied together, intermediate connectors are created:
        wg['ij']*wg['jk'] = Conn('ij,jk', [wg, wg])

        When all indexes are connected (there are exactly two occurences for each index), such as
        ring = wg['ij'] * wg['jk'] * wg['ki'] = Conn('ij,jk,ki',[wg,wg,wg])
        A new network object with the corresponding indices and components is automatically created.


    Useful Attribute:
        get the free indices of a connector by calling the .idxs attribute:

        In[1]:  conn.idxs
        Out[1]: 'ijkl'

    Notes:
        - Do NOT create a connector directly. In stead create a connector via the
          __getitem__ method of a component by passing a string:

          conn = comp['ijkl']

        - As an alternative to a string index, a tuple of integers can be given. These
          integers will be converted internally to a string:

          conn = comp['ijkl'] = comp[106,107,108,109]

          This can be useful for creating very large networks and when you run
          out of strings to index (Note that in python 2, the maximum integer number
          is 256, as unicode is not natively supported)

        - The connector does not check whether the number of indices given does correspond
          to the number of ports of the component.

    '''
    def __init__(self, s, components):
        ''' Connector
        Please do not use this initializer directly. In stead use the __getitem__ of a component.
        comp['ijkl'] -> Conn('ijkl',[comp])
        '''
        if not isinstance(s, (str, tuple, list)):
            s = [s]
        if isinstance(s, (tuple, list)):
            s = ''.join(i if isinstance(i, str) else chr(i) for i in s)
        if len(s) > 0  and s[0] == ',':
            self.__init__(s[1:], components)
            return
        if len(s) > 0 and s[-1] == ',':
            self.__init__(s[:-1], components)
            return
        self.s = s
        self.components = components
    
    def copy(self):
        return Connector(self.s, [comp.copy() for comp in self.components])

    @property
    def idxs(self):
        ''' Get free indices of the connector object '''
        return self._idxs(self.s)

    def _idxs(self, s):
        if ',' in s:
            return self._idxs(s.replace(',', ''))
        for c in s:
            if s.count(c) > 2:
                raise ValueError('Index %s occuring more than 2 times!'%c)
            if s.count(c) == 2:
                return self._idxs(s.replace(c, ''))
        return s

    def __mul__(self, other):
        if isinstance(other, Connector):
            s = ','.join(self.s.split(',')+other.s.split(','))
            return Connector(s, tuple(self.components)+tuple(other.components))
        else:
            raise ValueError('Cannot connect %s and %s'%(repr(self), repr(other)))

    def __imul__(self, other):
        if isinstance(other, Connector):
            self.s = ','.join(self.s.split(',')+other.s.split(','))
            self.components += tuple(other.components)
        else:
            raise ValueError('Cannot connect components %s and %s'%(repr(self), repr(other)))

    def __repr__(self):
        ret = []
        for s, c in zip(self.s.split(','), self.components):
            ret += [repr(c) + '_' + s]
        return '*'.join(ret)
