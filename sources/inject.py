'''
# Source Injection
This module defines an injector class.
Subclassing this injector class and running self._inject_sources() during initialization
will result in copying all the Source Classes into the new class.

## Source Injection in Network
This Feature is used by all the Network classes. This way, it is possible to create a
new Source with the reference to the Network that created it, without needing to
explicitly provide the network in the __init__ call.
'''

#############
## Imports ##
#############

from . import sources


############
## Useful ##
############

def is_source(cls):
    ''' Check if a specific class is a Source '''
    try:
        return issubclass(cls, sources.BaseSource)
    except TypeError:
        return False


#############
## Sources ##
#############

SOURCES = {k:v for k,v in sources.__dict__.items() if is_source(v)}


#####################
## Source injector ##
#####################

class SourceInjector(object):
    '''
    Special class that injects all the Source classes into the Network class
    '''
    def _inject_sources(self):
        ''' All Sources become attributes of the class that called this function.

        Note:
            This method should be called at most once, during the Network __init__.
        '''
        for sourcename, Source in SOURCES.items():  # loop over all sources
            # make a copy of the source class
            SourceCopy = type(sourcename, Source.__bases__, dict(Source.__dict__))
            # Set the reference to the right kind of network
            SourceCopy.nw = self
            # Save the source class inside the network.
            setattr(self, sourcename, SourceCopy)
