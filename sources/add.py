'''
## Source Injection in Network
By calling the `add_sources` function in the __init__ of a network, all source classes
will be available as attributes to this network. This Feature is used by all the
Network classes. This way, it is possible to create a new Source with the reference to
the Network that created it, without needing to explicitly provide the network in the
__init__ call of the source.
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
        return issubclass(cls, sources.Source)
    except TypeError:
        return False


#############
## Sources ##
#############

SOURCES = {k:v for k,v in sources.__dict__.items() if is_source(v)}


#####################
## Source injector ##
#####################

def add_sources(network):
    ''' All Sources become attributes of the network that calls this function.

    Note:
        This method should be called just once during __init__.
    '''
    for sourcename, Source in SOURCES.items():  # loop over all sources
        # make a dedicated subclass for the network:
        NetworkSource = type(sourcename, (Source,), {'nw':network})
        # Save the source class inside the network.
        setattr(network, sourcename, NetworkSource)
