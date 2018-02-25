'''
This is module defines an injector class. Subclassing this injector class
results in inheriting all Sources defined in source.py
'''

#############
## Imports ##
#############

from . import source


############
## Useful ##
############

def is_subclass(cls, base_cls):
    try:
        return issubclass(cls, base_cls)
    except TypeError:
        return False


#############
## Sources ##
#############

SOURCES = {k:v for k,v in source.__dict__.items() if is_subclass(v, source.BaseSource)}


#####################
## Source injector ##
#####################

class SourceInjector(object):
    '''
    Special class that injects all the Source classes into the Network class
    '''
    def inject_sources(self):
        ''' All Sources become attributes of the SourceInjector '''
        for sourcename, Source in SOURCES.items():
            SourceCopy = type(sourcename, Source.__bases__, dict(Source.__dict__))
            SourceCopy.nw = self
            setattr(self, sourcename, SourceCopy)
