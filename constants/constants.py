'''
# Constants
Useful constants for photonic simulations

'''

#############
## Imports ##
#############

import numpy as np


###############
## Constants ##
###############

# pi
pi = np.pi

# speed of light
c = 299792458.0

# standard room temperature:
T0 = 300

# conversion between FWHM and sigma:
sigma2fwhm = 2*np.sqrt(2*np.log(2))
fwhm2sigma = 1.0/sigma2fwhm
