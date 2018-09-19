'''
# Constants
Useful constants for photonic simulations

'''

#############
## imports ##
#############

from numpy import sqrt, log

###############
## Constants ##
###############

# pi
from numpy import pi

# speed of light
c = 299792458.0

# elementary charge
q = 1.6e-19

# planck constant
h = 6.626e-34

# standard room temperature:
T0 = 300

# conversion between FWHM and sigma:
sigma2fwhm = 2*sqrt(2*log(2))
fwhm2sigma = 1.0/sigma2fwhm
