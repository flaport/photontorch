''' Normal numpy function extensions that do not require autograd '''

#############
## Imports ##
#############

import numpy as np


###############
## functions ##
###############

def inv_sigmoid(x, eps=1e-8):
    ''' Inverse sigmoid function '''
    if x >= 1:
        x = 1 - eps
    elif x <= 0:
        x = eps
    return -np.log(1/x-1)
