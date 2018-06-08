#############
## Imports ##
#############

# Torch
import torch

# Other
import numpy as np
import matplotlib.pyplot as plt

# Relative
from ..components.terms import Detector


##########
## Plot ##
##########

# The plot function to plot the detected power of a network
def plot(network, detected, **kwargs):
    ''' Plot detected power versus time or wavelength

    Args:
        detected (np.array): detected power. Allowed shapes:
            * (#timesteps,)
            * (#timesteps, #detectors)
            * (#timesteps, #detectors, #batches)
            * (#timesteps, #wavelengths)
            * (#timesteps, #wavelengths, #detectors)
            * (#timesteps, #wavelengths, #detectors, #batches)
            * (#wavelengths,)
            * (#wavelengths, #detectors)
            * (#wavelengths, #detectors, #batches)
        the plot function is smart enough to figure out what to plot.

    Note:
        if #timesteps = #wavelengths, the plotting function will choose #timesteps
        as the first dimension
    '''

    # First we define a helper function
    def plotfunc(x, y, labels, **kwargs):
        ''' Helper function '''
        plots = plt.plot(x,y,**kwargs)
        if labels is not None:
            if len(labels) != len(plots):
                raise ValueError('# labels does not corresponds to # plots')
            for p, l in zip(plots, labels):
                p.set_label(l)
        if labels is not None and len(labels) > 1:
            # Shrink current axis by 10%
            box = plt.gca().get_position()
            plt.gca().set_position([box.x0, box.y0, box.width * 0.9, box.height])
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return plots

    # Handle y
    y = detected
    if torch.is_tensor(y):
        y = y.detach()
    y = np.squeeze(np.array(y, 'float32'))

    # Handle x
    time_mode = wl_mode = False
    if network.env.num_timesteps == y.shape[0]:
        time_mode = True
        x = network.env.t
    elif network.env.num_wl == y.shape[0]:
        wl_mode = True
        x = network.env.wls
    if not (time_mode or wl_mode):
        raise ValueError('First dimension should be #timesteps or #wavelengths')

    # Handle prefixes
    f = (int(np.log10(max(x))+0.5)//3)*3-3
    prefix = {12:'T', 9:'G', 6:'M', 3:'k', 0:'', -3:'m',
                -6:r'$\mu$', -9:'n', -12:'p', -15:'f'}[f]
    x = x*10**(-f)

    # Handle labels
    plt.ylabel('Intensity [a.u.]')
    plt.xlabel('Time [%ss]'%prefix if time_mode else 'Wavelength [%sm]'%prefix)

    # standard labels:
    detectors = [name for name, comp in network.components.items() if isinstance(comp, Detector)]
    wavelengths = ['%inm'%wl for wl in 1e9*network.env.wls]

    # Plot
    if y.ndim == 1:
        return plotfunc(x, y, None, **kwargs)

    if wl_mode:
        if y.ndim == 2:
            if y.shape[1] == network.num_detectors:
                labels = detectors
            else:
                labels = ['batch %i'%i for i in range(y.shape[1])]
            return plotfunc(x, y, labels, **kwargs)
        elif y.ndim == 3:
            y = y.reshape(0,2,1)
            labels = ['%i | %s'%(i, det) for i in range(y.shape[1] for det in detectors)]
            return plotfunc(x, y.reshape(network.env.num_wl, -1), labels, **kwargs)
        else:
            raise ValueError('When plotting in wavelength mode, the max dim of y should be < 4')

    if time_mode:
        if y.ndim == 2:
            if y.shape[1] == network.env.num_wl:
                labels = wavelengths
            elif y.shape[1] == network.num_detectors:
                labels = detectors
            else:
                labels = ['batch %i'%i for i in range(y.shape[1])]
            return plotfunc(x, y, labels, **kwargs)
        elif y.ndim == 3:
            if y.shape[1] == network.env.num_wl and y.shape[2] == network.num_detectors:
                labels = ['%s | %s'%(wl, det) for wl in wavelengths for det in detectors]
            elif y.shape[1] == network.env.num_wl and y.shape[2] != network.num_detectors:
                y = y.transpose(0,2,1)
                labels = ['%i | %s'%(b, wl) for b in range(y.shape[1]) for wl in wavelengths]
            elif y.shape[1] == network.num_detectors:
                y = y.transpose(0,2,1)
                labels = ['%i | %s'%(b, det) for b in range(y.shape[1]) for det in detectors]
            return plotfunc(x, y.reshape(network.env.num_timesteps, -1), labels, **kwargs)
        elif y.ndim == 4:
            y = y.transpose(0,3,1,2)
            labels = ['%i | %s | %s'%(b, wl, det) for b in range(y.shape[1]) for wl in wavelengths for det in detectors]
            return plotfunc(x, y.reshape(network.env.num_timesteps, -1), labels, **kwargs)
        else:
            raise ValueError('When plotting in time mode, the max dim of y should be < 5')

    # we should never get here:
    raise ValueError('Could not plot detected array. Are you sure you are you sure the '
                     'current simulation environment corresponds to the environment for '
                     'which the detected tensor was calculated?')
