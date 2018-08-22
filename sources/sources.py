''' Input Sources '''

#############
## Imports ##
#############

import torch
import numpy as np



############
## Source ##
############

class Source(object):
    ''' Photontorch Source '''
    nw = None
    def __init__(self, array, axes=None):
        ''' Source initialization

            Args:
                array (np.array): the source data
                axes (list[str]) = named axis order of the array.
                    defaults:
                        array.ndim == 1 -> axes=['time']
                        array.ndim == 2 -> axes=['time','batches']
                        array.ndim == 3 -> axes=['time','sources','batches']
                        array.ndim == 4 -> axes=['time','wavelengths','sources','batches']
                    possible axes keys:
                        'time' or 't'
                        'wavelengths' or 'w'
                        'sources' or 's'
                        'batches' or 'b'
        '''
        array = np.array(array, 'complex64')
        keys = {'time':'t','t':'t','wavelengths':'w','w':'w',
                'sources':'s','s':'s','batches':'b','b':'b'}
        if axes is None:
            axes = {
                0:[],
                1:['t'],
                2:['t','b'],
                3:['t','s','b'],
                4:['t','w','s','b'],
            }[array.ndim]
        else:
            axes = [keys[ax] for ax in axes]

        num_sources_known = 's' in axes

        for c in ['t','w','s','b']:
            if c not in axes:
                array = array[...,None]
                axes.append(c)

        order = []
        for c in ['t','w','s','b']:
            order.append(axes.index(c))

        array = array.transpose(order)
        axes = ['t','w','s','b']
        self.nt, self.nwl, self.ns, self.nb = array.shape


        if self.nwl > 1 and self.nwl != self.nw.env.num_wl:
            raise ValueError('Source array does not specify enough wavelengths')

        source = np.zeros((self.nt, self.nwl, self.nw.nmc, self.nb), 'complex64')
        if num_sources_known:
            source[:,:,:self.ns,:] = array
        else:
            source[:,:,:self.nw.num_sources,:] = array

        self.source = torch.stack([torch.tensor(np.real(source)), torch.tensor(np.imag(source))], 0)

        if self.nw.is_cuda:
            self.source = self.source.cuda()

    @property
    def shape(self):
        ''' get shape of tensor this source is trying to emulate '''
        return (2, self.nw.env.num_timesteps, self.nw.env.num_wl, self.nw.nmc, self.nb)

    def __getitem__(self, key):
        key = (key[0]%2, key[1]%self.nt) + key[2:]
        return self.source[key]


class ConstantSource(Source):
    ''' Photontorch Constant Source '''
    def __init__(self, amplitude=1.0):
        ''' ConstantSource initialization

            Args:
                amplitude (float | np.array): the source data.
                    if amplitude.ndim == 1 -> different amplitude for each batch is specified
                        with same amplitude at each of the sources. -> size = [batch_dim]
                    if amplitude.ndim == 2 -> different amplitude for each batch is specified
                        with different amplitude at each of the sources -> size = [source_dim, batch_dim]
        '''
        array = np.array(amplitude, 'complex64')

        if array.ndim > 2:
            raise ValueError('Amplitude should be max 2D')

        a1 = ['b']
        if array.ndim == 1 and array.shape[0] == self.nw.num_sources:
            a1 = ['s']

        axes = {
            0:[],
            1:a1,
            2:['s','b'],
        }[array.ndim]

        Source.__init__(self, array, axes=axes)
