''' connector tests '''

#############
## Imports ##
#############

import torch
import pytest
import photontorch as pt

from photontorch.networks.connector import Connector

from fixtures import conn, wg, s, d

###########
## Tests ##
###########

def test_connector(conn):
    pass

def test_connector_init(s):
    conn = s[ord('a')]
    assert conn.s == 'a'
    conn = s[',a,,,']
    assert conn.s == 'a'

def test_connector_idxs(wg,s,d):
    assert (wg['ac']*s['a']*d['b']).idxs == 'cb'

def test_connector_idxs_appearing_too_often(wg,s,d):
    with pytest.raises(ValueError):
        wg['ab']*s['b']*d['b']

def test_multiplication_with_no_connector(conn, wg):
    with pytest.raises(ValueError):
        new_conn = conn*wg

def test_connection_parsing(conn, wg):
    # normal connection
    conn.parse()

    # self connection
    wg['aa'].parse()

    # connection with components with same name
    s = pt.Source(name='s')
    d = pt.Source(name='s')
    comps, conns = (s['a']*d['a']).parse()
    assert comps['s'] is s
    assert comps['s1'] is d



###############
## Run Tests ##
###############

if __name__ == '__main__': # pragma: no cover
    pytest.main([__file__])
