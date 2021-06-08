import numpy as np
import scipy as sp
from scipy import linalg

from .autocorrelation import autocorrelation
from modules.utils import get_lb_method_error


def toeplitz_solver(ac, dim):
    '''
    toeplitz_solver solve toeplitz matrix.

    Parameters
    ----------
    ac: ndarray
        autocorrelation of sequence data
    dim: int
        dimention of coef

    Returns
    -------
    a: ndarray
        predicted coefficient
    e: float
        prediction error
    '''

    a = np.zeros(dim+1)
    a[0] = 1
    a[1] = -ac[1]/ac[0]

    e = ac[0] + (ac[1]*a[1])

    for i in range(2, dim + 1):
        k = -np.sum(a[:i] * ac[i:0:-1]) / e
        a[:i+1] += k * a[:i+1][::-1]
        e *= 1 - k * k

    return a, e


def lpc_method(wave_data, mode='mine', dim=100, clip_size=None):
    '''
    lpc_method extract spectrum envelope by LPC method

    Parameters
    ----------
    wave_data: ndarray (1d)
        audio data
    mode: 'mine' or 'scipy'
        toeplitz metrix solver type
    dim: int
        dimention of coef
    clip_size: int
        length of clipping data
        if None, length of wave_data

    Returns
    -------
    lpc_env: ndarray (1d)
        spectrum envelope
    '''

    if clip_size is None:
        clip_size = len(wave_data)

    ac = autocorrelation(wave_data)
    if mode == 'mine':
        a, e = toeplitz_solver(ac, dim)
    elif mode == 'scipy':
        a = linalg.solve_toeplitz((ac[:dim-1], ac[:dim-1]), ac[1:dim])
        e = get_lb_method_error(ac, a, dim) / 1000

    _, h = sp.signal.freqz(np.sqrt(e), a, clip_size, "whole")
    lpc_env = 20 * np.log10(np.abs(h))

    return lpc_env
