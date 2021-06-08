import numpy as np
import librosa
from scipy.fftpack import dct

import stft


def hz2mel(f, f0=1000):
    m0 = 1000 / np.log(1000/f0 + 1)
    return m0*np.log(f/f0 + 1)


def mel2hz(m, f0=1000):
    m0 = 1000 / np.log(1000/f0 + 1)
    return f0*(np.exp(m/m0) - 1)


def mel_filter_bank(sr, data_size, bank_size):
    mel_max = hz2mel(sr/2)

    n_max = data_size//2

    df = sr/data_size

    dmel = mel_max/(bank_size+1)
    mel_centers = np.arange(1, bank_size+1) * dmel
    f_centers = mel2hz(mel_centers)

    idx_centers = np.round(f_centers / df)
    start_idx = np.hstack(([0], idx_centers[0:bank_size - 1]))
    end_idx = np.hstack((idx_centers[1:bank_size], [n_max]))

    filter_bank = np.zeros((bank_size, n_max))
    for c in range(0, bank_size):
        inc = 1./(idx_centers[c]-start_idx[c])
        for i in np.arange(start_idx[c], idx_centers[c]):
            filter_bank[c, int(i)] = (i-start_idx[c])*inc

        dec = 1./(end_idx[c] - idx_centers[c])
        for i in np.arange(idx_centers[c], end_idx[c]):
            filter_bank[c, int(i)] = 1.0 - ((i - idx_centers[c]) * dec)

    return filter_bank, f_centers


def get_mfcc(input, sr, win_size, overlap, bank_size):
    spec = stft.faster_stft(input, win_size, overlap)[:win_size//2]
    filter_bank, _ = mel_filter_bank(sr, win_size, bank_size)
    mel_spec = np.dot(filter_bank, spec)
    mel_spec_db = librosa.amplitude_to_db(mel_spec)
    ceps = dct(mel_spec_db, axis=0)
    mfcc = ceps[:13]

    return mfcc


def calc_delta(X, width=2):
    #     X = np.pad(X, ((0,0), (width+1, width+1)), 'edge')

    k = np.arange(-width, width+1)
    _sum = np.sum(k**2)

    comp = []
    for i in range(width, X.shape[1]+width):
        try:
            comp.append(np.sum(k*X[:, i-width:i+width+1], axis=1))
        except ValueError:
            break
    comp = np.array(comp).T

    delta = comp/_sum

    return delta
