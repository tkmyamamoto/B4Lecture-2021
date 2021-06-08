# include flake8, black

import numpy as np


def stft(y, hop=0.5, win_length=1024):
    """
    Compute the Short Time Fourier Transform (STFT).

    Parameters:
        y : np.ndarray, real-valued
            Time series of measurement values.
        hop : float [scaler], optional
            Hop (Overlap) size.
        win_length : int [scaler], optional
            Window size.

    Return:
        F : np.ndarray
            Complex-valued matrix of short-term Fourier transform coefficients.
    """

    hop_length = int(win_length * hop)
    ynum = y.shape[0]
    window = np.hamming(win_length)

    F = []
    for i in range(int((ynum - hop_length) / hop_length)):
        tmp = y[i * hop_length : i * hop_length + win_length]
        # multiplied by window finction
        tmp = tmp * window
        # Fast Fourier Transform (FFT)
        tmp = np.fft.rfft(tmp)
        F.append(tmp)

    # (frame, freq) -> (freq, frame)
    F = np.transpose(F)
    return F


def istft(F, hop=0.5, win_length=1024):
    """
    Compute the Short Time Fourier Transform (STFT).

    Parameters:
        F : np.ndarray
            Complex-valued matrix of short-term Fourier transform coefficients.
        hop : float [scaler], optional
            Hop (Overlap) size.
        win_length : int [scaler], optional
            Window size.

    Return:
        y : np.ndarray
            Time domain signal.
    """

    hop_length = int(win_length * hop)
    window = np.hamming(win_length)
    # (freq, frame) -> (frame, freq)
    F = np.transpose(F)
    # Inverse Fast Fourier Transform (IFFT)
    tmp = np.fft.irfft(F)
    # divided by window function
    tmp = tmp / window
    # remove overlap
    tmp = tmp[:, :hop_length]
    y = tmp.reshape(-1)

    return y


def pre_emphasis(data, p=0.97):
    """
    Pre-emphasis filter data.

    Parameters:
        data : np.ndarray
            Wave data.
        p : float
            Coefficient of pre-emphasis filter.

    Return:
        pre_data : np.ndarray
            Pre-emphasis filterd data.
    """
    pre_data = np.zeros(len(data))
    for i in range(1, len(data)):
        pre_data[i] = data[i] - p * data[i - 1]
    return pre_data
