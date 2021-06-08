import numpy as np


def stft(data, win_size=1024, overlap=0.5):
    '''
    stft transform data by short-time Fourier transform
    Parameters
    ----------
    data: ndarray
        waveform data
    win_size: int
        window size
    overlap: float
        overlap size
    Returns
    -------
    cv_spec: ndarray
        complex-valued spectrogram
    '''

    data_length = len(data)
    shift_size = int(win_size*overlap)
    window = np.hamming(win_size)

    cv_spec = []
    for j in range(0,  data_length, shift_size):
        x = data[j:j+win_size]
        if win_size > len(x):
            break
        x = window * x
        x = np.fft.fft(x)
        cv_spec.append(x)

    # shape: (ite, complex-valued spectrogram) -> (complex-valued spectrogram, ite)
    cv_spec = np.array(cv_spec).T
    return cv_spec


def faster_stft(data, win_size, overlap):
    n_overlap = int(win_size * overlap)
    window = np.hamming(win_size)

    step = win_size - n_overlap
    shape = data.shape[:-1] + ((data.shape[-1] - n_overlap) // step, win_size)
    strides = data.strides[:-1] + (step * data.strides[-1], data.strides[-1])
    reshaped_data = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides)

    cv_spec = reshaped_data * window
    cv_spec = np.fft.fft(cv_spec, n=win_size)

    return cv_spec.T


def instft(data, win_size=1024, overlap=0.5):
    '''
    instft transform data by inverse short-time Fourier transform
    Parameters
    ----------
    data: ndarray
        complex-valued spectrogram
    win_size: int
        window size
    overlap: float
        overlap size
    Returns
    -------
    wave_data: ndarray
        waveform data
    '''

    shift_size = int(win_size*overlap)
    ite = data.shape[0]
    window = np.hamming(win_size)

    wave_data = np.zeros(ite*shift_size+win_size)
    for i in range(ite):
        x = data[i]
        x = np.fft.ifft(x).real * win_size
        wave_data[i*shift_size:i*shift_size+win_size] = wave_data[
            i * shift_size:i*shift_size+win_size] + x

    return wave_data
