import numpy as np

from .utils import get_framing_data


def get_cepstrum(input, is_clipping, is_framing,  clip_size=2048, win_size=1024, overlap=0.5):
    '''
    get_cepstrum extract cepstrum from input data

    Parameters
    ----------
    input: ndarray (1d)
        input data
    is_framing: bool
        whether to framing data
    is_clipping: bool
        whether to clip data
    clip_size: int
        clipping size, when clip data
    win_size: int
        window size, when framing data
    overlap: float
        overlap value, when framin data

    Returns
    -------
    ceps_db: ndarray (1d)
        cepstrum (db)
    '''

    assert not (is_clipping and is_framing), 'The two values (is_clipping, is_framing) ​​cannot be True same time'

    if is_clipping:
        spec = np.fft.rfft(input, clip_size)

    if is_framing:
        framing_data = get_framing_data(input, win_size, overlap)
        spec = np.fft.rfft(framing_data)

    spec_db = 20 * np.log10(np.abs(spec))
    ceps_db = np.fft.irfft(spec_db)

    return ceps_db.real


def get_ceps_peaks(ceps, sr, max=300, min=50):
    '''
    get_ceps_peak detect peaks from given cepstrun

    Parameters
    ----------
    ceps: ndarrray (1d)
        cepstrum (db)
    sr: int
        sampling rate (Hz)
    max: int
        max db
    min: int
        min db

    Returns
    -------
    peak_idxs: ndarray
        list of peak index
    '''

    max_cep_order = int(np.floor(sr / min))
    min_cep_order = int(np.floor(sr / max))

    peak_idxs = np.argmax(ceps[:, min_cep_order:max_cep_order], axis=1)
    peak_idxs = peak_idxs + min_cep_order

    return peak_idxs


def get_envelope(ceps, coef=20):
    '''
    get_envelope extract envelope from given cepstrum

    Parameters
    ----------
    ceps: ndarray (1d)
        cepstrum (db)
    coef: int
        lifter threshold

    Returns
    -------
    envelope: ndarray
        spectrum envelope
    '''

    ceps_liftered = ceps.copy()
    ceps_liftered[coef:-coef] = 0

    envelope = np.fft.rfft(ceps_liftered)

    return envelope
