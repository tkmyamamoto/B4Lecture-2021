import numpy as np
import librosa

from modules.stft import stft, faster_stft, instft


def spectrogram(wave_data, win_size=1024, overlap=0.5, mode='normal', scale='db'):
    '''
    Compute a spectrogram with short-time Fourier transforms.
    Parameters
    ----------
    wave_data: ndarray
        waveform data
    win_size: int
        window size
    overlap: float
        overlap size
    mode: 'normal' or 'faster'
        using stft function
    scale: 'db' or 'amp'
        return value scale. amplitude or db
    Returns
    -------
    spec: ndarray
        spectrogram
    '''

    if mode not in ['normal', 'faster']:
        raise ValueError(
            "Unknown value for mode {}, must be one of {'normal', 'faster'}".format(mode))

    if scale not in ['db', 'amp']:
        raise ValueError(
            "Unknown value for scale {}, must be one of {'db', 'amp'}".format(mode))

    func = stft
    if mode == 'faster':
        func = faster_stft

    spec = func(wave_data, win_size=win_size, overlap=overlap)

    if scale == 'db':
        # extract magnitude (amplitude spectrum) and phase (phase spectrum)
        # amplitude, phase = librosa.magphase(cs)
        # amplitude -> db
        spec = librosa.amplitude_to_db(np.abs(spec))
        # spec = np.log10(np.abs(spec)) * 10

    return spec
