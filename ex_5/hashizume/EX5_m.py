import numpy as np
import sys
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal
from scipy.fftpack.realtransforms import dct


def hz_mel(f):
    """
    helz => mel scale
    """
    return 2595 * np.log(f / 700.0 + 1.0)


def mel_hz(m):
    """
    mel scale => helz
    """
    return 700 * (np.exp(m / 2595) - 1.0)


def melFilterBank(fs, N, n_Channels):
    """
    create melFilterBank

    Parameters
    ----------
    fs : sampling frequency
    N : number of samples of FFT
    n_Channel : number of filters

    returns
    -------
    filterbank
    """
    # Nyquist frequency(Hz)
    fmax = fs / 2
    # Nyquist frequency(mel)
    melmax = hz_mel(fmax)
    # max index of frequency
    nmax = N // 2
    # frequency resolution
    df = fs / N
    # the center frequency of each filter no Mel scale
    dmel = melmax / (n_Channels + 1)
    melcenters = np.arange(1, n_Channels + 1) * dmel
    # convert the center frequency of each filter to frequency
    fcenters = mel_hz(melcenters)
    # convert to index of frequency
    indexcenter = np.round(fcenters / df)
    # index of the start position of each filter
    indexstart = np.hstack(([0], indexcenter[0 : n_Channels - 1]))
    # index of the stop position of each filter
    indexstop = np.hstack((indexcenter[1:n_Channels], [nmax]))
    filterbank = np.zeros((n_Channels, nmax))
    for c in range(0, n_Channels):
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        for i in range(int(indexstart[c]), int(indexcenter[c])):
            filterbank[c, i] = (i - indexstart[c]) * increment
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in range(int(indexcenter[c]), int(indexstop[c])):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank


def gene_mfcc(s, fs, nperseg, filterbank):
    f, t, spec = signal.stft(s, fs=fs, nperseg=nperseg)
    mspec = np.dot(filterbank, np.abs(spec[:-1]))
    mspec_db = librosa.amplitude_to_db(mspec)
    ceps = dct(mspec_db, axis=0)
    mfcc = ceps[1:13]
    return spec, mspec_db, mfcc


def main():
    args = sys.argv
    wav_filename = args[1]
    s, fs = librosa.load(wav_filename)
    N = 2048

    n_Channels = 20
    filterbank = melFilterBank(fs, N, n_Channels)

    spec, mspec_db, mfcc = gene_mfcc(s, fs, N, filterbank)

    librosa.display.specshow(mfcc, sr=fs, x_axis="time", y_axis="log")
    plt.colorbar()
    plt.title("mfcc")
    plt.savefig("mfcc")
    plt.clf
    plt.close

    librosa.display.specshow(
        librosa.amplitude_to_db(spec), sr=fs, x_axis="time", y_axis="log"
    )
    plt.colorbar()
    plt.title("spectrogram")
    plt.savefig("spec")
    plt.clf
    plt.close

    librosa.display.specshow(mspec_db, sr=fs, x_axis="time", y_axis="log")
    plt.colorbar()
    plt.title("mel-spectrogram")
    plt.savefig("melspec")
    plt.clf
    plt.close


if __name__ == "__main__":
    main()
