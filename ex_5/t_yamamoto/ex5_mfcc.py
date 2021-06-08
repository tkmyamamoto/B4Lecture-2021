# include flake8, black

import argparse
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack.realtransforms import dct
import numpy as np

import utils


def cal_m0(f0):
    """
    Calculate m0.

    Parameters:
        f0 : float
            Basis of frequency [Hz].

    Returns:
        m0 : float
            Basis of mel frequency [mel].
    """
    m0 = 1000.0 / np.log(1000.0 / f0 + 1.0)
    return m0


def freq2mel(f, f0=700.0):
    """
    Convert frequency to mel frequency.

    Parameters:
        f : ndarray
            Frequency.
        f0 : float, default 700.0
            Basis of frequency.

    Returns:
        m : ndarray
            Mel frequency.
    """
    m0 = cal_m0(f0)
    return m0 * np.log(f / f0 + 1.0)


def mel2freq(m, f0=700.0):
    """
    Convert mel frequency to frequency.

    Parameters:
        m : ndarray
            Mel frequency.
        f0 : float, default 700.0
            Basis of frequency.

    Returns:
        f : ndarray
            Frequency.
    """
    m0 = cal_m0(f0)
    return f0 * (np.exp(m / m0) - 1.0)


def melFilterBank(sr, win_length, n_channels):
    """
    Make mel filter bank.

    Parameters:
        sr : int
            Sampling frequency.
        win_length : int
            Window size.
        n_channels : int
            The number of channels.

    Returns:
        filterbank : ndarray
            Mel filter bank.
    """
    # nyquist frequency [Hz]
    f_nyq = sr / 2
    # nyquist frequency [mel]
    mel_nyq = freq2mel(f_nyq)
    # maximum frequency index
    nmax = win_length // 2
    # frequency resolution (Hz width per frequency index 1)
    df = sr / win_length
    # calculate center frequencies in mel-scaled in each filter
    dmel = mel_nyq / (n_channels + 1)
    melcenters = np.arange(1, n_channels + 1) * dmel
    # convert the center frequency to Hz scale in each filter
    fcenters = mel2freq(melcenters)
    # convert the center frequency to frequency index in each filter
    idx_center = np.round(fcenters / df)
    # index of start position of each filter
    idx_start = np.hstack(([0], idx_center[0 : n_channels - 1]))
    # index of end position of each filter
    idx_stop = np.hstack((idx_center[1:n_channels], [nmax]))
    filterbank = np.zeros((n_channels, nmax))
    # print(idx_stop)

    for c in range(0, n_channels):
        # calculate points from the slope of the left line of the triangular filter
        increment = 1.0 / (idx_center[c] - idx_start[c])
        for i in range(int(idx_start[c]), int(idx_center[c])):
            filterbank[c, i] = (i - idx_start[c]) * increment

        # calculate points from the slope of the right line of the triangular filter
        decrement = 1.0 / (idx_stop[c] - idx_center[c])
        for i in range(int(idx_center[c]), int(idx_stop[c])):
            filterbank[c, i] = 1.0 - ((i - idx_center[c]) * decrement)

    return filterbank, fcenters


def calc_mfcc(wav, hop, win_length, filterbank):
    """
    Calculate Mel Frequency Cepstrum Coeffcient(MFCC).

    Parameters:
        wav : ndarray, real-valued
            Time series of measurement values.
        hop : float
            Hop (Overlap) size.
        win_length : int
            Window size.
        filter_bank : ndarray
            mel filter bank

    Returns:
        mel_spec : ndarray (n_channels, n_frames)
            Mel scale spectrogram.
        mfcc : ndarray (n_channels, n_frames)
            Mel Frequency Cepstrum Coeffcient(MFCC).
    """
    pre_wav = utils.pre_emphasis(wav, p=0.97)
    spec = utils.stft(pre_wav, hop=hop, win_length=win_length)
    # hop_length = int(win_length * hop)
    # spec = spec[:, :hop_length]
    mel_spec = np.dot(filterbank, np.abs(spec[:-1]))

    mfcc = np.zeros_like(mel_spec)
    for i in range(mel_spec.shape[1]):
        mfcc[:, i] = dct(mel_spec[:, i], type=2, norm="ortho", axis=-1)

    return mel_spec, mfcc


def delta_mfcc(mfcc, k=2):
    """
    Calculate delta of Mel Frequency Cepstrum Coeffcient(ΔMFCC).
    (References : lecture materials in 2020)

    Parameters:
        mfcc : ndarray (n_channels, n_frames)
            Mel Frequency Cepstrum Coeffcient(MFCC).
        k : int, default 2
            Window of regression.
            The number of frames to see before and after.

    Returns:
        d_mfcc : ndarray (n_channels, n_frames)
            Delta of Mel Frequency Cepstrum Coeffcient(ΔMFCC).
    """
    mfcc_pad = np.pad(mfcc, [(k, k + 1), (0, 0)], "edge")
    k_sq = np.sum(np.arange(-k, k + 1) ** 2)
    m = np.arange(-k, k + 1)
    d_mfcc = np.zeros_like(mfcc)
    for i in range(mfcc.shape[0]):
        d_mfcc[i] = np.dot(m, mfcc_pad[i : i + k * 2 + 1])
    return d_mfcc / k_sq


def main(args):
    """
    fname = "aiueo.wav"
    """

    # get current working directory
    path = os.path.dirname(os.path.abspath(__file__))

    # load audio file
    fname = os.path.join(path, "data", args.fname)
    wav, sr = librosa.load(fname, mono=True)

    # plot signal
    plt.figure()
    ax = plt.subplot(111)
    librosa.display.waveplot(wav, sr=sr, color="g", ax=ax)
    ax.set(title="Original signal", xlabel="Time [s]", ylabel="Magnitude")
    save_fname = os.path.join(path, "result", "signal.png")
    plt.savefig(save_fname, transparent=True)
    plt.show()

    # parameter
    hop = 0.5
    win_length = 1024
    hop_length = int(win_length * hop)

    # make mel filter bank
    n_channels = 20  # the number of mel filter bank channels
    df = sr / win_length  # frequency resolution (Hz width per frequency index 1)
    filterbank, _ = melFilterBank(sr, win_length, n_channels)

    # plot mel filter bank
    for c in range(n_channels):
        plt.plot(np.arange(0, win_length / 2) * df, filterbank[c])

    plt.title("Mel filter bank")
    plt.xlabel("Frequency [Hz]")
    save_fname = os.path.join(path, "result", "MelFilterBank.png")
    plt.savefig(save_fname, transparent=True)
    plt.show()

    # spectrogram (ex1)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    amp = utils.stft(wav, hop=hop, win_length=win_length)
    db = librosa.amplitude_to_db(np.abs(amp))
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        ax=ax,
        cmap="rainbow",
    )
    ax.set(title="Spectrogram", xlabel=None, ylabel="Frequency [Hz]")
    fig.colorbar(img, aspect=10, pad=0.01, ax=ax, format="%+2.f dB")
    save_fname = os.path.join(path, "result", "spectrogram.png")
    plt.savefig(save_fname, transparent=True)
    plt.show()

    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 6))
    plt.subplots_adjust(hspace=0.6)

    # calculate mel spectrogram and mfcc
    mel_spec, mfcc = calc_mfcc(wav, hop, win_length, filterbank)

    # mel spectrogram
    wav_time = wav.shape[0] // sr
    f_nyq = sr // 2
    extent = [0, wav_time, 0, f_nyq]

    img = ax[0].imshow(
        librosa.amplitude_to_db(mel_spec),
        aspect="auto",
        extent=extent,
        cmap="rainbow",
    )
    ax[0].set(
        title="Mel spectrogram",
        xlabel=None,
        ylabel="Mel frequency [mel]",
        ylim=[0, 8000],
        yticks=range(0, 10000, 2000),
    )
    fig.colorbar(img, aspect=10, pad=0.01, ax=ax[0], format="%+2.f dB")

    # mfcc
    n_mfcc = 12
    extent = [0, wav_time, 0, n_mfcc]
    img = ax[1].imshow(
        np.flipud(mfcc[:n_mfcc]), aspect="auto", extent=extent, cmap="rainbow"
    )
    ax[1].set(
        title="MFCC sequence",
        xlabel=None,
        ylabel="MFCC",
        yticks=range(0, 13, 4),
    )
    fig.colorbar(img, aspect=10, pad=0.01, ax=ax[1], format="%+2.f dB")

    # d-mfcc
    d_mfcc = delta_mfcc(mfcc, k=2)

    img = ax[2].imshow(
        np.flipud(d_mfcc[:n_mfcc]), aspect="auto", extent=extent, cmap="rainbow"
    )
    ax[2].set(
        title="ΔMFCC sequence",
        xlabel=None,
        ylabel="ΔMFCC",
        yticks=range(0, 13, 4),
    )
    fig.colorbar(img, aspect=10, pad=0.01, ax=ax[2], format="%+2.f dB")

    # dd-mfcc
    dd_mfcc = delta_mfcc(d_mfcc, k=2)
    img = ax[3].imshow(
        np.flipud(dd_mfcc[:n_mfcc]), aspect="auto", extent=extent, cmap="rainbow"
    )
    ax[3].set(
        title="ΔΔMFCC sequence",
        xlabel="Time [s]",
        ylabel="ΔΔMFCC",
        yticks=range(0, 13, 4),
    )
    fig.colorbar(img, aspect=10, pad=0.01, ax=ax[3], format="%+2.f dB")

    save_fname = os.path.join(path, "result", "mfcc_result.png")
    plt.savefig(save_fname, transparent=True)
    plt.show()


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(description="MFCC analyze")
    parser.add_argument("fname", type=str, help="Load filename")
    args = parser.parse_args()
    main(args)
