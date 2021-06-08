# include flake8, black

import argparse
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from stft import stft


def data_to_frame(data, win_length, hop_length):
    """
    Divide data into short frames.

    Parameters:
        data : 1D numpy.ndarray
            Wave data.
        win_length : int
            Window size.
        hop_length : int
            Shift size.

    Returns:
        frames : 2D numpy.ndarray, shape=(frame, time)
            Cut wav data.
    """
    data_size = data.shape[0]
    frames = []
    for i in range(0, data_size, hop_length):
        if win_length > data_size - i:
            break
        tmp = data[i : i + win_length]
        frames.append(tmp)

    return frames


def findpeaks(data, distance):
    """
    Find index of maximum peak of data.

    Parameters:
        data : numpy.ndarray
            data array
        distance : int
            Required minimal horizontal distance (>= 1) in samples
            between neighbouring peaks.

    Returns:
        peaks : numpy.ndarray
            Index of maximum peak of data.
    """
    frames = data_to_frame(data, 2 * distance + 1, 1)
    peaks_list = np.copy(data[distance:-distance])

    # remove point except for peak
    peaks_list[np.argmax(frames, axis=1) != distance] = -np.inf

    peaks = np.argmax(peaks_list, axis=0) + distance

    return peaks


def autocorr(data, win_length, hop_length):
    """
    Definition of autocorrelation function.

    Parameters:
        data : 1D numpy.ndarray
            Wave data.
        win_length : int
            Window size.
        hop_length : int
            Shift size.

    Returns:
        ac : numpy.ndarray
            Auto correlation results. (ac[i] = r_i)
    """
    ac = []
    frames = data_to_frame(data, win_length, hop_length)

    for frame in frames:
        spec = np.fft.fft(frame)
        power = spec * spec.conjugate()
        ac.append(np.fft.irfft(power, axis=0).real)
    ac = np.array(ac, dtype=float)

    return ac.transpose()[:win_length]


def f0_autocorr(ac, sr):
    """
    Find the fundamental frequency F0 using the autocorrelation function.

    Parameters:
        ac : numpy.ndarray
            Autocorrelation function.
        sr : int
            Sampling frequency.

    Returns:
        f0 : numpy.ndarray
            Fundamental frequency F0.
    """
    peaks = findpeaks(ac, 20)
    f0 = sr / peaks

    # remove low power point
    f0[ac[0] < 0.4] = 0

    return f0


def ceps(db, lift_th, sr):
    """
    Calculate spectral envelope with cepstrum and find the fundamental
    frequency F0 from microstructure.

    Parameters:
        db : numpy.ndarray
            Log scale spectrogram.
        lift_th : int
            Threshold to separate envelope and microstructure.
        sr : int
            Sampling frequency.

    Returns:
        f0 : numpy.ndarray
            Fundamental frequency F0.
        env : numpy.ndarray
            Spectral envelope by cepstrum.
    """
    ceps = np.fft.ifft(db, axis=0)

    # calculate f0
    peaks = findpeaks(ceps[lift_th:], 50) + lift_th
    f0 = sr / peaks

    # liftering
    env = ceps.copy()
    env[lift_th:-lift_th] = 0
    env = np.fft.fft(env, axis=0).real

    return f0, env


def levinson_durbin(r):
    """
    Algorithm of Levinson-Durbin.

    Parameters:
        r : numpy.ndarray
            Autocorrelation function.

    Returns:
        a : numpy.ndarray
            Coefficient of LPC.
        e : numpy.ndarray
            Resudual variance.
    """
    a = np.zeros_like(r)
    a[0] = 1.0
    sig_sq = r[0]
    for p in range(1, a.shape[0]):
        w = np.sum(a[:p] * r[p:0:-1], axis=0)
        k = w / sig_sq
        sig_sq = sig_sq - k * w
        a[1 : p + 1] = a[1 : p + 1] - k * a[p - 1 :: -1]
    e = np.sqrt(sig_sq)

    return a, e


def lpc(ac, order, win_length):
    """
    Calculate spectral envelope with LPC.

    Parameters:
        ac : numpy.ndarray
            Autocorrelation function.
        order : int
            Dimention of coefficient of LPC.
        win_length : int
            Window size.

    Returns:
        env : numpy.ndarray
            Spectral envelope by LPC.
    """
    r = ac[:order]
    a, e = levinson_durbin(r)

    # Calculate frequency response to residuals
    env = e / np.fft.rfft(a, win_length, axis=0)
    env = librosa.amplitude_to_db(np.abs(env))

    return env


def main(args):
    fname = args.fname
    start = args.start
    """
    fname = "aiueo.wav"
    start = 4.3
    """

    # load audio file
    # get current working directory
    path = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(path, "wav", fname)
    wav, sr = librosa.load(fname, mono=True)

    # parameter
    hop = 0.5
    win_length = 1024
    hop_length = int(win_length * hop)
    lift_th = 100
    lpc_order = 64

    # STFT
    amp = stft(wav, hop=hop, win_length=win_length)
    # convert an amplitude spectrogram to dB-scaled spectrogram
    db = librosa.amplitude_to_db(np.abs(amp))

    # calculate f0
    ac = autocorr(wav, win_length, hop_length)
    f0_ac = f0_autocorr(ac, sr)
    f0_ceps, env_ceps = ceps(db, lift_th, sr)

    # plot f0
    fig, ax = plt.subplots()
    # draw spectrogram (log scale)
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        ax=ax,
        cmap="rainbow",
    )
    ax.set(title="Fundamental frequency f0", xlabel="Time [s]", ylabel="Frequency [Hz]")
    fig.colorbar(img, aspect=40, pad=0.01, ax=ax, format="%+2.f dB")
    t = np.linspace(0, (wav.size - win_length) / sr, f0_ac.size)
    plt.plot(t, f0_ac, color="black", label="autocorr")
    plt.plot(t, f0_ceps, color="magenta", label="ceps")
    plt.ylim(0, 2000)
    plt.legend()
    save_fname = os.path.join(path, "result", "f0.png")
    plt.savefig(save_fname)
    # plt.show()

    # calculate frame index from "start".
    frame = int((start * sr) // hop_length)

    # calculate envelope.
    env_lpc = lpc(ac, lpc_order, win_length)[:, frame]
    env_lpc = env_lpc[: len(env_lpc) // 2]

    # plot envelope.
    fig, ax = plt.subplots()
    ax.set(
        title="Envelope (time = {:.1f}s)".format(start),
        xlabel="Frequency [Hz]",
        ylabel="Amplitude [dB]",
    )
    freq = np.linspace(0, 8000, win_length // 2 + 1)
    plt.plot(freq, db[:, frame], label="original")
    plt.plot(freq, env_ceps[:, frame], label="cepstrum")
    freq = np.linspace(0, 8000, win_length // 4)
    plt.plot(freq, env_lpc, label="lpc(deg={})".format(lpc_order))
    plt.legend()
    save_fname = os.path.join(path, "result", "envelope.png")
    plt.savefig(save_fname)
    plt.show()


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(
        description="Fundamental frequency F0 and spectral envelope."
    )
    parser.add_argument("fname", type=str, help="Load filename")
    parser.add_argument(
        "-t",
        "--start",
        type=float,
        help="Time to show envelope",
        required=True,
    )
    args = parser.parse_args()
    main(args)
