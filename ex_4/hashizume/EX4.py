import numpy as np
import sys
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal


def AutoCorrelation(y, lim):
    """
    Culculate AutoCorrelation function of y

    Parameters
    ----------
    y : input
    lim : limit

    returns
    -------
    AutoCorrelation function of y
    """
    AC = []
    for i in range(lim):
        if i == 0:
            AC.append(np.sum(y * y))
        else:
            AC.append(np.sum(y[0:-i] * y[i:]))
    return np.array(AC)


def detect_peak(AC):
    """
    recode peaks of AC in an array "peak"

    Parameters
    ----------
    AC : AutoCorrelation function

    returns
    -------
    peaks of AC

    """
    peak = []
    # exclude the beginning(first peak) and the end
    for i in range(AC.shape[0] - 2):
        if AC[i] < AC[i + 1] and AC[i + 1] > AC[i + 2]:
            peak.append([i + 1, AC[i + 1]])
    return np.array(peak)


def calc_AC(x, win_length, hop_length, sr):
    """
    estimate f0 series
    Calculate the autocorrelation for each frame
    and extract the second peak

    Parameters
    ----------
    x : input signal
    win_length : length of window
    hop_length : hop size
    sr : sampling rate

    returns
    -------
    f0 series of AC (sr / second peaks)

    note
    ----
    The first peak is excluded,
    so "the largest peak = the second peak"

    """
    f0series = []

    # number of samples
    N = x.shape[0]

    # number of times to apply the window
    T = int((N - hop_length) / hop_length)

    for t in range(T):
        # data cut out by the window
        x_flame = x[t * hop_length : t * hop_length + win_length]

        # calculate Autocorrelation
        AC = AutoCorrelation(x_flame, x_flame.shape[0])

        # narrow down peak candidates
        peak = detect_peak(AC)
        peak_index = peak[:, 0]
        peak_value = peak[:, 1]

        # sort in descending order of value
        peak_value_sorted = np.sort(peak_value)[::-1]

        second_peak_value = peak_value_sorted[0]

        # index of "second peak"
        second_peak_index = np.where(AC == second_peak_value)[0]

        # calculate the f0 series
        f0series.append(sr / second_peak_index[0])
    return np.array(f0series)


def db(x, dBref):
    """
    linear → db conversion

    Parameters
    ----------
    x : input
    dBref : decibel reference
    """

    y = 20 * np.log10(x / dBref)
    return y


def lin(x, dBref):
    """
    db → linear conversion

    Parameters
    ----------
    x : input
    dBref : decibel reference
    """

    y = dBref * 10 ** (x / 20)
    return y


def norm(spectrum, dBref):
    """
    normalization

    Parameters
    ----------
    x : input
    dBref : decibel reference
    """

    spectrum = lin(spectrum, dBref)
    spectrum = np.abs(spectrum / (len(spectrum) / 2))
    spectrum = db(spectrum, dBref)
    return spectrum


def levinson(r):
    """
    levinson durbin algorithm
    """
    a = np.zeros(len(r))
    a[0] = 1
    sigma = r[0]
    for p in range(1, a.shape[0]):
        w = np.sum(a[:p] * r[p:0:-1], axis=0)
        k = w / sigma
        sigma = sigma - k * w

        a[1 : p + 1] = a[1 : p + 1] - k * a[p - 1 :: -1]
    e = np.sqrt(sigma)
    return a, e


def lpc(x, win_length, deg):
    ac = AutoCorrelation(x, x.shape[0])
    r = ac[:deg]
    a, e = levinson(r)
    env = e / np.fft.rfft(a, win_length, axis=0)
    env = librosa.amplitude_to_db(np.abs(env))

    return env


def main():
    args = sys.argv
    wav_filename = args[1]
    origin_signal, sr = librosa.load(wav_filename)

    # F0 estimate
    # Autocorrelation
    win_length = 1024
    hop_length = 512

    f0series = calc_AC(origin_signal, win_length, hop_length, sr)

    f0series = np.concatenate([[f0series[0]], f0series, [f0series[-1]]])
    s = np.array(origin_signal).astype(np.float32)

    spec = librosa.stft(s, win_length, hop_length)
    spec_db = librosa.amplitude_to_db(np.abs(spec))
    librosa.display.specshow(spec_db, y_axis="log")

    plt.plot(np.arange(spec_db.shape[1]), f0series, color="b", linewidth=2.0)
    plt.xlabel("time[s]")
    plt.ylabel("freqency[Hz]")
    plt.title("AC")
    plt.savefig("AC")
    plt.clf()
    plt.close

    # cepstrum

    # fft
    spec = np.fft.fft(origin_signal)
    # linear → db
    spec_db = db(spec, 2e-5)
    # ifft
    ceps_db = np.real(np.fft.ifft(spec_db))
    # LPF
    index = 50
    ceps_db[index : len(ceps_db) - index] = 0
    # fft ceps_db_low : spectrum envelope
    ceps_db_low = np.fft.fft(ceps_db)

    # calculate the amplitude component of the audio spectrum
    spec_db_amp = norm(spec_db, 2e-5)
    # calculate the amplitude component of the spectrum envelope
    ceps_db_low_amp = norm(ceps_db_low, 2e-5)

    # LPC
    env_LPC = lpc(origin_signal, win_length, 32)

    # plot

    # axis settting
    frequency = np.linspace(0, sr, len(origin_signal))
    quefrency = np.arange(0, len(origin_signal)) / sr

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("SPL [dB]")

    plt.title("envelope")

    plt.plot(frequency, spec_db_amp)
    plt.plot(frequency, ceps_db_low_amp, lw="4")
    plt.plot(env_LPC)
    plt.plot()

    plt.xticks(np.arange(0, 20000, 1000))
    plt.xlim(0, 5000)
    plt.yticks(np.arange(-200, 200, 20))
    plt.ylim(-60, 60)

    plt.savefig("env")
    plt.close()


if __name__ == "__main__":
    main()
