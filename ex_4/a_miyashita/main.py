import argparse
import time

import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve_toeplitz
import soundfile

import ex_1.a_miyashita.main as ex1


def frame(input, win_size, shift_size):
    """
    Cut signal to short frames.

    # Args
        input (ndarray, axis=(time,...)) :input signal
        win_size (int)                      :window size
        shift_size (int)                    :shift size

    # Returns
        frames (ndarray, axis=(frame, time,...))

    # Notes
        If input is more than 1-D, results are caluclated elementwise
        along 1st dementsion.
        Then input.shape[1:] and frames.shape[2:] are same.
    """
    length = input.shape[0]
    n_frame = (length - win_size) // shift_size + 1
    frames = np.zeros((n_frame, win_size) + input.shape[1:])
    i = 0
    for start in range(0, length - win_size + 1, shift_size):
        frames[i] = input[start : start + win_size]
        i = i + 1
    return frames


def peak(input, neighbor):
    """
    Calculate index of maximum peak of input.

    # Args
        input (ndarray) :input array
        neighbor (int)  :
            this function regards input[i] as peak
            if input[i] >= input[j] for any j s.t. |i - j| <= neighbor.
            (input[:neighbor] or input[-neighbor:] is never selected.)

    # Returns
        peaks (ndarray) :index of maximum peak of input

    # Notes
        If input is more than 1-D, results are caluclated elementwise
        along 1st dementsion.
        Then input.shape[1:] and peaks.shape are same.
    """
    frames = frame(input, 2 * neighbor + 1, 1)
    candidate = np.copy(input[neighbor:-neighbor])
    # remove point except for peak
    candidate[np.argmax(frames, axis=1) != neighbor] = -np.inf
    # calculate maximum of peak
    peaks = np.argmax(candidate, axis=0) + neighbor
    return peaks


def auto_corr(signal, win_size):
    """
    Calculate auto-correlation.

    # Args
        signal (ndarray)    :input signal
        win_size (int)      :size of each frame

    # Returns
        ac (ndarray, axis=(gap, frame))

        ac[i] = r_i
    """
    frames = frame(signal, win_size, win_size // 2)

    # convolution (product in frequency domain)
    spec = np.fft.rfft(frames, win_size * 2)
    power = spec * spec.conj()
    ac = np.fft.irfft(power)

    ac = np.transpose(ac)

    # remove symmetry point
    ac = ac[:win_size]

    return ac


def ac_to_f0(ac, sr):
    """
    Culculate f0 from auto-correlation

    # Args
        ac (ndarray, axis=(gap, frame)) :auto-correlation
        sr (int)     :sampling rate

    # Returns
        f0 (ndarray, axis=(frame,)) :fundamental frequency
    """
    peaks = peak(ac, 10)
    f0 = sr / peaks

    # remove low power point
    f0[ac[0] < 0.2] = 0

    return f0


def ceps(spec_db, threshold, sr):
    """
    Sepalate spectrum to envelope and microstructure by cepstrum method.
    Caluculate f0 from microstructure.

    # Args
        spec_db (ndarray, axis=(freq, frame))  :input log spectrogram
        threshold (int) :boundary between envelope and microstructure
        sr (int)        :sampling rate

    # Returns
        f0 (ndarray, axis=(frame,))         :fundamental frequency
        env (ndarray, axis=(freq, frame))   :envelope log spectrogram
        micro (ndarray, axis=(freq, frame)) :microstructure log spectrogram
    """
    fsize = spec_db.shape[0]
    # restore negative frequency
    spec_db = np.concatenate([spec_db[-2:0:-1], spec_db], axis=0)

    # calculate cepstrum
    ceps = np.fft.rfft(spec_db, axis=0)
    ceps = ceps.real

    # liftering
    env = np.zeros_like(ceps)
    env[:threshold] = ceps[:threshold]
    micro = np.zeros_like(ceps)
    micro[threshold:] = ceps[threshold:]

    # calculate f0
    peaks = peak(ceps[threshold:], 10) + threshold
    f0 = sr / peaks

    # remove low power point
    f0[ceps[0] < -40000] = 0

    # return to frequency domain
    env = np.fft.irfft(env, axis=0)[-fsize:]
    micro = np.fft.irfft(micro, axis=0)[-fsize:]

    return f0, env, micro


def levinson_durbin(r):
    """
    Solve Yule-Walker equation.

    # Args
        r (ndarray) :auto-correlation

    # Returns
        a (ndarray) :linear predictive coefficient
        e (ndarray) :standard deviation of prediction error

    # Notes
        If r is more than 1-D, results are caluclated elementwise
        along 1st dementsion.
    """
    a = np.zeros_like(r)
    a[0] = 1.0
    sigma = r[0]
    for p in range(1, a.shape[0]):
        w = np.sum(a[:p] * r[p:0:-1], axis=0)
        k = w / sigma
        sigma = sigma - k * w

        # a_i = a_i - k_p * a_{p-i+1}
        a[1 : p + 1] = a[1 : p + 1] - k * a[p - 1 :: -1]

    e = np.sqrt(sigma)
    return a, e


def lpc(signal, win_size, deg, method):
    """
    Calculate envelope by linear prediction

    # Args
        signal (ndarray)    :input signal
        win_size (int)      :size of each frame
        deg (int)           :degree of prediction
        method (str)        :method for solve Yule-Walker equation

    # Returns
        env (ndarray, axis=(freq, frame)) :envelope log spectrogram
    """
    ac = auto_corr(signal, win_size)
    r = ac[:deg]

    start = time.time()

    # solve Yule-Walker equation
    if method == 'ld_ewise':
        a, e = levinson_durbin(r)

    elif method == 'ld_for':
        a = np.zeros_like(r)
        a[0] = 1.0
        e = np.zeros(a.shape[1])
        for i in range(a.shape[1]):
            a[:, i], e[i] = levinson_durbin(r[:, i])

    elif method == 'scipy':
        a = np.zeros_like(r)
        a[0] = 1.0
        for i in range(r.shape[1]):
            a[1:, i] = solve_toeplitz(r[:-1, i], -r[1:, i])

        e = np.sqrt(np.sum(a * r, axis=0))

    interval = time.time() - start
    print("time: ", interval)

    # calculate frequency response
    env = e / np.fft.rfft(a, win_size, axis=0)

    env = librosa.amplitude_to_db(np.abs(env))

    return env


def main():
    # process args
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str, help="input filename with extension .wav")
    parser.add_argument(
        "--f0", 
        choices=['ac', 'ceps'], 
        default='ac', 
        help="method for calculate f0",
    )
    parser.add_argument(
        "--lpc",
        choices=['ld_ewise', 'ld_for', 'scipy'],
        default='ld_ewise',
        help="method for solve Yule-Walker equation",
    )
    args = parser.parse_args()
    
    signal, sr = soundfile.read(args.fname)
    win_size = 1024
    spec = ex1.stft(signal, win_size)
    spec_db = librosa.amplitude_to_db(np.abs(spec))

    # calculate f0
    if args.f0 == 'ac':
        ac = auto_corr(signal, win_size)
        f0 = ac_to_f0(ac, sr)
    elif args.f0 == 'ceps':
        f0 = ceps(spec_db, 68, sr)[0]

    # plot f0
    librosa.display.specshow(
        spec_db, sr=sr, hop_length=win_size // 2, x_axis='time', y_axis='linear', cmap='rainbow',
    )
    t = np.linspace(0, (signal.size - win_size) / sr, f0.size)
    plt.ylim(0, 1000)
    plt.plot(t, f0, color='black')
    plt.title("F0 ({})".format(args.f0))
    plt.xlabel("Time[sec]")
    plt.ylabel("Frequency[Hz]")
    plt.colorbar()
    plt.show()

    # calculate envelope
    env1 = lpc(signal, win_size, 32, args.lpc)
    env2 = ceps(spec_db, 68, sr)[1]

    # plot envelope
    w = np.linspace(0, sr / 2, win_size // 2 + 1)
    plt.plot(w, spec_db[:, 135], label="original")
    plt.plot(w, env1[:, 135], label="lpc")
    plt.plot(w, env2[:, 135], label="cepstrum")
    plt.title("Envelope")
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
