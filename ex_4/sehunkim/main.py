import numpy as np
from matplotlib import pyplot as plt
import matplotlib.collections as collections
from scipy.signal import find_peaks
import scipy.signal as sig
import argparse
import librosa
import librosa.display
import soundfile


def autocorr(wave_array, sr):
    """
    Function for finding f0

    Parameters:
    wave_array : sound data
    sr : sampling rate of input data

    Return:
    f0 : calculated f0
    """
    m = 500
    temp = 0
    r = np.array([])
    for i in range(m):
        temp = 0
        for j in range(len(wave_array) - m - 1):
            temp += wave_array[j] * wave_array[j+i]
        r = np.append(r, temp)

    peaks_r, _ = find_peaks(r)
    max_peak = 0
    n_max_peak = 0
    for i in range(len(peaks_r)):
        if max_peak < r[peaks_r[i]]:
            max_peak = r[peaks_r[i]]
            n_max_peak = peaks_r[i]

    if n_max_peak == 0:
        f0 = 0
    else:
        f0 = sr/n_max_peak

    plt.title("autocorrelation")
    plt.plot(r, label='r(m)')
    plt.plot(peaks_r, r[peaks_r], "x", label='peak')
    label = 'max peak, $f0$ = {:.2f} Hz'.format(f0)
    plt.plot(n_max_peak, max_peak, "x", color='r', label=label)
    plt.xlabel("m")
    plt.ylabel("r")
    plt.legend()
    plt.savefig("autocorr.png")
    plt.clf()
    #input("Press Enter to continue...")

    return f0


def autocorr_func(data):
    data_len = len(data)
    cor = np.zeros(data_len)
    for i in range(data_len):
        cor[i] = data[:data_len - i] @ data[i:]
    return cor


def cepstrum(frame, sr):
    time_vector = np.arange(len(frame)) / sr
    dt = 1./sr
    freq_vector = np.fft.rfftfreq(len(frame), d=dt)

    windowed_signal = np.hamming(len(frame)) * frame

    spec = np.fft.rfft(frame)
    log_spec = 20 * np.log10(np.abs(spec))

    """
    plt.xlabel('frequency (Hz)')
    plt.title('Fourier spectrum')
    plt.savefig("Fourier spectrum.png")
    plt.clf()    
    """
    cepstrum = np.fft.rfft(log_spec)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.rfftfreq(log_spec.size, df)

    peaks_ceps, _ = find_peaks(
        np.abs(cepstrum), threshold=1, distance=10, prominence=1)

    max_peak = 0
    n_max_peak = 0
    for i in range(len(peaks_ceps)):
        if max_peak < np.abs(cepstrum[peaks_ceps[i]]) and (1/quefrency_vector[peaks_ceps[i]]) < 500:
            max_peak = np.abs(cepstrum[peaks_ceps[i]])
            n_max_peak = peaks_ceps[i]
    if max_peak == 0:
        f0 = 1
    else:
        f0 = 1/(n_max_peak *
                (quefrency_vector[n_max_peak]-quefrency_vector[n_max_peak-1]))

    fig, ax = plt.subplots()
    label = 'max peak, $f0$ = {:.2f} Hz'.format(f0)
    plt.plot(quefrency_vector, np.abs(cepstrum))
    plt.plot(peaks_ceps * quefrency_vector[1],
             np.abs(cepstrum[peaks_ceps]), "x", label='peak')
    plt.plot(n_max_peak * quefrency_vector[1],
             max_peak, "x", color='r', label=label)
    valid = (quefrency_vector <= 1/f0)
    collection = collections.BrokenBarHCollection.span_where(
        quefrency_vector, ymin=0, ymax=np.abs(cepstrum).max(), where=valid, facecolor='green', alpha=0.5, label='valid pitches')
    ax.add_collection(collection)
    plt.ylim([0, 2000])
    plt.xlabel('Quefrency (s)')
    plt.title('Cepstrum')
    plt.legend()
    plt.savefig("cepstrum.png")
    plt.clf()

    cepstrum = np.nan_to_num(cepstrum)
    lifter = np.zeros(len(cepstrum))
    lifter[:int(n_max_peak * 1)] = 1
    liftered_cepstrum = lifter * cepstrum
    liftered_spec = np.fft.irfft(liftered_cepstrum)
    if len(spec) > len(liftered_spec):
        while len(spec) != len(liftered_spec):
            liftered_spec = np.append(
                liftered_spec, liftered_spec[len(liftered_spec)-1])
    exp_liftered_spec = np.exp(liftered_spec)

    plt.plot(freq_vector, log_spec, color='y', label='spectrum')
    plt.plot(freq_vector, liftered_spec, color='r', label='liftered_spectrum')
    plt.xlabel('frequency (Hz)')
    plt.title('power spectrum')
    plt.legend()

    return f0


def lev_durb(cor, order, sr):
    a = np.zeros(order + 1)
    e = np.zeros(order + 1)

    a[0] = 1.0
    a[1] = - cor[1] / cor[0]
    e[1] = cor[0] + a[1] * cor[1]
    lam = - cor[1] / cor[0]

    for k in range(1, order):
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * cor[k + 1 - j]
        lam /= e[k]
        u = [1]
        u.extend(a[i] for i in range(1, k + 1))
        u.append(0)
        v = [0]
        v.extend(a[i] for i in range(k, 0, -1))
        v.append(1)
        a = np.array(u) + lam * np.array(v)
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]


def lpc(frame, order, frame_size, sr):
    dt = 1./sr
    cor = autocorr_func(frame)
    cor = cor[:len(cor)//2]
    a, e = lev_durb(cor, order, sr)
    h = sig.freqz(np.sqrt(e), a, frame_size, "whole")[1]
    env_lpc = 20 * np.log10(np.abs(h))
    freq_vector = np.fft.rfftfreq(len(env_lpc), d=dt)

    plt.plot(freq_vector[:len(env_lpc)//2],
             env_lpc[:len(env_lpc)//2], color='g', label='lpc')
    plt.legend()
    plt.savefig("liftered spectrum.png")
    plt.clf()
    return env_lpc


def main():
    parser = argparse.ArgumentParser(
        description='Program for applying digital filter.\nFile name, filter type, filtering frequency are required.')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)
    parser.add_argument("-fs", dest="frame_size", type=int,
                        help='Frame size (optional default = 1024)', required=False, default=2048)
    parser.add_argument("-ov", dest="overlap", type=float,
                        help='Frame overlap ratio (optional default = 0.5)', required=False, default=0.5)
    parser.add_argument("-o", dest="lpc_order", type=int,
                        help='order for lpc', required=False, default=64)
    args = parser.parse_args()
    frame_size = args.frame_size
    overlap = args.overlap
    lpc_order = args.lpc_order
    wave_array, sr = librosa.load(args.filename, sr=22050)
    duration = len(wave_array)/sr
    X = librosa.stft(wave_array)
    Xdb = librosa.amplitude_to_db(abs(X))

    f0_autocorr = np.array([])
    f0_ceps = np.array([])
    n_of_frame = len(wave_array)//int(frame_size * (1-overlap))
    
    for frame_no in range(n_of_frame):
        frame = wave_array[int(frame_no * frame_size * (1-overlap)): int((frame_no+1) * frame_size * (1 - overlap))]
        f0_autocorr = np.append(f0_autocorr, autocorr(frame, sr))
        f0_ceps = np.append(f0_ceps, cepstrum(frame, sr))
        env_lpc = lpc(frame, lpc_order, frame_size, sr)
        frame = []
        input("Press Enter to continue...")

    x = np.linspace(0, duration, n_of_frame)
    librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
    plt.plot(x, f0_autocorr, color='r', label='f0_autocorrelation')
    plt.plot(x, f0_ceps, color='b', label='f0_cepstrum')
    plt.ylim([0, 2000])
    plt.legend()
    plt.savefig("spectrogram.png")


if __name__ == "__main__":
    main()
