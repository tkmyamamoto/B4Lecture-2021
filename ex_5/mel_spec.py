import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import argparse
import math

def delta(mfcc):
    p_mfcc = np.append(mfcc, np.zeros((1,len(mfcc[0]))), axis=0)
    
    delta = np.zeros(mfcc.shape)
    for i in range(1, len(p_mfcc)-1):
        delta[i] = p_mfcc[i+1] - p_mfcc[i-1]
    return delta

def mfcc(data, sample_rate, nfilt):
    frame_size = 0.025
    frame_stride = 0.01
    NFFT = 512
    num_ceps = 12
    pre_emphasis = 0.97

    emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum



    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    dlt = delta(mfcc)
    dltdlt = delta(dlt)


    plt.subplot(4, 1, 1)
    plt.title('Original Signal')
    plt.specgram(data, Fs = sample_rate)
    plt.colorbar()
    plt.ylabel('Hz')    

    plt.subplot(4, 1, 2)
    librosa.display.specshow(mfcc.T)
    plt.colorbar()
    plt.ylabel('MFCC')

    plt.subplot(4, 1, 3)
    librosa.display.specshow(dlt.T)
    plt.colorbar()
    plt.ylabel('Delta')    

    plt.subplot(4, 1, 4)
    librosa.display.specshow(dltdlt.T)
    plt.colorbar()
    plt.ylabel('Delta delta')  
    plt.xlabel('time')
    plt.savefig("mfcc.png")



def main():
    parser = argparse.ArgumentParser(description='Program for applying digital filter.\nFile name, filter type, filtering frequency are required.')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)

    args = parser.parse_args()
    wave_array, sr = librosa.load(args.filename, sr=22050)
    mfcc(wave_array, sr, 40)


if __name__ == "__main__":
    main()
