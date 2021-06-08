import os
import argparse
import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def stft(y, frame_size, overlap):
    F = []
    ynum = y.shape[0]
    window_func = signal.hamming(frame_size)

    for i in range(0, ynum, frame_size-overlap):
        frame = y[i:i+frame_size]
        if len(frame) == frame_size:
            frame = frame * window_func
            F.append(np.fft.rfft(frame)[::-1])
   
    F = np.transpose(F)
    return F


def istft(F, frame_size, overlap):
    y = []
    F = np.transpose(F)
    window_func = signal.hamming(frame_size)

    for i in range(len(F)):
        f = np.fft.irfft(F[i][::-1]) / window_func
        if i == 0:
            y.extend(f)
        else:
            y.extend(f[overlap:frame_size])

    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    y, sr = librosa.load(args.input)
    time = np.arange(0, len(y)) / sr

    frame_size = 1024
    overlap = 512

    stft_data = stft(y, frame_size, overlap)
    istft_data = istft(stft_data, frame_size, overlap)
    for i in range(len(time) - len(istft_data)):
        istft_data.append(0)
    spec = np.log(np.abs(stft_data) ** 2)

    
    fig, axes = plt.subplots(3, 1, sharex=True)
    
    axes[0].plot(time, y)
    axes[2].plot(time, istft_data)
    im = axes[1].imshow(spec, extent=[0, time[-1], 0, (sr/2) / 1000], aspect="auto")
    cbar = fig.colorbar(im, ax = axes[1])

    fig.tight_layout()
    plt.subplots_adjust(left=0.1, right=1.0, bottom=0.1, top=0.9)
    fig.set_figheight(5.2)
    fig.set_figwidth(5.5)
    fig.canvas.draw()

    ax0pos = axes[0].get_position()
    ax1pos = axes[1].get_position()
    ax2pos = axes[2].get_position()
    axes[0].set_position([ax0pos.x0, ax0pos.y0, ax1pos.width, ax0pos.height])
    axes[1].set_position([ax1pos.x0, ax1pos.y0, ax1pos.width, ax1pos.height])
    axes[2].set_position([ax2pos.x0, ax2pos.y0, ax1pos.width, ax2pos.height])

    axes[0].set_title("Original signal")
    axes[1].set_title("Spectrogram")
    axes[2].set_title("Re-synthesized signal")

    axes[0].set_ylabel("Magnitude")
    axes[1].set_ylabel("Frequency[kHz]")
    axes[2].set_ylabel("Magnitude")
    axes[2].set_xlabel("Time[s]")

    plt.show(block=True)
    fig.savefig(args.out)


if __name__ == "__main__":
    main()
