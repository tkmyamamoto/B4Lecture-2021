import os
import scipy
import librosa
import argparse
import numpy as np
import spectrogram as sp
import matplotlib.pyplot as plt


def calc_m0(f0):
    """
    input  f0 : frequency constants used for mel conversion
    output m0 : mel frequency constants used for mel conversion
    """

    m0 = 1000 / np.log(1000 / f0 + 1)
    return m0


def to_mel(f, f0=700):
    """
    input  f  : frequency
           f0 : frequency constants used for mel conversion
    output m  : mel frequency
    """

    m0 = calc_m0(f0)
    m = m0 * np.log(f / f0 + 1)
    return m


def to_hertz(m, f0=700):
    """
    input  m  : mel frequency
           f0 : frequency constants used for mel conversion
    output f  : frequency
    """

    m0 = calc_m0(f0)
    f = f0 * (np.exp(m / m0) - 1)
    return f


def melfilter_bank(sr, N, ch_num=20):
    """
    input
    sr     : sampling rate
    N      : frame size
    ch_num : number of melfilter bank channels
    output
    filter_bank : melfilter bank
    """

    # calc some index
    f_max = sr / 2
    m_max = to_mel(f_max)
    n_max = N // 2
    df = sr / N
    dm = m_max / (ch_num + 1)
    m_center = np.arange(1, ch_num + 1) * dm
    f_center = to_hertz(m_center)
    idx_center = np.array(np.round(f_center / df), dtype="int64")
    idx_start = np.hstack(([0], idx_center[0:ch_num-1]))
    idx_stop = np.hstack((idx_center[1:ch_num], [n_max]))

    # calc melfilter bank
    filter_bank = np.zeros((ch_num, n_max))
    for i in range(0, ch_num):
        increment = 1.0 / (idx_center[i] - idx_start[i])
        for j in range(idx_start[i], idx_center[i]):
            filter_bank[i, j] = (j - idx_start[i]) * increment
        decrement = 1.0 / (idx_stop[i] - idx_center[i])
        for j in range(idx_center[i], idx_stop[i]):
            filter_bank[i, j] = 1.0 - (j - idx_center[i]) * decrement

    return filter_bank


def mfcc(data, N, overlap, filter_bank, dim=12):
    """
    input
    data    : audio data
    N       : frame size
    overlap : overlap size
    dim     : mfcc dimension
    output
    mfcc    : mfcc result
    """

    # calc stft
    spec = sp.fft(data, N, overlap)
    spec = spec[:, :N//2]

    # multiply melfilter bank and discrete cosine transform
    m_spec = 20 * np.log10(np.abs(spec) @ filter_bank.T)
    mfcc = np.zeros_like(m_spec)
    mfcc = scipy.fftpack.realtransforms.dct(m_spec, type=2, norm="ortho", axis=-1)
    
    return mfcc[:, :dim]


def delta_mfcc(mfcc):
    """
    input  mfcc  : mfcc data
    output delta : dynamic fluctuation component
    """

    l = 2
    mfcc_pad = np.pad(mfcc, ((l, l), (0, 0)), "edge")
    k = np.arange(-l, l+1)
    den = np.sum(k ** 2)
    num = np.zeros_like(mfcc)
    for i in range(l, mfcc.shape[0]):
        num[i] = k @ mfcc_pad[i-l:i+l+1]
    delta = num / den

    return delta


def main():
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input file path")
    parser.add_argument("--frame", type=int, default=1024, help="data frame size")
    parser.add_argument("--overlap", type=float, default=0.5, help="stft overlap rate")
    parser.add_argument("--f0", type=int, default=700, help="const of culc m0")
    args = parser.parse_args()

    # check output dir exist
    if not os.path.exists('./out'):
        os.makedirs('./out')

    # load audio data
    data, sr = librosa.load(args.input)
    time = np.arange(0, len(data)) / sr

    # get filter bank and calc mfcc, delta mfcc, double delta mfccc
    filter_bank = melfilter_bank(sr, args.frame)
    mfcc_data = mfcc(data, args.frame, int(args.frame*args.overlap), filter_bank)
    d_mfcc = delta_mfcc(mfcc_data)
    dd_mfcc = delta_mfcc(d_mfcc)
    # calc spectrogram
    spec = sp.spec(data, args.frame, int(args.frame*args.overlap))

    # draw all data and save figure
    fig, ax = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    im0 = ax[0].imshow(spec, extent=[0, time[-1], 0, (sr/2)], aspect="auto", cmap="rainbow")
    cbar0 = fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(mfcc_data.T[::-1], extent=[0, time[-1], 0, 12], aspect="auto", cmap="rainbow")
    cbar1 = fig.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow(d_mfcc.T[::-1], extent=[0, time[-1], 0, 12], aspect="auto", cmap="rainbow")
    cbar2 = fig.colorbar(im2, ax=ax[2])
    im3 = ax[3].imshow(dd_mfcc.T[::-1], extent=[0, time[-1], 0, 12], aspect="auto", cmap="rainbow")
    cbar3 = fig.colorbar(im3, ax=ax[3])

    ax[0].set_title('Spectrogram')
    ax[0].set_ylabel('Frequency [Hz]')
    ax[1].set_title('MFCC sequence')
    ax[1].set_ylabel('MFCC')
    ax[2].set_title(r'$\Delta$MFCC sequence')
    ax[2].set_ylabel(r'$\Delta$MFCC')
    ax[3].set_title(r'$\Delta\Delta$MFCC sequence')
    ax[3].set_xlabel('time [s]')
    ax[3].set_ylabel(r'$\Delta\Delta$MFCC')

    plt.show(block=True)
    fig.savefig('./out/mfcc.png')


if __name__ == '__main__':
    main()
