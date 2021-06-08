import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt

def stft(signal, n_fft=512):
    """
    Args:
      signal  (ndarray, shape=(length,)):input signal
      n_fft   (int, optional)           :window width

    """
    n_shift = n_fft//2

    # get signal length
    length = signal.shape[0]

    # zero padding
    signal = np.pad(signal, [0, n_fft])

    n_frame = (length - n_fft) // n_shift + 1
    spectrogram = np.zeros((n_fft // 2 + 1, n_frame), dtype=np.complex128)

    i = 0
    for start in range(0, length - n_fft + 1, n_shift):
        # cut signal
        signal_cut = signal[start : start + n_fft]

        # apply window function
        signal_cut = signal_cut * np.hamming(n_fft)

        # fast fourier transform
        spectrogram[:, i] = np.fft.rfft(signal_cut)

        i = i + 1

    return spectrogram


def istft(spectrogram):
    """
    Args:
      spectrogram (ndarray, shape=(n_fft//2+1, n_frame))  :input spectrogram

    """

    # (frequency, frame) -> (frame, frequency)
    spectrogram_t = np.transpose(spectrogram)

    # inverse fast fourier transform
    signal_cut = np.fft.irfft(spectrogram_t)

    n_fft = signal_cut.shape[1]

    # merge overlap part
    first = signal_cut[:, :n_fft//2]
    latter = signal_cut[:, n_fft//2:]
    signal_cut = first
    signal_cut[1:] = signal_cut[1:] + latter[:-1]

    # concatnate each frame
    signal = signal_cut.reshape(-1)

    return signal


if __name__ == "__main__":
    # load
    origin_signal, rate = librosa.load('sample.wav')

    n_fft = 512

    # stft
    spectrogram = stft(origin_signal, n_fft)

    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

    # istft
    resynthesized_signal = istft(spectrogram)

    # plot
    fig, ax = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=1.0)
    
    librosa.display.waveplot(origin_signal, sr=rate, x_axis='time', ax=ax[0])
    ax[0].set(title='Original signal', xlabel='Time[sec]', ylabel='Magnitude')
    
    img = librosa.display.specshow(spectrogram_db, sr=rate, hop_length=n_fft//2, x_axis='time', y_axis='linear', ax=ax[1])
    ax[1].set(title='Spectrogram', xlabel='Time[sec]', ylabel='Frequency[Hz]')
    fig.colorbar(img, ax=ax[1])

    librosa.display.waveplot(resynthesized_signal, sr=rate, x_axis='time', ax=ax[2])
    ax[2].set(title='Re-synthesized signal', xlabel='Time[sec]', ylabel='Magnitude')

    ax_pos_0 = ax[0].get_position()
    ax_pos_1 = ax[1].get_position()
    ax_pos_2 = ax[2].get_position()
    ax[0].set_position([ax_pos_0.x0, ax_pos_0.y0, ax_pos_1.width, ax_pos_1.height])
    ax[2].set_position([ax_pos_2.x0, ax_pos_2.y0, ax_pos_1.width, ax_pos_1.height])

    plt.savefig('result')
