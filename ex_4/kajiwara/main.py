import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from modules.autocorrelation import autocorrelation, get_ac_peaks
from modules.cepstrum import get_cepstrum, get_ceps_peaks, get_envelope
from modules.spectrogram import spectrogram
from modules.lpc import lpc_method

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def main(args):
    audio_path = Path(args.audio_path)

    result_path = Path(args.save_path)
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    result_path = result_path/timestamp
    if not result_path.exists():
        try:
            result_path.mkdir(parents=True)
        except Exception as err:
            print(err)

    wave_data, sr = librosa.load(audio_path)
    win_size = 1024
    overlap = 0.5

    db_spec = spectrogram(
        wave_data, win_size=win_size, overlap=overlap, mode='faster', scale='db')

    ''' extract f0 by autocorrelation method'''
    ac = autocorrelation(
        wave_data, is_framing=True, win_size=win_size, overlap=overlap)
    peaks = get_ac_peaks(ac)
    f0 = sr / peaks
    times = np.linspace(0, (wave_data.size - win_size) / sr, f0.size)
    librosa.display.specshow(
        db_spec, sr=sr, x_axis='time', y_axis='linear', cmap='rainbow')
    plt.plot(times, f0, color="green")
    plt.ylim(0, 1000)
    plt.title("F0 AC")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.savefig(result_path/'f0_ac.png')
    plt.clf()
    plt.close()

    '''extract f0 by cepstrum method'''
    ceps_db = get_cepstrum(wave_data, is_clipping=False, is_framing=True)
    peak_index = get_ceps_peaks(ceps_db, sr)
    f0 = sr / peak_index
    times = np.linspace(0, (wave_data.size - win_size) / sr, f0.size)
    librosa.display.specshow(
        db_spec, sr=sr, x_axis='time', y_axis='linear', cmap='rainbow')
    plt.plot(times, f0, color="green")
    plt.ylim(0, 1000)
    plt.title("F0 Cepstrum")
    plt.xlabel("Time[sec]")
    plt.ylabel("Frequency[Hz]")
    plt.savefig(result_path/'f0_ceps.png')
    plt.clf()
    plt.close()

    '''extract envelope'''
    clip_size = len(wave_data)
    dim = 100

    fscale = np.fft.fftfreq(clip_size, d=1.0 / sr)

    freq = np.fft.rfft(wave_data, clip_size)
    amp = 20 * np.log10(np.abs(freq))

    ceps_db = get_cepstrum(wave_data, is_clipping=True, is_framing=False, clip_size=clip_size)
    ceps_env = get_envelope(ceps_db, dim)

    lpc_env = lpc_method(wave_data, dim=dim, clip_size=clip_size)

    sp_lpc_env = lpc_method(wave_data, mode='scipy', dim=dim, clip_size=clip_size)

    plt.plot(fscale[:clip_size//2], amp[:clip_size//2], label='spectrum')
    plt.plot(fscale[:clip_size//2], ceps_env[:clip_size//2], label='cepstrum')
    plt.plot(fscale[:clip_size//2], lpc_env[:clip_size//2], label='lpc')
    plt.plot(fscale[:clip_size//2], sp_lpc_env[:clip_size//2], label='sp_lpc')
    plt.title("amplitude characteristic")
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(result_path/'enveloop.png')
    plt.clf()
    plt.close()


if __name__ == "__main__":
    description = 'Example: python main.py ./audio/sample.wav -s ./result'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('audio_path', help='path to audio data')
    parser.add_argument('-s', '--save-path', default='./result', help='path to save the result')

    args = parser.parse_args()

    main(args)
