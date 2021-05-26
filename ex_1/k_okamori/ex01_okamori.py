import numpy as np
import soundfile as sf  # 音声データの読み取り
import matplotlib.pyplot as plt  # 出力データのプロット

# 定数の定義
NFFT = 1024  # frame の大きさ
OVERLAP = int(NFFT / 2)  # half shift を採用
window = np.hamming(NFFT)  # 窓関数はハミング窓を使用

# 音声データのロード
#   fname: ファイル名


def wav_load(fname):
    wave, samplerate = sf.read(fname)
    return wave, samplerate

# 時間信号グラフの描画
#   ax: グラフ領域
#   title: グラフのタイトル
#   time:   x座標のデータ
#   wave:   y座標のデータ


def draw_signal(ax, title, time, wave):
    ax.plot(time, wave)  # 時間信号の描画
    ax.set_title(title)  # タイトルの追加
    ax.set_xlabel("time[s]")   # x軸ラベルの追加
    ax.set_ylabel("Magnitude")  # y軸ラベルの追加

# スペクトログラムグラフの描画
#   ax:グラフ領域
#   title: グラフのタイトル
#   spec: スペクトログラムのデータ
#   wave_length: 音源の長さ
#   samplerate: サンプルレート


def draw_spectrogram(fig, ax, title, spec, wave_length, samplerate):
    spec = np.log(np.square(np.abs(spec)))
    im = ax.imshow(spec.T, extent=[0, wave_length, 0, samplerate / 2],
                   cmap="rainbow", aspect="auto", origin="lower")  # スペクトログラムの描画
    fig.colorbar(im, ax=ax)
    ax.set_title(title)             # タイトルの追加
    ax.set_xlabel("time[s]")        # x軸ラベルの追加
    ax.set_ylabel("Frequency[kHz]")  # y軸ラベルの追加


# 短時間フーリエ変換(STFT)
#   wave: 入力波形
def stft(wave):
    spec = np.zeros([len(wave) // OVERLAP - 1, int(NFFT / 2) + 1],
                    dtype=np.complex)  # spec:出力データ
    for idx in range(0, len(wave), OVERLAP):
        frame = wave[idx: idx + NFFT]  # frame の切り出し
        if len(frame) == NFFT:  # frame が全部切り出せるときのみ変換
            windowed = window * frame  # 窓関数をかける
            result = np.fft.rfft(windowed)  # フーリエ変換
            for i in range(spec.shape[1]):
                spec[int(idx / OVERLAP)][i] = result[i]  # 計算結果を出力データに追加
    return spec

# 短時間フーリエ逆変換(ISTFT)
#   spec: スペクトログラム
#   wave_size: wave の要素数


def i_stft(spec, wave_size):
    wave = np.zeros(wave_size)  # wave:出力データ
    for idx in range(wave_size // OVERLAP - 2):
        frame = np.fft.irfft(spec[idx])  # frame の切り出し
        wave[idx * OVERLAP: idx * OVERLAP + NFFT] += np.real(frame)  # フーリエ逆変換
    return wave


# main
def main():
    fig = plt.figure()  # 以下，グラフ領域の確保
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    wave, samplerate = wav_load("sound.wav")  # 音声データのロード
    wave_size = len(wave)  # wave の要素数
    wave_length = wave_size / samplerate  # wave の長さ
    time = np.arange(0, wave_size) / samplerate  # 横軸を time に設定
    draw_signal(ax1, "Original signal", time, wave)  # 元データの描画
    spec = stft(wave)  # 短時間フーリエ変換
    draw_spectrogram(fig, ax2, "Spectrogram", spec,
                     wave_length, samplerate)  # スペクトログラムデータの描画
    wave = i_stft(spec, wave_size)  # 短時間フーリエ逆変換
    draw_signal(ax3, "Re-synthesized signal", time, wave)  # 復元データの描画
    fig.tight_layout()  # レイアウト調整
    plt.savefig('ex01.png')  # 保存

    sf.write("out.wav", wave, samplerate,
             format="WAV", subtype='PCM_16')  # 変換後音声データの書き出し


if __name__ == "__main__":
    main()
