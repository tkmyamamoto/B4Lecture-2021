import numpy as np
import scipy.signal as sp  # プリエンファシスフィルタ作成に使用
from scipy.fftpack.realtransforms import dct  # 離散コサイン変換
import soundfile as sf  # 音声データの読み込み
import matplotlib.pyplot as plt  # 結果のプロット

# 定数の定義
NFFT = 1024  # frame の大きさ
OVERLAP = int(NFFT // 2)  # half shift を採用
window = np.hamming(NFFT)  # 窓関数はハミング窓を使用
P = 0.97  # プリエンファシス係数
F0 = 700  # 周波数パラメータ
M0 = 1000 / np.log(1000/F0+1)  # F0 から導かれるパラメータ
CHANNEL = 20  # メルフィルタバンクのチャネル数
MFCC_DIM = 12  # 離散コサイン変換後取り出す次数
DELTA_LENGTH = 2  # デルタ計算に用いる前後のフレーム数

# MFCC を実装するクラス


class Mfcc:
    # コンストラクタ
    def __init__(self, fname):
        self.wave, self.samplerate = sf.read(fname)

    # Hz を mel に変換
    def hz_2_mel(self, f):
        return M0 * np.log(f / F0 + 1)

    # mel を Hz に変換
    def mel_2_hz(self, m):
        return F0 * (np.exp(m / M0) - 1)

    # 短時間フーリエ変換
    def stft(self):
        self.spec = np.zeros([int(NFFT / 2) + 1,
                              len(self.wave) // OVERLAP - 1],)  # self.spec:出力データ

        for idx in range(0, len(self.wave), OVERLAP):
            frame = self.wave[idx: idx + NFFT]  # frame の切り出し
            if len(frame) == NFFT:  # frame が全部切り出せるときのみ変換
                frame = sp.lfilter([1.0, -P], 1, frame)  # プリエンファシス
                windowed = window * frame  # 窓関数をかける
                result = np.fft.rfft(windowed)  # フーリエ変換
                for i in range(self.spec.shape[0]):
                    # 計算結果を[dB]にして出力データに追加
                    self.spec[i][int(idx / OVERLAP)] \
                        = 20 * np.log10(np.abs(result[i]))

    # メルフィルタバンクの作成
    def create_mel_filter_bank(self):
        self.mel_filter_bank = np.zeros((CHANNEL, NFFT // 2 + 1))  # メルフィルタバンク
        mel_max = self.hz_2_mel(self.samplerate // 2)  # メル尺度での最大の周波数
        self.mfb_center_list = np.zeros(CHANNEL)  # 三角波の頂点のリスト(プロット用)

        for i in range(CHANNEL):
            start = int(np.round(self.mel_2_hz(mel_max * i / (CHANNEL + 1))
                                 * (NFFT // 2) / (self.samplerate // 2)))  # 三角波の始点
            middle = int(np.round(self.mel_2_hz(mel_max * (i + 1) / (CHANNEL + 1))
                                  * (NFFT // 2) / (self.samplerate // 2)))  # 三角波の頂点
            end = int(np.round(self.mel_2_hz(mel_max * (i + 2) / (CHANNEL + 1))
                               * (NFFT // 2) / (self.samplerate // 2)))  # 三角波の終点

            self.mfb_center_list[i] \
                = self.mel_2_hz(mel_max * (i + 1) / (CHANNEL + 1))  # 三角波の頂点をリストに格納

            self.mel_filter_bank[i, start: middle + 1] \
                = np.arange(0, middle + 1 - start) / (middle - start)  # 三角波の前半
            self.mel_filter_bank[i, end: middle: -1] \
                = np.arange(0, end - middle) / (end - middle)  # 三角波の後半

    # MFCC, ΔMFCC, ΔΔMFCC の作成
    def create_mfcc(self):
        spec_mfb \
            = np.dot(self.mel_filter_bank, self.spec)  # スペクトル[dB]とメルフィルタバンクの内積
        self.mfcc \
            = dct(spec_mfb, type=2, norm="ortho", axis=0)[:MFCC_DIM]  # 離散コサイン変換
        self.d_mfcc = self.create_delta(self.mfcc, DELTA_LENGTH)  # ΔMFCC
        self.dd_mfcc = self.create_delta(self.d_mfcc, DELTA_LENGTH)  # ΔΔMFCC

    # デルタの作成
    #   mfcc: 元データ
    #   length: デルタ計算に用いる前後のフレーム数
    def create_delta(self, mfcc, length):
        delta = np.zeros_like(mfcc)  # デルタを格納する配列
        mfcc = np.pad(mfcc, ((0, 0), (length, length)),
                      "edge")  # 両端に端と同じデータを追加

        # 最小二乗法を用いて回帰係数を求める
        for i in range(length, mfcc.shape[1] - length):
            for j in range(MFCC_DIM):
                x = np.vstack((np.ones(2 * length + 1),
                              np.arange(-length, length + 1)))
                y = mfcc[j, i - length: i + length + 1]
                delta[j, i - length] = (np.linalg.inv(x @ x.T)
                                        @ x @ y.T)[1]  # 正規方程式

        return delta

    # 結果のプロット
    def draw_spec(self):
        self.fig = plt.figure(figsize=(8.0, 10.0))  # グラフ領域の確保
        ax1 = self.fig.add_subplot(511)  # Original Signal
        ax2 = self.fig.add_subplot(512)  # Mell Filter Bank
        ax3 = self.fig.add_subplot(513)  # MFCC
        ax4 = self.fig.add_subplot(514)  # dMFCC
        ax5 = self.fig.add_subplot(515)  # ddMFCC
        wave_length = len(self.wave) // self.samplerate  # 音源データの長さ

        im1 = ax1.imshow(self.spec, extent=[0, wave_length, 0, self.samplerate / 2],
                         cmap="rainbow", aspect="auto", origin="lower")  # スペクトログラムの描画
        self.fig.colorbar(im1, ax=ax1)
        ax1.set_title("Originl Signal")   # タイトルの追加
        ax1.set_xlabel("time[s]")         # x軸ラベルの追加
        ax1.set_ylabel("Frequency[kHz]")  # y軸ラベルの追加

        for i in range(self.mel_filter_bank.shape[0]):
            ax2.plot(np.arange(0, self.samplerate // 2, self.samplerate // 2 /
                     self.mel_filter_bank.shape[1]), self.mel_filter_bank[i])  # プロット
        ax2.set_title("Mel Filter Bank")   # タイトルの追加
        ax2.set_xlabel("Frequency[Hz]")         # x軸ラベルの追加

        im3 = ax3.imshow(self.mfcc, extent=[0, wave_length, 0, MFCC_DIM],
                         cmap="rainbow", aspect="auto", origin="lower")  # スペクトログラムの描画
        self.fig.colorbar(im3, ax=ax3)
        ax3.set_title("MFCC")             # タイトルの追加
        ax3.set_xlabel("time[s]")         # x軸ラベルの追加
        ax3.set_ylabel("MFCC")            # y軸ラベルの追加

        im4 = ax4.imshow(self.d_mfcc, extent=[0, wave_length, 0, MFCC_DIM],
                         cmap="rainbow", aspect="auto", origin="lower")  # スペクトログラムの描画
        self.fig.colorbar(im4, ax=ax4)
        ax4.set_title("ΔMFCC")            # タイトルの追加
        ax4.set_xlabel("time[s]")         # x軸ラベルの追加
        ax4.set_ylabel("ΔMFCC")           # y軸ラベルの追加

        im5 = ax5.imshow(self.dd_mfcc, extent=[0, wave_length, 0, MFCC_DIM],
                         cmap="rainbow", aspect="auto", origin="lower")  # スペクトログラムの描画
        self.fig.colorbar(im5, ax=ax5)
        ax5.set_title("ΔΔMFCC")           # タイトルの追加
        ax5.set_xlabel("time[s]")         # x軸ラベルの追加
        ax5.set_ylabel("ΔΔMFCC")          # y軸ラベルの追加

        self.fig.tight_layout()  # レイアウト調整

# main


def main():
    fname = "aiueo.wav"  # 入力ファイル名
    mfcc = Mfcc(fname)  # インスタンス化

    mfcc.stft()  # 短時間フーリエ変換
    mfcc.create_mel_filter_bank()  # メルフィルタバンクの作成
    mfcc.create_mfcc()  # MFCC, ΔMFCC, ΔΔMFCC の作成
    mfcc.draw_spec()  # 結果のプロット

    plt.savefig("ex05_mfcc.png")  # 結果の保存


if __name__ == "__main__":
    main()
