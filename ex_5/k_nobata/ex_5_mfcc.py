#!/usr/bin/env python
# coding: utf-8

import argparse

import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.signal as ssig
from scipy.fftpack import dct
import scipy

#係数(1.0, -p)のFIRフィルタを作成
def preemphasis(data,p):
    """
    paramerters
    ---
    data:numpy.ndarray
         audio time series
    p:int
      filter coefficient
    ---
    
    return
    ---
    f:numpy.ndarray
      FIR filter
    ---
    """
    f = ssig.lfilter([1.0,-p],1,data)
    return f

#メルフィルタバンクを作成
def melFB(sr, F_size, numCh):
    """
    parameters
    ---
    sr:int
       sampling rate
    F_size:int
           frame size
    numCh:int
          the number of channel
    ---
          
    return
    ---
    fb:numpy.ndarray
       mel filterbank
    ---
    
    """
    nf = sr / 2
    mel_nf = hz_to_mel(nf)
    nmax = F_size // 2
    df = sr / F_size
    dmel = mel_nf / (numCh + 1)
    mel_cen = np.arange(1, numCh + 1) * dmel
    f_cen = mel_to_hz(mel_cen)  
    i_cen = np.round(f_cen / df)
    i_sta = np.hstack(([0], i_cen[0:numCh - 1]))
    i_sto = np.hstack((i_cen[1:numCh], [nmax]))
    fb = np.zeros((numCh, nmax))

    for ch in range(numCh):
        increment= 1.0 / (i_cen[ch] - i_sta[ch])
        for i in range(int(i_sta[ch]), int(i_cen[ch])):
            fb[ch, i] = (i - i_sta[ch]) * increment
        
        decrement = 1.0 / (i_sto[ch] - i_cen[ch])
        for i in range(int(i_cen[ch]), int(i_sto[ch])):
            fb[ch, i] = 1.0 - ((i - i_cen[ch]) * decrement)

    return fb

#Hz,mel変換
def hz_to_mel(f):
    """
    paramerter
    ---
    f:numpy.ndarray or float
      frequency
    ---
    
    return
    ---
    m:numpy.ndarray or float
      mel frequency
    ---
    """
    m = 2595 * np.log(f / 700.0 + 1.0)
    return m

def mel_to_hz(m):
    """
    paramerter
    ---
    m:numpy.ndarray or float
      mel frequency
    ---
    
    return
    ---
    f:numpy.ndarray or float
      frequency
    ---
    """
    f = 700 * (np.exp(m / 2595) - 1.0)
    return f

#動的変動成分
def delta(data):
    """
    Parameter
    ---
    data : numpy.ndarray
        input data
    ---
    
    Return
    ---
    delta : numpy.ndarray
        delta data
    ---
    """

    i = 2
    new_data = np.vstack((data[0], data[0], data, data[-1], data[-1]))
    k = np.arange(-i, i+1)
    delta = np.zeros((data.shape[0], data.shape[1]))
    for j in range(data.shape[0]):
        delta[j] = np.dot(k, new_data[j:j+2*i+1]) / np.sum(np.square(k))

    return delta

def png(mfcc, d, dd, Ts):
    """
    Parameter
    ---
    mfcc:numpy.ndarray
         mfcc
    d : numpy.ndarray
        Δmfcc
    dd : numpy.ndarray
         ΔΔmfcc
    Ts:float
       time of sound data
    ---
    
    """

    fig = plt.figure(figsize=(10,5))
    
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    img = ax1.imshow(mfcc.T, extent=[0, Ts, 0, 12], aspect="auto", origin="lower")
    ax1.set_ylabel("MFCC", fontsize=15)
    fig.colorbar(img,  ax=ax1)
    
    img = ax2.imshow(d.T, extent=[0, Ts, 0, 12], aspect="auto", origin="lower")
    ax2.set_ylabel("ΔMFCC", fontsize=15)
    fig.colorbar(img, ax=ax2)

    img = ax3.imshow(dd.T, extent=[0, Ts, 0, 12], aspect="auto", origin="lower")
    ax3.set_ylabel("ΔΔMFCC", fontsize=15)
    fig.colorbar(img, ax=ax3)
    
    plt.xlabel("time[s]", fontsize=15)
    plt.tight_layout()
    plt.savefig("mfcc.png")
    plt.show()
    plt.close()

def main(args):
    #ファイル読込
    #fname = "/Users/nobatakoki/B4輪行/ccnobata/新規録音.wav"
    fname = args.fname
    data, sr = librosa.load(fname, sr=16000)

    #条件設定
    F_size = args.F_size
    overlap = F_size // 2 
    F_num = data.shape[0] 
    Ts = float(F_num) / sr 
    S_num = int(F_num // (F_size - overlap) - 1) 

    win = np.hamming(F_size)
    #p = 0.97
    p = args.p
    pe_data = preemphasis(data, p)

    #音声データを適切な長さのフレームに分割、窓関数を適応、対数を取る
    windata = np.zeros((S_num, F_size))
    for i in range(S_num):
        windata[i] = pe_data[i*overlap:i*overlap+F_size] * win
    fftdata = np.fft.fft(windata)
    logdata = np.log10(np.abs(fftdata))

    # メルフィルタバンクを作成
    #numCh = 20  # メルフィルタバンクのチャネル数
    numCh = args.numCh
    df = sr / F_size   # 周波数解像度（周波数インデックス1あたりのHz幅）
    fb = melFB(sr, F_size, numCh)

    #メルフィルタバンクを掛けて、メルスペクトルに変換
    mspec = np.log10(np.dot(np.abs(fftdata[:,:F_size//2]), fb.T))

    #離散コサイン変換
    mfcc = dct(mspec)

    #デルタ
    d = delta(mfcc)
    dd = delta(d)

    # メルフィルタバンクのプロット
    for i in np.arange(numCh):
        plt.plot(np.arange(0, F_size / 2) * df, fb[i])
    #savefig("melfilterbank.png")
    plt.show()

    plt.figure(figsize=(10,5))
    plt.specgram(data, Fs=sr, scale_by_freq="True")
    plt.colorbar()
    plt.title("Spectrogram", fontsize=15)
    plt.xlabel("time[s]", fontsize=15)
    plt.ylabel("frequency[Hz]", fontsize=15)
    plt.tight_layout()
    plt.savefig("spectrogram.png")
    plt.show()
    plt.close()
    png(mfcc, d, dd, Ts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", default="/Users/nobatakoki/B4輪行/ccnobata/新規録音.wav")
    parser.add_argument("--p", default=0.97)
    parser.add_argument("--numCh", default=20)
    parser.add_argument("--F_size", default=1024)
    args = parser.parse_args()

    main(args)