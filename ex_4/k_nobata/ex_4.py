#!/usr/bin/env python
# coding: utf-8

import argparse

import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssig

#自己相関関数
def autocorrelation(data, order=None):
    """
    parameters
    --
    data:numpy.ndarray
         audio time series

    returns
    ---
    ac:numpy.ndarray
       autocorrelation
    """
    len_data = len(data)
    ac = np.zeros(len_data)
    if order is None:
        order = len_data
    for l in range(order):
        for i in range(len_data - l):
            ac[l] += data[i] * data[i+l]
        
    return ac

#自己相関関数のピーク
def peak(ac):
    """
    parameter
    ---
    ac:numpy.ndarray
       autocorrelation
       
    return
    ---
    m:int
      peak of input
    """
    peak = np.zeros(ac.shape[0]-2)
    #前後で比較
    for i in range(ac.shape[0]-2):#最初と最後は除く
        if ac[i]<ac[i+1] and ac[i+1]>ac[i+2]:
            peak[i] = ac[i+1]
    m0 = np.argmax(peak)
    return m0

#自己相関法による基本周波数
def f0_ac(data, sr, F_size, overlap, S_num):
    """
    parameters
    --
    data:numpy.ndarray
         audio time series
    sr:int
       sampling rate
    F_size:int
           frame size
    overlap:int
            overlap size
    S_num:int
           a number of shift
           
    return
    --
    f0_ac:np.ndarray
         fundamental frequency
    """
    win = np.hamming(F_size) 
    f0_ac = np.zeros(S_num)
    for i in range(S_num):
        windata = data[i*overlap:i*overlap+F_size] * win
        cor = autocorrelation(windata)
        peak_m = peak(cor[:len(cor)//2])
        if peak_m == 0:
            f0_ac[i] = 0
        else:
            f0_ac[i] = sr / peak_m
            
    return f0_ac

#ケプストラム作成
def cepstrum(data):
    """
    parameters
    --
    data:numpy.ndarray
         audio time series

    returns
    --
    cep:numpy.ndarray
        cepstrum
    
    """
    power_spe = np.fft.fft(data) 
    log_power_spec = np.log10(np.abs(power_spe))
    cep = np.real(np.fft.ifft(log_power_spec)) 
    
    return cep

#ケプストラム法による基本周波数
def f0_cep(data, sr, F_size, overlap, S_num, lif):
    """
    parameters
    --
    data:numpy.ndarray
         audio time series
    sr:int
       sampling rate
    F_size:int
           frame size
    overlap:int
            overlap size
    S_num:int
           a number of frame
    lif:int
        lifter index
           
    return
    --
    f0:np.ndarray
       fundamental frequency
    """
    win = np.hamming(F_size)
    f0 = np.zeros(S_num)
    for i in range(S_num):
        windata = data[i*overlap:i*overlap+F_size] * win
        cep = cepstrum(windata)
        
        m0 = np.argmax(cep[lif:len(cep)//2]) + lif
        f0[i] = sr / m0
    
    return f0

#基本周波数とスペクトログラムのプロット
def f_plot(f0_a, f0_c, data, Ts, sr, lif):
    """
    parameters
    --
    f0_a and f0_c:np.ndarray
                  fundamental frequency
    data:numpy.ndarray
         audio time series
    sr:int
       sampling rate
    Ts:float
       time of sound data
    sr:int
       sampling rate
    """
    plt.figure(figsize=(10,8))
    plt.specgram(data, Fs=sr, cmap="rainbow", scale_by_freq="True")
    xa_0 = Ts/len(f0_a)
    xc_0 = Ts/len(f0_c)
    xa_axis = np.arange(xa_0, Ts, xa_0)
    xc_axis = np.arange(xc_0, Ts, xc_0)
    plt.plot(xa_axis, f0_a[1:], label="Autocorrelation", color="blue")
    plt.plot(xc_axis, f0_c[1:], label="Cepstrum", color="green")
    plt.xlabel("Times[s]", fontsize=15)
    plt.ylabel("Frequency[Hz]", fontsize=15)
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("spectrogram"+str(lif)+".png")
    plt.show()
    plt.close()

def cep_m(data, lif):
    """
    parameters
    ---
    data:numpy.ndarray
         audio time series
    lif:int
        lifter index
    
    return
    ---
    cep_env:numpy.ndarray
            spectral envelop
    """
    cep = cepstrum(data)
    cep[lif:len(cep)-lif] = 0
    
    cep_env = 20 * np.real(np.fft.fft(cep))
    
    return cep_env

#レビンソンダービン
def LevinsonDurbin(r, order):
    """
    pareameters
    ---
    r:numpy.ndarray
      autocorrelation
    order:int
          degree
    
    returns
    ---
    a:numpy.ndarray
    　lpc coefficient
    e:numpy.ndarray
      minimun error
    """
    a = np.zeros(order + 1)
    e = np.zeros(order + 1)

    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]

    for k in range(1, order):
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]

        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)

        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]

#係数(1.0, -p)のFIRフィルタを作成
def preemphasis(data,p):
    """
    paramerters
    ---
    data:numpy.ndarray
         audio time series
    p:int
      filter coefficient
    
    return
    ---
    f:numpy.ndarray
      FIR filter
    """
    f = ssig.lfilter([1.0,-p],1,data)
    return f

#lpc法によるスペクトル包絡
def lpc_m(data, order, F_size):
    """
    parameters
    ---
    data:numpy.ndarray
         audio time series      
    order:int
          degree
    F_size:int
           frame size
    return
    ---
    lpc_env:numpy.ndarray
            spectral envelop
    """
    r = autocorrelation(data, order+1)
    r = r[:len(r)//2]
    a, e = LevinsonDurbin(r, order)
    
    w, h = ssig.freqz(np.sqrt(e), a, F_size, "whole")
    lpc_env = 20*np.log10(np.abs(h))
    
    return lpc_env

#スペクトル包絡描画
def spe(log, cep, lpc, F_size, sr, lif, deg):
    """
    parameters
    ---
    log:numpy.ndarray
        spectrum
    cep:numpy.ndarray
        spectral envelop
    lpc:numpy.ndarray
        spectral envelop
    F_size:int
           frame size
    sr:int
       sampling rate
    
    """
    plt.figure(figsize=(10,8))
    f_axis = np.fft.fftfreq(F_size, d=1.0/sr)
    plt.plot(f_axis[:F_size//2], log[:len(log)//2], label="Spectrum", color="blue")
    plt.plot(f_axis[:F_size//2], cep[:len(log)//2], label="Cepstrum", color="green")
    plt.plot(f_axis[:F_size//2], lpc[:len(log)//2], label="LPC", color="red")
    plt.xlabel("Frequency[Hz]", fontsize=15)
    plt.ylabel("Log amplitude spectrum[dB]", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectrum"+str(lif)+"_"+str(deg)+".png")
    plt.show()
    plt.close()

def main(args):
    #f_name = "/Users/nobatakoki/B4輪行/ccnobata/新規録音.wav"
    f_name = args.fname
    data, sr = librosa.load(f_name,sr=16000)    

    #基本周波数計算用
    F_size = 1024 #frame size
    overlap = F_size // 2 #オーバーラップ率
    F_num = data.shape[0] # フレームの要素数
    Ts = float(F_num) / sr # 波形の長さ
    S_num = int(F_num // (F_size - overlap) - 1)  # 短時間区間数
    win = np.hamming(F_size)
    #lif = 32
    lif = args.lif
    
    #基本周波数計算&プロット
    f0_c = f0_cep(data, sr, F_size, overlap, S_num, lif)
    f0_a = f0_ac(data, sr, F_size, overlap, S_num)
    f_plot(f0_a, f0_c, data, Ts, sr, lif)
    
    #スペクトル包絡用
    p = 0.97
    s = 2.0
    s_frame  =int(s*sr)
    pe_deta = preemphasis(data, p)
    windata=pe_deta[s_frame:s_frame+F_size] * win
    #deg = 64
    deg = args.deg
    
    #計算
    log = 20 * np.log10(np.abs(np.fft.fft(windata)))
    cep = cep_m(windata, lif)
    lpc = lpc_m(windata, deg, F_size)
    
    spe(log, cep, lpc, F_size, sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", default="/Users/nobatakoki/B4輪行/ccnobata/新規録音.wav")
    parser.add_argument("--lif", default=32)
    parser.add_argument("--deg", default=64)
    args = parser.parse_args()

    main(args)
