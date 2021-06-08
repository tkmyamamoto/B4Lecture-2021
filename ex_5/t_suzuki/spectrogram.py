import numpy as np
from scipy import signal


def fft(y, frame_size, overlap):
    spec = []
    ynum = y.shape[0]
    window_func = signal.hamming(frame_size)

    for i in range(0, ynum, int(frame_size-overlap)):
        frame = y[i:i+frame_size]
        if len(frame) == frame_size:
            frame = frame * window_func
            spec.append(np.fft.fft(frame)[::-1])
   
    return np.array(spec)


def spec(y, frame_size, overlap):
    F = []
    ynum = y.shape[0]
    window_func = signal.hamming(frame_size)

    for i in range(0, ynum, int(frame_size-overlap)):
        frame = y[i:i+frame_size]
        if len(frame) == frame_size:
            frame = frame * window_func
            F.append(np.fft.rfft(frame)[::-1])
   
    F = np.transpose(F)
    return 20 * np.log10(np.abs(F))
