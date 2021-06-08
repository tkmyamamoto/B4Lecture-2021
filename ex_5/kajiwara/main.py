import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import librosa

import kmeans
import mfcc

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def main(args):
    audio_path = Path(args.audio_path)
    wave_data, sr = librosa.load(audio_path)

    df1 = pd.read_csv('../data1.csv')
    df2 = pd.read_csv('../data2.csv')
    df3 = pd.read_csv('../data3.csv')

    result_path = Path(args.save_path)
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    result_path = result_path/timestamp
    if not result_path.exists():
        try:
            result_path.mkdir(parents=True)
        except Exception as err:
            print(err)

    '''clustering data1'''
    clf_1 = kmeans.KMeans(n_clusters=4, init='random', max_ite=300, random_state=44)
    pred_1 = clf_1.fit_predict(df1.values)
    df1['pred'] = pred_1
    # df1.plot(kind="scatter", x='x', y='y', c="pred", cmap='rainbow')
    plt.scatter(df1['x'], df1['y'], c=df1['pred'])
    plt.savefig(result_path/"data1.png")
    plt.clf()
    plt.close()

    '''clustering data2'''
    clf_2 = kmeans.KMeans(n_clusters=2, init='random', max_ite=300, random_state=44)
    pred_2 = clf_2.fit_predict(df2.values)
    df2['pred'] = pred_2
    # df2.plot(kind="scatter", x='x', y='y', c="pred", cmap='rainbow')
    plt.scatter(df2['x'], df2['y'], c=df2['pred'])
    plt.savefig(result_path/"data2.png")
    plt.clf()
    plt.close()

    '''clustering data3'''
    clf_3 = kmeans.KMeans(n_clusters=4, init='random', max_ite=300, random_state=44)
    pred_3 = clf_3.fit_predict(df3.values)
    df3['pred'] = pred_3

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df3['x'].values, df3['y'].values, df3['z'].values, c=df3['pred'])
    plt.legend()
    plt.savefig(result_path/'data3.png')
    plt.clf()
    plt.close()

    '''mfcc'''
    win_size = 2048
    overlap = 0.5
    bank_size = 20

    _mfcc = mfcc.get_mfcc(wave_data, sr, win_size, overlap, bank_size)

    plt.imshow(_mfcc, cmap='rainbow', aspect='auto', origin='lower')
    plt.colorbar()
    plt.savefig(result_path/'mfcc.png')
    plt.clf()
    plt.close()

    dmfcc = mfcc.calc_delta(_mfcc)
    plt.imshow(dmfcc, cmap='rainbow', aspect='auto', origin='lower')
    plt.colorbar()
    plt.savefig(result_path/'dmfcc.png')
    plt.clf()
    plt.close()

    ddmfcc = mfcc.calc_delta(dmfcc)
    plt.imshow(ddmfcc, cmap='rainbow', aspect='auto', origin='lower')
    plt.colorbar()
    plt.savefig(result_path/'ddmfcc.png')
    plt.clf()
    plt.close()


if __name__ == "__main__":
    description = 'Example: python main.py ./audio/sample.wav -s ./result'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('audio_path', help='path to audio data')
    parser.add_argument('-s', '--save-path', default='./result', help='path to save the result')

    args = parser.parse_args()

    main(args)
