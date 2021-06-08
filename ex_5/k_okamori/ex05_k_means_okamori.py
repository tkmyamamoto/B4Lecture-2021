import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import argparse


class ClusterByKmeans:  # k-means法を実装するクラス

    # コンストラクタ
    #   data: csvデータ
    #   n_divide: 分割数
    #   n_trial: 初期値を決める試行回数
    def __init__(self, fname, data, n_divide, n_trial):
        self.fname = fname
        self.data = data
        self.n_data = self.data.shape[0]  # データ数
        self.n_divide = n_divide
        self.n_trial = n_trial

    # 初期値の決定

    def set_default(self):
        self.default = np.zeros((self.n_divide, self.data.shape[1]))
        distance = 0

        # 試行回数の分だけランダムで点を取る
        for i in range(self.n_trial):
            rand = np.random.choice(  # 乱数の取得
                np.arange(self.n_data), self.n_divide, replace=False)

            # 点同士の距離の最小値が最大になる組み合わせを採用する
            distance_min = np.inf
            for j in range(self.n_divide-1):
                for k in range(j+1, self.n_divide):
                    distance_tmp = np.linalg.norm(
                        self.data[rand[j]]-self.data[rand[k]])
                    if distance_tmp < distance_min:
                        distance_min = distance_tmp

            if distance_min > distance:
                distance = distance_min
                for j in range(self.n_divide):
                    self.default[j] = self.data[rand[j]]

    # クラスタ分割

    def divide_cluster(self):

        self.data_and_tag = np.append(  # 各データの右端にクラスタを区別するタグをつける
            self.data, np.zeros((self.n_data, 1)), axis=1)  # (x, y, (z), tag)
        is_changed = True  # 前回から変化したかどうか

        # 重心が変化しなくなるまで続ける
        while is_changed:
            is_changed = False
            default_previous = np.copy(self.default)  # 前回の重心を記録

            # 最も近い重心のタグをつける
            for i in range(self.n_data):
                distance = np.inf
                for j in range(self.n_divide):
                    distance_tmp = np.linalg.norm(
                        self.data_and_tag[i, :-1]-self.default[j])
                    if distance_tmp < distance:
                        distance = distance_tmp
                        self.data_and_tag[i, -1] = j

            # 新しい重心を求める
            for i in range(self.n_divide):
                self.default[i] = np.average(
                    self.data_and_tag[self.data_and_tag[:, -1] == i, :-1], axis=0)

            # 重心が前回と変化したか判定
            is_changed = not np.array_equal(self.default, default_previous)

    def plot(self):
        self.fig = plt.figure()  # グラフ領域の確保
        ax = self.fig.add_subplot(111)  # グラフ領域の描画

        # クラスタの描画
        for i in range(self.n_divide):
            ax.scatter(self.data_and_tag[self.data_and_tag[:, -1] == i, 0],
                       self.data_and_tag[self.data_and_tag[:, -1] == i, 1])

        # 重心の描画
        ax.scatter(self.default[:, 0], self.default[:, 1])

        ax.set_title(f"data = {self.fname}, n = {self.n_divide}")  # タイトルの追加


class ClusterByKmeans3D(ClusterByKmeans):  # k-means法を実装するクラス(3次元)

    def plot(self):
        self.fig = plt.figure()  # グラフ領域の確保
        ax = self.fig.add_subplot(111, projection="3d")  # グラフ領域の描画

        # クラスタの描画
        for i in range(self.n_divide):
            ax.scatter(self.data_and_tag[self.data_and_tag[:, -1] == i, 0],
                       self.data_and_tag[self.data_and_tag[:, -1] == i, 1],
                       self.data_and_tag[self.data_and_tag[:, -1] == i, 2])

        # 重心の描画
        ax.scatter(self.default[:, 0], self.default[:, 1], self.default[:, 2])

        ax.set_title(f"data = {self.fname}, n = {self.n_divide}")  # タイトルの追加

# main


def main(args):
    csv = args.fname  # 元データ (CSV)
    n_divide = args.n_divide  # 分割する個数
    n_trial = args.n_trial  # 初期値決定の試行回数

    data = np.loadtxt(fname=csv, dtype="float",
                      delimiter=',', skiprows=1)  # csv の読み込み

    if data.shape[1] == 2:  # 2次元の場合
        kmeans = ClusterByKmeans(csv, data, n_divide, n_trial)
    elif data.shape[1] == 3:  # 3次元の場合
        kmeans = ClusterByKmeans3D(csv, data, n_divide, n_trial)
    else:
        print(f"対応していないデータ形式です．")  # エラー表示
        sys.exit(1)  # プログラム終了

    kmeans.set_default()  # 初期値の決定
    kmeans.divide_cluster()  # クラスタ分割
    kmeans.plot()  # 結果のプロット
    plt.savefig(f"ex05_k_means({csv}).png")  # 出力ファイルの保存


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-means 法によるクラスタ分割")
    parser.add_argument("fname", type=str, help="ファイル名")
    parser.add_argument("n_divide", type=int, help="分割する個数")
    parser.add_argument("n_trial", type=int, help="初期値決定の試行回数")
    args = parser.parse_args()

    main(args)
