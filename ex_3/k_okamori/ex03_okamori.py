import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# 最小二乗法を実現するクラス(1変数)


class LeastSqueares():

    # コンストラクタ
    #   data_observed: csv から得たデータ
    #   degree: 近似に用いる次数
    #   nc: 正則化項
    def __init__(self, data_observed, degree, nc):
        self.data_observed = data_observed
        self.degree = degree
        self.nc = nc

    # 係数を格納する行列の作成
    #   [係数, x^n]
    def create_polynomial(self):
        list = np.zeros((self.degree+1, 2))
        list[:, 1] = np.arange(self.degree+1)

        return list

    # 回帰係数の計算
    def calculate(self):
        self.poly_list = self.create_polynomial()  # 係数を格納する行列の作成
        x_observed = np.ones(
            (self.data_observed.shape[0], self.poly_list.shape[0]))  # x の観測値
        y_observed = np.array(self.data_observed[:, 1])  # y の観測値

        for i in range(1, self.poly_list.shape[0]):
            x_observed[:, i] = self.data_observed[:,
                                                  0] ** self.poly_list[i, 1]  # X(行列)の作成

        self.poly_list[:, 0] = (np.linalg.inv(
            x_observed.T @ x_observed + self.nc * np.eye(x_observed.shape[1]))
            @ x_observed.T @ y_observed.T)  # 正規方程式の解の導出

        return self.poly_list

    # 結果の出力
    def plot(self):
        self.fig = plt.figure()  # グラフ領域の確保
        ax = self.fig.add_subplot(111)  # グラフ領域の描画

        x_ideal = np.arange(self.data_observed[:, 0].min(),
                            self.data_observed[:, 0].max(), 0.01)  # 回帰式の x 座標
        y_ideal = np.full(x_ideal.shape[0], self.poly_list[0, 0])  # 回帰式の y 座標
        fx_name = "$y={:.2f}".format(self.poly_list[0, 0])  # 回帰式の名前(凡例用)

        for i in range(1, self.poly_list.shape[0]):  # 回帰式に従って x, y 座標に代入
            y_ideal += (self.poly_list[i, 0] *
                        x_ideal ** self.poly_list[i, 1])
            fx_name += "{:+.2f}x".format(self.poly_list[i, 0])  # 以下、回帰式の名前の編集
            if self.poly_list[i, 1] == 1:
                continue
            fx_name += "^{{{:.0f}}}".format(self.poly_list[i, 1])

        fx_name += "$"

        ax.scatter(
            self.data_observed[:, 0], self.data_observed[:, 1],
            label="observed value")  # 元データの散布図の描画
        ax.plot(x_ideal, y_ideal, color="red", label=fx_name)  # 回帰式の描画
        ax.legend()  # レイアウト調整

# 最小二乗法を実現するクラス(2変数)


class LeastSqueares3D(LeastSqueares):  # 1変数のクラスを継承

    # 係数を格納する行列の作成
    #   [係数, x^n, y^n]
    def create_polynomial(self):
        list = np.zeros((1, 3))

        for i in range(1, self.degree+1):
            for j in range(i+1):
                list = np.append(list, np.array([[0, i-j, j]]), axis=0)

        return list

    # 回帰係数の計算
    def calculate(self):
        self.poly_list = self.create_polynomial()  # 係数を格納する行列の作成
        xy_observed = np.ones(
            (self.data_observed.shape[0], self.poly_list.shape[0]))  # x, y の観測値
        z_observed = np.array(self.data_observed[:, 2])  # z の観測値

        for i in range(1, self.poly_list.shape[0]):
            xy_observed[:, i] = (self.data_observed[:, 0] ** self.poly_list[i, 1]
                                 * self.data_observed[:, 1] ** self.poly_list[i, 2])  # X(行列)の作成

        self.poly_list[:, 0] = (np.linalg.inv(
            xy_observed.T @ xy_observed + self.nc * np.eye(xy_observed.shape[1]))
            @ xy_observed.T @ z_observed.T)  # 正規方程式の解の導出

        return self.poly_list

    # 結果のプロット
    def plot(self):
        self.fig = plt.figure()  # グラフ領域の確保
        ax = self.fig.add_subplot(111, projection="3d")  # グラフ領域の描画

        x_ideal = np.arange(self.data_observed[:, 0].min(),
                            self.data_observed[:, 0].max(), 0.01)  # 回帰式の x 座標
        y_ideal = np.arange(self.data_observed[:, 1].min(),
                            self.data_observed[:, 1].max(), 0.01)  # 回帰式の ｙ 座標
        z_ideal = np.full((x_ideal.shape[0], y_ideal.shape[0]),
                          self.poly_list[0, 0]).T  # 回帰式の z 座標
        x_ideal, y_ideal = np.meshgrid(x_ideal, y_ideal)  # xy 平面の格子点の作成
        fx_name = "$z={:.2f}".format(self.poly_list[0, 0])  # 回帰式の名前(凡例用)

        for i in range(1, self.poly_list.shape[0]):  # 回帰式に従って x, y 座標に代入
            z_ideal += (self.poly_list[i, 0] *
                        x_ideal ** self.poly_list[i, 1] *
                        y_ideal ** self.poly_list[i, 2])
            fx_name += "{:+.2f}".format(self.poly_list[i, 0])  # 以下、回帰式の名前の編集
            if self.poly_list[i, 1] >= 1:
                fx_name += "x"
                if self.poly_list[i, 1] > 1:
                    fx_name += "^{{{:.0f}}}".format(self.poly_list[i, 1])
            if self.poly_list[i, 2] >= 1:
                fx_name += "y"
                if self.poly_list[i, 2] > 1:
                    fx_name += "^{{{:.0f}}}".format(self.poly_list[i, 2])

        fx_name += "$"

        ax.scatter(
            self.data_observed[:, 0], self.data_observed[:, 1],
            self.data_observed[:, 2], label="observed value")  # 元データの散布図の描画
        ax.plot_wireframe(x_ideal, y_ideal, z_ideal,
                          color="red", label=fx_name)  # 回帰式の描画
        ax.legend()  # レイアウト調整

# main


def main():
    csv = sys.argv[1]  # 元データ (CSV)
    degree = int(sys.argv[2])  # 回帰式の次数
    nc = float(sys.argv[3])  # 正則化項

    data_observed = np.loadtxt(
        fname=csv, dtype="float", delimiter=',', skiprows=1)  # csv の読み込み

    if data_observed.shape[1] == 2:  # 1変数の場合
        data = LeastSqueares(data_observed, degree, nc)
    elif data_observed.shape[1] == 3:  # 2変数の場合
        data = LeastSqueares3D(data_observed, degree, nc)

    if "data" in locals():
        data.calculate()  # 回帰式の計算
        data.plot()  # 結果のプロット
        plt.savefig("ex03(" + csv + ").png")  # 出力ファイルの保存
    return


if __name__ == "__main__":
    main()
