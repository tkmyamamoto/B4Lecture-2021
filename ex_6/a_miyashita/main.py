import argparse

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg.linalg import norm


def normalize(data, axis=None):
    """
        Normalize input data.

        # Args
            data (ndarray): input data
            axis ({int, None}, default=None):
                Axis that statistics are calculated along. If axis==None, 
                data is normalized along all dimensions.

        # Returns
            out (ndarray): normalized data
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=None, keepdims=True)
    return (data - mean) / std


class Pca:
    """
        Class for principal component analysis.
    """
    def fit(self, x, y=None):
        """
            Calculate principal component of given data.

            # Args
                x (ndarray, shape=(sample_size, sample_dimension))
                    :input data
                y (None): dummy
        """
        # calculate covariance
        x_ = np.mean(x, axis=0)
        dev = x - x_
        cov = dev.T @ dev

        # solve eigenvalue problem
        lam, W = np.linalg.eig(cov)

        # sort by contribution rate
        idx = np.argsort(lam)[::-1]
        self.lam = lam[idx]
        self.W = W[:,idx]

    def predict(self, x):
        """
            Convert basis to that calculated by PCA.

            # Args
                x (ndarray, shape=(sample_size, sample_dimension))
                    :input data

            # Returns
                out (ndarray, shape=(sample_size, sample_dimension)):
                    coefficient of new basis.
        """
        return x @ self.W

    def contribution_rate(self):
        """
            Calculate contribution rate.

            # Returns
                cr (ndarray, axis=(dimension,)): contribution rate
                ccr (ndarray, axis=(dimension,)): cumlative contribution rate
        """
        # calculate contribution rate
        cr = self.lam / np.sum(self.lam)
        # calculate cumlative contribution rate
        ccr = np.cumsum(cr)
        ccr = np.insert(ccr, 0, 0)
        return cr, ccr



def main():
    # process args
    parser = argparse.ArgumentParser(description="Principal component analysis")
    parser.add_argument("sc", type=str, help="input filename followed by extention .csv")
    args = parser.parse_args()

    data = np.loadtxt(f"{args.sc}.csv", delimiter=',')
    print(f">>> data = np.loadtxt({args.sc}.csv, delimiter=',')")
    print(">>> print(data.shape)")
    print(data.shape)

    data = normalize(data, axis=0)

    pca = Pca()
    pca.fit(data)
    # calculate contribution rate
    cr, ccr = pca.contribution_rate()

    if data.shape[1] == 2:
        # plot
        plt.plot([0, pca.W[0, 0]], [0, pca.W[1, 0]], label=f"$z_0 (cr={cr[0]:.3f})$")
        plt.plot([0, pca.W[0, 1]], [0, pca.W[1, 1]], label=f"$z_1 (cr={cr[1]:.3f})$")
        plt.scatter(data[:,0], data[:,1], facecolor='None', edgecolors='r')
        plt.title(args.sc)
        plt.xlabel("$x_0$")
        plt.ylabel("$x_1$")
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
        plt.axes().set_aspect('equal')
        plt.show()

    if data.shape[1] == 3:
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot([0, pca.W[0, 0]], [0, pca.W[1, 0]], [0, pca.W[2, 0]], label=f"$z_0 (cr={cr[0]:.3f})$")
        ax.plot([0, pca.W[0, 1]], [0, pca.W[1, 1]], [0, pca.W[2, 1]], label=f"$z_1 (cr={cr[1]:.3f})$")
        ax.plot([0, pca.W[0, 2]], [0, pca.W[1, 2]], [0, pca.W[2, 2]], label=f"$z_2 (cr={cr[2]:.3f})$")
        ax.scatter3D(data[:,0], data[:,1], data[:,2], c='r')
        ax.set_title(args.sc)
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")
        ax.set_zlabel("$x_2$")
        ax.legend()
        plt.show()

        data_pca = pca.predict(data)

        plt.scatter(data_pca[:,0], data_pca[:,1], facecolor='None', edgecolors='r')
        plt.title(args.sc)
        plt.xlabel("$z_0$")
        plt.ylabel("$z_1$")
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
        plt.axes().set_aspect('equal')
        plt.show()

    if data.shape[1] >= 4:
        # plot
        plt.plot(ccr)
        plt.plot([0, ccr.size], [0.9, 0.9], linestyle='dashed')
        plt.title(f"{args.sc} CCR")
        plt.xlabel("dimension")
        plt.ylabel("cumlative contribution rate")
        plt.show()

        data_pca = pca.predict(data)

        plt.scatter(data_pca[:,0], data_pca[:,1], facecolor='None', edgecolors='r')
        plt.title(args.sc)
        plt.xlabel("$z_0$")
        plt.ylabel("$z_1$")
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
        plt.axes().set_aspect('equal')
        plt.show()

        plt.scatter(data_pca[:,98], data_pca[:,99], facecolor='None', edgecolors='r')
        plt.title(args.sc)
        plt.xlabel("$z_{{98}}$")
        plt.ylabel("$z_{{99}}$")
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
        plt.axes().set_aspect('equal')
        plt.show()


if __name__ == "__main__":
    main()      