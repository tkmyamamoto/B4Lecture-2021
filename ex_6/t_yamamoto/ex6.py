# include flake8, black

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def standardize(data):
    """
    Standardize input data.

    Parameters:
        data : ndarray
            Input data from csv file.

    Returns:
        ndarray
            Standardized data.
    """
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


def pca(data, std=True):
    """
    Principal component analysis.

    Parameters:
        data : ndarray
            Input data from csv file.
        std : bool
            If std is True, data is standardized.

    Returns:
        sort_eig_val : ndarray (shape = (data.shape[1],))
            Eigenvalues sorted in descending order.
        sort_eig_vec : ndarray (shape = (data.shape[1], data.shape[1]))
            Eigenvectors sorted in descending order.
        cont_rate : ndarray (shape = (data.shape[1],))
            Contribution rate.
    """
    # standardization
    if std:
        data = standardize(data)

    # calculate covariance matrix
    diff_data = data - np.mean(data, axis=0)
    # cov.shape = (data.shape[1], data.shape[1])
    cov = (diff_data.T @ diff_data) / data.shape[0]

    # eigenvalue and eigenvector
    eig_val, eig_vec = np.linalg.eig(cov)

    # sort eigenvalue and eigenvector in descending order
    sort_eig_val = np.sort(eig_val)[::-1]
    sort_eig_vec = eig_vec[:, np.argsort(eig_val)[::-1]]

    # contribution rate
    cont_rate = sort_eig_val / np.sum(sort_eig_val)

    return sort_eig_val, sort_eig_vec, cont_rate


def main(args):
    """
    fname = "data3.csv"
    std = False
    """
    fname = args.fname
    std = args.std

    # get current working directory
    path = os.path.dirname(os.path.abspath(__file__))

    ftitle, _ = os.path.splitext(fname)

    # load csv file and convert to ndarray
    data = pd.read_csv(os.path.join(path, "data", fname), header=None).values

    _, eig_vec, cont_rate = pca(data, std=std)

    cmap = plt.get_cmap("tab10")

    if fname == "data1.csv":
        x_label = np.linspace(-np.max(np.abs(data)), np.max(np.abs(data)), 100)
        coef = eig_vec[1] / eig_vec[0]

        fig = plt.figure()
        fig.add_subplot(
            111,
            title=ftitle,
            xlabel="X1",
            ylabel="X2",
            aspect="equal",
        )
        plt.scatter(data[:, 0], data[:, 1], s=20, c="darkblue", label="data")
        for i in range(data.shape[1]):
            plt.plot(
                x_label,
                coef[i] * x_label,
                label=f"Contribution rate: {round(cont_rate[i], 3)}",
                c=cmap(i + 1),
            )
        plt.grid(ls="--")
        plt.legend()
        save_fname = os.path.join(path, "result", "data1_pca.png")
        plt.savefig(save_fname, transparent=True)
        plt.show()

    elif fname == "data2.csv":
        coef_y = eig_vec[1] / eig_vec[0]
        coef_z = eig_vec[2] / eig_vec[0]

        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            projection="3d",
            title=ftitle,
            xlabel="X1",
            ylabel="X2",
            zlabel="X3",
            xlim=[-2, 2],
            ylim=[-2, 2],
            zlim=[-2, 2],
        )
        ax.scatter3D(
            data[:, 0], data[:, 1], data[:, 2], s=20, c="darkblue", label="data"
        )
        for i in range(data.shape[1]):
            x_label = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
            x_label = x_label[-2 < coef_z[i] * x_label]
            x_label = x_label[coef_z[i] * x_label < 2]
            plt.plot(
                x_label,
                coef_y[i] * x_label,
                coef_z[i] * x_label,
                label=f"Contribution rate: {round(cont_rate[i], 3)}",
                c=cmap(i + 1),
            )
        plt.grid(ls="--")
        plt.legend()
        save_fname = os.path.join(path, "result", "data2_pca.png")
        plt.savefig(save_fname, transparent=True)

        def update(i):
            """
            Move view point.

            Parameters:
                i : int
                    Number of frames.

            Returns:
                fig : matplotlib.figure.Figure
                    Figure viewed from angle designated by view_init function.
            """

            ax.view_init(elev=30.0, azim=3.6 * i)
            return fig

        # animate
        ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
        save_fname = os.path.join(path, "result", "data2_pca.gif")
        ani.save(save_fname, writer="pillow")
        plt.show()

        data_dimreduced = standardize(data) @ eig_vec
        fig = plt.figure()
        fig.add_subplot(
            111,
            title=f"{ftitle}_transformed",
            xlabel="Z1",
            ylabel="Z2",
        )
        plt.scatter(
            data_dimreduced[:, 0],
            data_dimreduced[:, 1],
            s=20,
            c="darkblue",
            label="data",
        )
        plt.grid(ls="--")
        plt.legend()
        save_fname = os.path.join(path, "result", "data2_dim.png")
        plt.savefig(save_fname, transparent=True)
        plt.show()

    else:
        """
        sum = 0
        i = 0
        while sum < 0.9:
            sum += cont_rate[i]
            i += 1
        print(i, sum)
        """

        cum_cont_rate = np.cumsum(cont_rate)
        point = [np.where(cum_cont_rate < 0.9, 1, cum_cont_rate).argmin()]
        point.append(cum_cont_rate[point[0]])
        point[0] += 1
        print(point)

        fig = plt.figure()
        fig.add_subplot(
            111,
            title=f"{ftitle}_cumulative contribution rate",
            xlabel="Dimension",
            ylabel="Cumulative contribution rate",
            xticks=np.append(np.linspace(0, 100, 6), point[0]),
            yticks=np.append(np.linspace(0, 1, 6), 0.9),
        )
        plt.plot(range(1, len(cont_rate) + 1), cum_cont_rate, c="darkblue")
        plt.axvline(x=point[0], c="r", linestyle="dashed")
        plt.axhline(y=0.9, c="r", linestyle="dashed")
        plt.grid(ls="--")
        save_fname = os.path.join(path, "result", "data3_cont.png")
        plt.savefig(save_fname, transparent=True)
        plt.show()


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(description="k-means clustering.")
    parser.add_argument("fname", type=str, help="Load filename")
    parser.add_argument("--std", action="store_false", help="Standardization")
    args = parser.parse_args()
    main(args)
