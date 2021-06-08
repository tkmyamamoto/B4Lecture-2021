import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# from mpl_toolkits.mplot3d.axes3d import Axes3D


def center(S):
    """
    input  S   : Set of clusters
    output res : center of gravity of each clusters
    """
    
    res = []

    if type(S) is list:
        for s in S:
            c = np.array(s).sum(axis=0) / len(s)
            res.append(c)
    else:
        c = S.sum(axis=0) / len(S)
        res.append(c)
        
    return np.array(res)


def k_means(data, k, eps, A, D0=np.inf):
    """
    input
    data : ndarray
           target of clustering data
    k    : int
           number of clusters
    eps  : float
           threshold of kmeans
    A    : ndarray
           before codebook
    D0   : float
           before error
    output
    A    : ndarray
           clustering codebook
    S    : ndarray in list
           clustering set
    """

    n = len(data)
    if data.shape[1] == 2:
        tmp = np.empty((0, 2))
    else:
        tmp = np.empty((0, 3))
    S = [tmp for _ in range(k)]

    # calc distance between data and A
    dist_list = np.sqrt(((data[:, :, np.newaxis] - A.T[np.newaxis, :, :])**2).sum(axis=1))
    near_point = dist_list.argmin(axis=1)
    for idx, x in zip(near_point, data):
        S[idx] = np.r_[S[idx], [x]]

    # calc error
    for i in range(dist_list.shape[1]):
        dist_list.T[i][near_point != i] = 0
    D = dist_list.sum() / n

    if (D0 - D) / D < eps:
        return A, S
    else:
        return k_means(data, k, eps, center(S), D)


def lbg(data, k, eps, A, M=1):
    """
    input
    data : ndarray
           target of clustering data
    k    : int
           number of clusters
    eps  : float
           threshold of kmeans
    A    : ndarray
           before codebook
    M    : int
           current number of clusters
    output
    A    : ndarray
           initial codebook
    """

    delta = np.array([1e-5 for _ in range(data.shape[1])])

    if data.shape[1] == 2:
        tmp = np.empty((0, 2))
    else:
        tmp = np.empty((0, 3))
    for y in A:
        tmp = np.r_[tmp, [y + delta]]
        tmp = np.r_[tmp, [y - delta]]
    A = tmp

    A, S = k_means(data, 2*M, 0.001, A)
    if A.shape[0] == k:
        return A
    else:
        M = 2*M
        return lbg(data, k, eps, A, M)


def k_means_pp(data, k):
    """
    input
    data : ndarray
           target of clustering data
    k    : int
           number of clusters
    output
    A    : ndarray
           initial codebook
    """

    n = len(data)
    first_idx = np.random.choice(n, 1, replace=False)
    A = data[first_idx]

    # calc probability
    P = ((A - data) ** 2).sum(axis=1) / ((A - data) ** 2).sum()
    next_idx = np.random.choice(n, 1, replace=False, p=P)
    A = np.r_[A, data[next_idx]]

    if k > 2:
        for _ in range(k-2):
            dist_list = ((data[:, :, np.newaxis] - A.T[np.newaxis, :, :])**2).sum(axis=1)
            near_point = dist_list.argmin(axis=1)
            for i in range(dist_list.shape[1]):
                dist_list.T[i][near_point != i] = 0

            P = dist_list.sum(axis=1) / dist_list.sum()
            next_idx = np.random.choice(n, 1, replace=False, p=P)
            A = np.r_[A, data[next_idx]]

    return A


def minimax(data, k):
    """
    input
    data : ndarray
           target of clustering data
    k    : int
           number of clusters
    output
    A    : ndarray
           initial codebook
    """

    n = len(data)
    first_idx = np.random.choice(n, 1, replace=False)
    A = data[first_idx]
    next_idx = ((A - data) ** 2).sum(axis=1).argmax()
    A = np.r_[A, [data[next_idx]]]

    if k > 2:
        for _ in range(k-2):
            dist_list = ((data[:, :, np.newaxis] - A.T[np.newaxis, :, :])**2).sum(axis=1)
            near_point = dist_list.argmin(axis=1)
            for i in range(dist_list.shape[1]):
                dist_list.T[i][near_point != i] = 0

            next_idx = dist_list.sum(axis=1).argmax()
            A = np.r_[A, [data[next_idx]]]

    return A


def random_choice(data, k):
    """
    input
    data : ndarray
           target of clustering data
    k    : int
           number of clusters
    output
    A    : ndarray
           initial codebook
    """

    n = len(data)
    first_idx = np.random.choice(n, k, replace=False)
    A = data[first_idx]
    return A


def main():
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input data file path")
    parser.add_argument("--k", type=int, required=True, help="""division number of clustering:
                                                                must be power of 2 if you use LBG""")
    parser.add_argument("--eps", type=float, default=0.001, help="threshold of k-means")
    parser.add_argument("--method", type=int, default=0, help="""inintial value determination method: 
                                                                 0 -> random choice
                                                                 1 -> k-means++
                                                                 2 -> LBG
                                                                 3 -> minimax""")
    args = parser.parse_args()

    # check output dir exist
    if not os.path.exists('./out'):
        os.makedirs('./out')

    # loading data
    data = pd.read_csv(args.input).to_numpy()
    data_name = args.input.split('/')[1]

    # get initial codebook
    if args.method == 0:
        A = random_choice(data, args.k)
    elif args.method == 1:
        A = k_means_pp(data, args.k)
    elif args.method == 2:
        A = lbg(data, args.k, args.eps, center(data))
    else:
        A = minimax(data, args.k)
    A, S = k_means(data, args.k, args.eps, A)

    # plot 2D
    if data.shape[1] == 2:
        fig, ax = plt.subplots()
        for s in S:
            ax.scatter(s[:, 0], s[:, 1])
        ax.scatter(A[:, 0], A[:, 1], color='black', s=80, label='Codebook')
        ax.set(xlabel='X', ylabel='Y')
        plt.legend()
        plt.show(block=True)
        fig.savefig(f'./out/clustering_{data_name}_{args.method}_{args.k}.png')

    # plot 3D animation
    else:
        fig = plt.figure()
        ax = Axes3D(fig)

        for s in S:
            ax.scatter(s[:, 0], s[:, 1], s[:, 2])
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], color='black', s=80, label='Codebook')
        ax.set(xlabel='X', ylabel='Y', zlabel='Z')
        plt.legend()
        
        def rotate(angle):
            ax.view_init(azim=angle)
        anim = animation.FuncAnimation(
                fig, rotate, frames=180, interval=50)
        anim.save(f'./out/clustering_data3_{args.method}_{args.k}.gif', dpi=80)


if __name__ == '__main__':
    main()
