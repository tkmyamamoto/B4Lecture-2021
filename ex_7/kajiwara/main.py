import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def gaussian(x, mu, sigma):
    """
    calculate multi dimention gauss distribution

    Parameters
    ----------
    x: ndarray (dim, )
        one sample of data
    mu: ndarray (dim, )
        means of distribution
    sigma: ndarray (dim, )
        variances of distribution

    Returns
    -------
    gaussian: ndarray (dim, )
        gauss distribution
    """

    dim = len(x)

    sigma_inv = np.linalg.inv(sigma)
    exp = (x - mu).T @ sigma_inv @ (x - mu)
    exp /= 2

    denomination = (2*np.pi)**dim * np.linalg.det(sigma)

    gaussian = np.exp(-exp) / np.sqrt(denomination)

    return gaussian


def calc_gaussian(X, mu, sigma):
    """
    caluculate gaussian of each sample of input data

    Parameters
    ----------
    X: ndarray (n, dim)
        input data
    mu: ndarray (dim, )
        means of distribution
    sigma: ndarray (dim, dim)
        variances of distribution (dim, )

    Returns
    -------
    gaussians: ndarray (n, )
        gaussian list of each sample in input
    """

    n = len(X)
    gaussians = np.zeros(n)
    for i, x in enumerate(X):
        gaussians[i] = gaussian(x, mu, sigma)

    return gaussians


def mixture_gaussian(X, Pi, Mu, Sigma):
    """
    caluculate mixture gauss distribution

    Parameters
    ----------
    X: ndarray (n, dim)
        input data
    Pi: (class, dim)
        mixing coefficient of each class
    Mu: ndarray (class num, dim)
        means of each class
    Sigma: ndarray (class, dim, dim)
        variances of each class

    Returns
    -------
    m_gaussian: ndarray (class, n)
        mixture gauss distribution
    """

    n = len(X)
    k = len(Pi)
    m_gaussian = np.zeros((k, n))
    for j in range(k):
        m_gaussian[j] = Pi[j]*calc_gaussian(X, Mu[j], Sigma[j])

    return m_gaussian


def log_likelihood(X, Pi, Mu, Sigma):
    """
    calc log likelihood

    Parameters
    ----------
    X: ndarray (n, dim)
        input data
    Pi: (class, dim)
        mixing coefficient of each class
    Mu: ndarray (class num, dim)
        means of each class
    Sigma: ndarray (class, dim, dim)
        variances of each class

    Returns
    -------
    log_l: float
        log likelihood
    """

    n = len(X)

    m_gaussian = mixture_gaussian(X, Pi, Mu, Sigma)
    sum_g = np.sum(m_gaussian, axis=0)

    log_l = 0
    for i in range(n):
        log_l += np.log(sum_g[i])

    return log_l


def em_algorithm(X, Pi, Mu, Sigma, epsilon):
    """
    run EM algorithm

    Parameters
    ----------
    X: ndarray (n, dim)
        input data
    Pi: (class, dim)
        mixing coefficient of each class
    Mu: ndarray (class num, dim)
        means of each class
    Sigma: ndarray (class, dim, dim)
        variances of each class

    Returns
    -------
    ite: int 
        count of iteration
    likelihoods: list[float]
        likelihood list of each iteration
    Pi: (class, dim)
        mixing coefficient of each class
    Mu: ndarray (class num, dim)
        means of each class
    Sigma: ndarray (class, dim, dim)
        variances of each class
    """

    print('init parameters')
    print(f'means:\n {Mu}\n variances:\n {Sigma}\n mixture coef:\n {Pi}\n')

    ite = 0
    likelihoods = []
    log_l = log_likelihood(X, Pi, Mu, Sigma)

    likelihoods.append(log_l)
    while True:
        # E step
        N = len(X)

        m_gaussian = mixture_gaussian(X, Pi, Mu, Sigma)
        burden_rate = m_gaussian / np.sum(m_gaussian, axis=0)[None:, ]

        # M step
        N = len(X)
        K = len(Pi)
        N_k = np.sum(burden_rate, axis=1)[:, None]

        Pi = N_k/N

        for k in range(K):
            Sigma[k] = 0
            for n in range(N):
                Sigma[k] += burden_rate[k][n]*(X[n] - Mu[k])[:, None]@((X[n] - Mu[k]))[None, :]
        Sigma = Sigma / N_k[:, None]

        Mu = (burden_rate @ X) / N_k

        l_new = log_likelihood(X, Pi, Mu, Sigma)
        gap = l_new-likelihoods[ite]

        print(f'iteration: {ite},  likelihood: {l_new},  gap: {gap}')

        if gap < epsilon:
            print('='*10)
            print(f'finished  iteration: {ite},  likelihood: {l_new}')
            print(f'means:\n {Mu}\n variances:\n {Sigma}\n mixture coef:\n {Pi}\n')
            print('='*10)
            break

        ite += 1
        likelihoods.append(l_new)

    return ite, likelihoods, Mu, Sigma, Pi


def set_initial(X, k):
    """
    initialize parameters of em algorithm

    Parameters
    ----------
    X: ndarray (n, dim)
        input data
    k: int
        number of class

    Returns
    -------
    Pi: (class, dim)
        mixing coefficient of each class
    Mu: ndarray (class num, dim)
        means of each class
    Sigma: ndarray (class, dim, dim)
        variances of each class
    """

    dim = X.shape[1]
    mu = np.random.randn(k, dim)
    sigma = np.array([np.eye(dim) for _ in range(k)])
    pi = np.array([1/k for i in range(k)])

    return mu, sigma, pi


def main(args):
    if args.result_path:
        result_path = Path(args.result_path)
        timestamp = datetime.now().strftime(TIME_TEMPLATE)
        result_path = result_path/timestamp
        if not result_path.exists():
            result_path.mkdir(parents=True)

    # loading data
    df1 = pd.read_csv('../data1.csv', header=None)
    df2 = pd.read_csv('../data2.csv', header=None)
    df3 = pd.read_csv('../data3.csv', header=None)

    # df to nd ndarray
    data1 = df1.values
    data2 = df2.values
    data3 = df3.values

    """data1"""
    k_1 = 2
    mu, sigma, pi = set_initial(data1, k_1)
    epsilon = 0.000001
    ite, likelihoods, mu, sigma, pi = em_algorithm(data1, pi, mu, sigma, epsilon)

    x = np.linspace(np.min(data1), np.max(data1), 100)
    y = np.zeros(100)
    for i in range(k_1):
        y += pi[i] * st.multivariate_normal.pdf(x, mu[i], sigma[i])
    plt.scatter(data1, np.zeros_like(data1), s=2, label='original')
    plt.plot(x, y, label='GMM')
    plt.scatter(mu, np.zeros_like(mu), marker='x', label='centroid')
    plt.xlabel('x')
    plt.ylabel('probability density')
    plt.legend()
    plt.title('data1')
    plt.savefig(result_path / 'data1-1.png')
    plt.close()
    plt.clf()

    plt.plot([i for i in range(ite+1)], likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('log likelihood')
    plt.savefig(result_path / 'data1-2.png')
    plt.close()
    plt.clf()

    """data2"""
    k_2 = 3
    mu, sigma, pi = set_initial(data2, k_2)
    epsilon = 0.000001
    ite, likelihoods, mu, sigma, pi = em_algorithm(data2, pi, mu, sigma, epsilon)

    n = len(data2)
    x = np.linspace(-1, 4, n)
    y = np.linspace(-1, 4, n)
    ax_x, ax_y = np.meshgrid(x, y)
    pos = np.dstack((ax_x, ax_y))
    z = np.zeros((n, n))
    for i in range(k_2):
        z += pi[i] * st.multivariate_normal.pdf(pos, mu[i], sigma[i])
    plt.contour(ax_x, ax_y, z)
    plt.scatter(data2[:, 0], data2[:, 1], facecolor='None', edgecolors='r')
    plt.scatter(mu[:, 0], mu[:, 1], marker='x')
    plt.title('data2')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.savefig(result_path / 'data2-1.png')
    plt.close()
    plt.clf()

    plt.plot([i for i in range(ite+1)], likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('log likelihood')
    plt.savefig(result_path / 'data2-2.png')
    plt.close()
    plt.clf()

    """data3"""
    k_3 = 2
    mu, sigma, pi = set_initial(data3, k_3)
    epsilon = 0.000001
    ite, likelihoods, mu, sigma, pi = em_algorithm(data3, pi, mu, sigma, epsilon)

    n = len(data3)
    x = np.linspace(-1, 4, n)
    y = np.linspace(-1, 4, n)
    ax_x, ax_y = np.meshgrid(x, y)
    pos = np.dstack((ax_x, ax_y))
    z = np.zeros((n, n))
    for i in range(k_3):
        z += pi[i] * st.multivariate_normal.pdf(pos, mu[i], sigma[i])
    plt.contour(ax_x, ax_y, z)
    plt.scatter(data3[:, 0], data3[:, 1], facecolor='None', edgecolors='r')
    plt.scatter(mu[:, 0], mu[:, 1], marker='x')
    plt.title('data3')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.savefig(result_path / 'data3-1.png')
    plt.close()
    plt.clf()

    plt.plot([i for i in range(ite+1)], likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('log likelihood')
    plt.savefig(result_path / 'data3-2.png')
    plt.close()
    plt.clf()


if __name__ == "__main__":
    description = 'Example: python main.py ./result'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-rs', '--result-path', default='./result', help='path to save the result')

    args = parser.parse_args()

    main(args)
