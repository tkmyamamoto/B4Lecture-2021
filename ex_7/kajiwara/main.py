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
    x: (dim)
    mu: (dim)
    sigma: (dim, dim)
    """

    dim = len(x)

    sigma_inv = np.linalg.inv(sigma)
    exp = (x - mu).T @ sigma_inv @ (x - mu)
    exp /= 2

    denomination = (2*np.pi)**dim * np.linalg.det(sigma)

    return np.exp(-exp) / np.sqrt(denomination)


def calc_gaussian(X, mu, sigma):
    """
    X: (n, dim)
    mu: (dim)
    sigma: (dim, dim)
    """

    n = len(X)
    gaussians = np.zeros(n)
    for i, x in enumerate(X):
        gaussians[i] = gaussian(x, mu, sigma)

    return gaussians


def mixture_gaussian(X, Pi, Mu, Sigma):
    """
    X: (n, dim)
    pi: (class num, dim)
    mu: (class num, dim)
    sigma: (class num, dim, dim)
    """

    n = len(X)
    k = len(Pi)
    m_gaussian = np.zeros((k, n))
    for j in range(k):
        m_gaussian[j] = Pi[j]*calc_gaussian(X, Mu[j], Sigma[j])

    return m_gaussian


def log_likelihood(X, Pi, Mu, Sigma):
    n = len(X)

    m_gaussian = mixture_gaussian(X, Pi, Mu, Sigma)
    sum_g = np.sum(m_gaussian, axis=0)

    l = 0
    for i in range(n):
        l += np.log(sum_g[i])

    return l


def em_algorithm(X, pi, mu, sigma, epsilon):
    ite = 0
    likelihoods = []
    l = log_likelihood(X, pi, mu, sigma)

    likelihoods.append(l)
    while True:
        # E step
        N = len(X)

        m_gaussian = mixture_gaussian(X, pi, mu, sigma)
        burden_rate = m_gaussian / np.sum(m_gaussian, axis=0)[None:, ]

        # M step
        N = len(X)
        K = len(pi)
        N_k = np.sum(burden_rate, axis=1)[:, None]

        pi = N_k/N

        for k in range(K):
            sigma[k] = 0
            for n in range(N):
                sigma[k] += burden_rate[k][n]*(X[n] - mu[k])[:, None]@((X[n] - mu[k]))[None, :]
        sigma = sigma / N_k[:, None]

        mu = (burden_rate @ X) / N_k

        l_new = log_likelihood(X, pi, mu, sigma)
        gap = l_new-likelihoods[ite]

        print(f'iteration: {ite},  likelihood: {l_new},  gap: {gap}')

        if gap < epsilon:
            print(f'finished  iteration: {ite},  likelihood: {l_new}')
            break

        ite += 1
        likelihoods.append(l_new)

    return ite, likelihoods, mu, sigma, pi


def setInitial(X, k):
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
    mu, sigma, pi = setInitial(data1, k_1)
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
    mu, sigma, pi = setInitial(data2, k_2)
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
    mu, sigma, pi = setInitial(data3, k_3)
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
