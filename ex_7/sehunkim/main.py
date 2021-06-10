import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.stats import multivariate_normal
import argparse
import csv


def plot1d(data,K,pi,mu,sigma,likelihood):
    """
    Function for plotting 2D data
    --------------------
    Parameters  :
    data        : Data array to plot
    """
    x = np.linspace(np.min(data),np.max(data), 100)
    y = np.zeros(100)
    for i in range(K):
        y += pi[i] * multivariate_normal.pdf(x,mu[i],sigma[i])

    plt.scatter(data, np.zeros_like(data), s=2, label='original data')
    plt.plot(x,y, c='orange', label='GMM')
    plt.scatter(mu, np.zeros_like(mu), marker='x', c='r', label='centroids')

    plt.xlabel('x')
    plt.ylabel('probability density')
    plt.legend()
    plt.title('1D GMM (K = %d)'%(K))
    plt.savefig("result_1d.png")


def plot2d(data,K,pi,mu,sigma,likelihood):
    """
    Function for plotting 2D data
    --------------------
    Parameters  :
    data        : Data array to plot
    """
    data_x, data_y = data.T
    mu_x, mu_y = mu.T

    x = np.linspace(np.min(data_x),np.max(data_x),100)
    y = np.linspace(np.min(data_y),np.max(data_y),100)
    X, Y = np.meshgrid(x,y)
    XY = np.dstack([X, Y])
    Z = np.zeros((100,100))
    for i in range(K):
        Z += pi[i] * multivariate_normal.pdf(XY,mu[i],sigma[i])


    fig = plt.figure()
    ax = Axes3D(fig)
    def rotate(angle):
        ax.view_init(azim=angle)
    print(np.zeros_like(data).shape)
    ax.scatter3D(data_x, data_y, np.zeros_like(data_x), s=2,label='original data')
    ax.contour3D(X,Y,Z, 50)
    ax.scatter(mu_x,mu_y,np.zeros_like(mu_x), c='r', label = 'Calculated centroids')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('probability density')

    plt.legend()
    rot_animation = animation.FuncAnimation(fig, rotate, frames=100, interval=50)
    rot_animation.save('rotation3D.gif', dpi=80)


def gauss(x, mean, cov):
    dim = x.shape[0]
    cov_inv = np.linalg.inv(cov)
    coef = 1 / ((2*np.pi) ** (dim / 2) * np.sqrt(np.linalg.det(cov)))
    gaussian = coef * np.exp(-((x-mean).T @ cov_inv @ (x-mean))/2)
    #normal_distribution = np.exp(-1/2 * (x-mean).T @ np.linalg.inv(cov) @ (x-mean)) / np.sqrt(((2 * np.pi) ** len(x)) * np.linalg.det(cov))
    return gaussian


def em_algo(x, K, D, mu, sigma, pi):
    q = np.zeros([x.shape[0], K])
    # E step
    for j in range(K):
        dist = x - mu[j]
        inv_sigma = np.linalg.inv(sigma[j])
        mahalanobis = np.sum(dist @ inv_sigma * dist, axis =1)
        exp = np.exp(-1 * mahalanobis/2)
        det_sigma = np.linalg.det(sigma[j])
        q[:,j] = pi[j] * exp / np.sqrt(det_sigma)
    gamma = q / np.sum(q, axis= 1, keepdims = True)

    # M step
    mu = (gamma.T @ x) / np.sum(gamma, axis=0, keepdims= True).T
    for k in range(K):
        dist = x - mu[k]
        sigma[k] = ((np.tile(gamma[:, k], (x.shape[1], 1)) * dist.T) @ dist) / np.sum(gamma[:, k])
    pi = np.sum(gamma, axis = 0) / np.sum(gamma)

    return pi, mu, sigma, q


def gmm_classification(x, K, D):
    # Initialize each variable
    max_iter = 20
    N = len(x)
    mu = np.linspace(np.min(x),np.max(x),K)
    sigma = np.zeros((K,D,D))
    matrix = np.zeros((D,D))
    for i in range(K):
        for j in range(D):
            matrix[j][j] = np.random.uniform(1,2)
            matrix[D-j-1][j] = np.random.uniform(0,1)
        sigma[i] = matrix
    pi = [1/K] * K

    old_likelihood = 0
    new_likelihood = 1
    likelihood = []

    for i in range(max_iter):
        if abs(old_likelihood - new_likelihood) < 0.1:
            break
        old_likelihood = new_likelihood
        pi, mu, sigma, q = em_algo(x, K, D, mu, sigma, pi)
        new_likelihood = np.sum(np.log(np.sum(q/(2*np.pi)** (D/2), axis =1)))
        likelihood.append(new_likelihood)

    return pi, mu, sigma, likelihood

def main():
    parser = argparse.ArgumentParser(
        description='Program for fitting GMM model')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)
    parser.add_argument("-k", dest="K", type=int, help='Number of clusters', required=False, default=3)
    args = parser.parse_args()
    data = pd.read_csv(args.filename, header=None).values
    dim = np.shape(data)[1]
    K = args.K
    pi, mu, sigma, likelihood = gmm_classification(data,K, dim)

    if dim == 1:
        plot1d(data,K,pi,mu,sigma,likelihood)

    elif dim == 2:
        plot2d(data,K,pi,mu,sigma,likelihood)


if __name__ == "__main__":
    main()
