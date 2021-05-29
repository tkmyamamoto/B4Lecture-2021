import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import argparse
import csv


def csv_open(fname):
    """
    Function for opening csv file

    Parameters :
    fname   : Name of the file

    Return :
    file    : Data array in csv file
    """
    file = pd.read_csv(fname)
    file = np.array(file)
    return file

def rotate(angle):
    ax.view_init(azim=angle)

def reg_2d(data, order, nc):
    """
    Finding regression assumption for 2D data and plot.

    Parameters :
    data    : Original data(2D array)
    order   : Order for regression function
    nc      : Normalization coefficient 
    """

    x, y = data.T
    phi = np.zeros([len(x), order+1])
    for i in range(order+1):
        phi[:, i] += x ** i

    I = np.eye(order + 1)
    w = np.dot(np.dot(np.linalg.inv(I * nc + np.dot(phi.T, phi)), phi.T), y)

    reg_x = np.arange(min(x), max(x), 0.0001)
    reg_y = np.zeros(len(reg_x))

    for i in range(len(reg_x)):
        for j in range(len(w)):
            reg_y[i] += (reg_x[i] ** j) * w[j]

    # Plot original data and regression assumption
    plt.scatter(x, y, label='Original data')
    plt.plot(reg_x, reg_y, label='Regression assumption', color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("result_2d.png")


def reg_3d(data, order_x, order_y, nc):
    """
    Finding regression assumption for 3D data and plot.

    Parameters :
    data    : Original data(2D array)
    order_x : Order for x in regression function
    order_y : Order for y in regression function
    nc      : Normalization coefficient 
    """

    x, y, z = data.T
    phi = np.zeros([len(x), order_x + order_y + 1])
    for i in range(order_x+1):
        phi[:, i] += x ** i
    for j in range(order_y):
        phi[:, j + order_x + 1] += y ** (j + 1)

    I = np.eye(order_x + order_y + 1)
    w = np.dot(np.dot(np.linalg.inv(I * nc + np.dot(phi.T, phi)), phi.T), z)

    reg_x = np.arange(min(x), max(x), 0.01)
    reg_y = np.arange(min(y), max(y), (max(y)-min(y)) /
                      ((max(x)-min(x)) / 0.01))
    reg_x, reg_y = np.meshgrid(reg_x, reg_y)
    reg_z = np.zeros([len(reg_y), len(reg_x)])
    for i in range(0, order_x + 1):
        reg_z += (reg_x ** i) * w[i]
    for j in range(1, order_y + 1):
        reg_z += (reg_y ** j) * w[j + order_x]

    # Plot original data and regression assumption
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.set_xlabel('z')
    ax.view_init(20, 210)
    ax.scatter(x, y, z, label='Original data', color='r')
    plt.legend()
    ax.plot_wireframe(reg_x, reg_y, reg_z, color='g',
                      label='Regression assumption', linewidth=0.2)
    ax.legend()
    plt.savefig("result_3d.png")


def main():
    parser = argparse.ArgumentParser(
        description='Program for plotting regression equation.\nFile name and order for regression equation are required.')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)
    parser.add_argument("-ox", dest="order_x", type=int,
                        help='Regression order for x', required=True)
    parser.add_argument("-oy", dest="order_y", type=int,
                        help='Regression order for y (optional). Default = 3', required=False, default=3)
    parser.add_argument("-nc", dest="nc", type=int,
                        help='Normalization coefficient (optional). Default = 0', required=False, default=0)
    args = parser.parse_args()
    data = csv_open(args.filename)
    dimension = np.shape(data)[1]
    if dimension == 3:
        reg_3d(data, args.order_x, args.order_y, args.nc)
    else:
        reg_2d(data, args.order_x, args.nc)


if __name__ == "__main__":
    main()
