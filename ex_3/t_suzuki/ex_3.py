import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def reg_2d(x, y, d, l):
    """
    input
    x : array_like, shape(M,)
        x-coordinates of the M sample points (x[i], y[i])
    y : array_like, shape(M,)
        y-coordinates of the sample points
    d : int
        regression line degrees
    l : regularization parameter

    output
    x_label : regression x label
    y_reg   : regression y value
    w       : regression factor
    """

    phi = np.array([x ** n for n in range(d + 1)]).T
    I = np.eye(d+1)
    w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)+l*I), phi.T), y)
    
    x_label = np.arange(x.min(), x.max(), 0.1)
    y_reg = np.array([np.dot(w, np.array([np.sum(x ** n) for n in range(d + 1)])) for x in x_label])

    return x_label, y_reg, w


def reg_3d(x, y, z, x_d, y_d, l):
    """
    input
    x : array_like
        coordinates of the sample points
    y : array_like
        coordinates of the sample points
    z : array_like
        coordinates of the sample points
    x_d : int
          regression line degrees of x
    y_d : int
          regression line degrees of y
    l : regularization parameter

    output
    w : regression factor
    """
    
    phi = np.zeros([len(x), x_d+y_d+1])
    for i in range(x_d+1):
        phi[:, i] = x ** i
    for j in range(y_d):
        phi[:, x_d+1+j] = y ** (j+1)
    I = np.eye(x_d+y_d+1) 
    w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)+l*I), phi.T), z)
    return w


def get_label_2d(w):
    """
    input  : regression factor
    output : regression label
    """
    label = 'y='
    d = len(w)

    for i, w_i in enumerate(w[::-1]):
        if i == 0:
            if i == d-2:
                label += f'{w_i:.2}x'
            else:
                label += f'{w_i:.2}$x^{{{d-1-i}}}$'
        elif i == d-2:
            label += f'{w_i:+.2}x'
        elif i == d-1:
            label += f'{w_i:+.2}'
        else:
            label += f'{w_i:+.2}$x^{{{d-1-i}}}$'

    return label


def get_label_3d(w, x_d, y_d):
    """
    input  : regression factor
    output : regression label
    """
    
    w = w.flatten()
    label = f'z={w[0]:.2}'
    for i in range(1, x_d+y_d+1):
        if i == 1:
            label += f'{w[i]:+.2}x'
        elif i <= x_d:
            label += f'{w[i]:+.2}$x^{{{i}}}$'
        elif i == x_d+1:
            label += f'{w[i]:+.2}y'
        else:
            label += f'{w[i]:+.2}$y^{{{i-x_d}}}$'

    return label


def main():
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="input file path")
    parser.add_argument("--reg_x_d", type=int, required=True, help="regression degrees of x")
    parser.add_argument("--reg_y_d", type=int, default=2, help="regression degrees of y")
    parser.add_argument("--reg_dim", type=int, required=True, help="regression dimension")
    parser.add_argument("--lam", type=float, default=0.5, help="regularization parameters")
    parser.add_argument("--output_name", type=str, default='result', help='output file name')
    args = parser.parse_args()

    if not os.path.exists('./out'):
        os.makedirs('./out')
    
    # load data
    data = pd.read_csv(args.input_path)

    if args.reg_dim == 2:
        # split data
        x, y = np.array(data.x1), np.array(data.x2)
        
        # regression
        x_reg, y_reg, w = reg_2d(x, y, args.reg_x_d, args.lam)
        reg_label = get_label_2d(w)

        # setting graphic space 
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
    
        # plot data and add label
        ax.scatter(x, y, facecolor='None', edgecolors='blue', label='Observed data')
        ax.plot(x_reg, y_reg, color='red', label=reg_label)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(color='gray', linestyle='--')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02,), borderaxespad=0)
    
        # show and save fig
        plt.show(block=True)
        fig.savefig(f'./out/{args.output_name}')

    else:
        # split data
        x, y, z = np.array(data.x1), np.array(data.x2), np.array(data.x3)

        # regression
        w = reg_3d(x, y, z, args.reg_x_d, args.reg_y_d, args.lam)
        reg_label = get_label_3d(w, args.reg_x_d, args.reg_y_d)

        # setting graphic
        fig = plt.figure()
        ax = Axes3D(fig)
        
        # plot scatter
        ax.scatter(x, y, z, label='Observed data', color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # prepare regression data
        x_reg = np.arange(x.min(), x.max(), 0.1)
        y_reg = np.arange(y.min(), y.max(), 0.1)
        mesh_x, mesh_y = np.meshgrid(x_reg, y_reg)
        z_reg = np.zeros([len(y_reg), len(x_reg)])
        for i in range(args.reg_x_d+1):
            z_reg += w[i] * (mesh_x**i)
        for j in range(args.reg_y_d):
            z_reg += w[j+args.reg_x_d+1] * (mesh_y**(j+1))
        
        # plot regression
        ax.plot_wireframe(mesh_x, mesh_y, z_reg, color='red', label=reg_label)
        ax.legend()

        # show and save fig
        plt.show(block=True)
        fig.savefig(f'./out/{args.output_name}')


if __name__ == "__main__":
    main()
