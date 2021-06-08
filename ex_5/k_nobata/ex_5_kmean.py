#!/usr/bin/env python
# coding: utf-8

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 

#ファイル読み込み
def file(num):
    """
    paramerter
    ---
    num:int
        file number
    
    return
    ---
    data:numpy.ndarray
        csv data
    """
    f_name = "/Users/nobatakoki/B4輪行/B4Lecture-2021/ex_5/data"+str(num)+".csv"
    df = pd.read_csv(f_name)
    data = df.values
    return data

#2次元散布図
def scatter_2d(data, save, K, clu):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    save:str
         save file name
    K:int
      the number of cluster
    clu:numpy.ndarray
        cluster
    """
    fig = plt.figure(figsize=(8,6))
    
    cmap = plt.get_cmap("tab10")
    for i in range(K):
        cdata = data[clu==i]
        x = cdata[:,0]
        y = cdata[:,1]
        #計算値とデータをプロット
        plt.scatter(x, y,color=cmap(i))
    
    #ラベル作成
    plt.xlabel("x",fontsize=18)
    plt.ylabel("y",fontsize=18)
    
    plt.grid() 
    #plt.legend() #凡例の追加
    plt.tight_layout()
    plt.savefig(save)
    plt.show()

#3次元散布図
def scatter_3d(data, save, K, clu):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    save:str
         save file name
    K:int
      the number of cluster
    clu:numpy.ndarray
        cluster
    """
    fig = plt.figure(figsize=(32,24))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    cmap = plt.get_cmap("tab10")

    #クラスタごとで描画
    for i in range(K):
        cdata = data[clu==i]
        x = cdata[:,0]
        y = cdata[:,1]
        z = cdata[:,2]
        
        ax1.plot(x, y, z, marker="o", linestyle='None', color=cmap(i))
        ax2.plot(x, y, z, marker="o", linestyle='None', color=cmap(i))
        ax3.plot(x, y, z, marker="o", linestyle='None', color=cmap(i))

    
    #ラベル作成
    ax1.set_xlabel("x",fontsize=24)
    ax1.set_ylabel("y",fontsize=24)
    ax1.set_zlabel("z",fontsize=24)
    ax2.set_xlabel("x",fontsize=24)
    ax2.set_ylabel("y",fontsize=24)
    ax2.set_zlabel("z",fontsize=24)
    ax3.set_xlabel("x",fontsize=24)
    ax3.set_ylabel("y",fontsize=24)
    ax3.set_zlabel("z",fontsize=24)

    plt.grid()
    #plt.legend()  #凡例の追加
    ax2.view_init(elev=0, azim=-90) #回転
    ax3.view_init(elev=0, azim=0)
    plt.tight_layout()
    plt.savefig(save)
    plt.show()

#ミニマックス法
def minimax(data, K):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster
      
    return
    ----
    cen:numpy.ndarray
         center of cluster
    """
    num, dim = data.shape
    cidx = [] #中心のインデックス
    cidx = np.append(cidx, random.randint(0, num-1))
    dis = np.zeros((K, num))
    cen = np.zeros((K, dim))
    for k in range(K):
        cen[k] = data[int(cidx[k])]
        r = np.sum((data - data[int(cidx[k])])**2, axis = 1) #距離計算
        dis[k] =  r #距離保存
        
        cidx = np.append(cidx, np.argmax(np.min(dis[:k+1], axis = 0)))  #距離最大の次の中心
    #clu = np.argmin(dis, axis=0)
    return cen

#kmeans++
def kplus(data, K):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster
      
    return
    ----
    cen:numpy.ndarray
         center of cluster
    """
    num, dim = data.shape
    cidx = [] #中心のインデックス
    cidx = np.append(cidx, random.randint(0, num-1))
    dis = np.zeros((K, num))
    cen = np.zeros((K, dim))
    pr = np.zeros(num)
    for k in range(K):
        #距離計算
        cen[k] = data[int(cidx[k])]
        r = np.sum((data - data[int(cidx[k])])**2, axis = 1) #距離計算
        dis[k] =  r #距離保存
        
        #確率作成
        pr = np.min(dis[:k+1], axis = 0)
        pr = pr / np.sum(pr)
        
        #次の中心
        x = np.random.choice(np.arange(num), 1, p=pr)
        cidx = np.append(cidx, x)

    #clu = np.argmin(dis, axis=0)
    return cen

#LGB法
def LGB(data, K, cen):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster
    cen:numpy.ndarray
        center of cluster
    
    return
    ----
    newcen:numpy.ndarray
           new center of cluster
    """
    #クラスタの中心をふたつに分ける
    delta = 0.01
    #cenn = np.zeros(())
    cen_b = cen-delta
    cen_a = cen+delta
    newcen = np.concatenate((cen_b, cen_a))
    M = newcen.shape[0]
    #kmeansアルゴリズム
    newcen, clu = kmean(data, M, newcen)
    if newcen.shape[0] >= K:
        newcen = newcen[random.sample(range(len(newcen)), K)]
        return newcen
    else:
        return LGB(data, K, newcen)
        
#初期値決定方法
def method(data, K, mname):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster
    mname:str
      method name
      
    return
    ----
    cen:numpy.ndarray
         center of cluster
    
    """
    if mname == "minimax":
        cen = minimax(data, K)
    elif mname == "kplus":
        cen = kplus(data, K)
    elif mname == "LGB":
        ce = np.zeros((1,2))
        ce[0] = np.mean(data, axis=0)
        cen = LGB(data, K, ce)
    else:
        print("error:method name")
        sys.exit(1)
    return cen

#kmeanアルゴリズム
def kmean(data, K, cen):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster
    cen:numpy.ndarray
         center of cluster
    clu:numpy.ndarray
        cluster
      
    return
    ----
    newcen:numpy.ndarray
           new center of cluster
    clu:numpy.ndarray
        cluster
    """
    num, dim = data.shape
    dis = np.zeros((K, num))
    newcen = np.zeros((K, dim))
    while (True):
        for k in range(0, K):
            r = np.sum((data - cen[k])**2, axis = 1) #距離計算
            dis[k] = r #距離保存
            
        clu = np.argmin(dis, axis=0)
        
        for i in range(0, K):            
            newcen[i] = data[clu==i].mean(axis=0)

        if np.allclose(cen,newcen) is True:
            break
        cen = newcen
    return newcen, clu

def main(args):
    n = args.fnum
    c = args.clo
    save = args.save
    methodname = args.method

    data = file(n)
    
    #クラスタ計算&描画
    cen = method(data, c, methodname)
    ncen, clu = kmean(data, c, cen)
    if data.shape[1] == 2:
        scatter_2d(data, save, c, clu)
    
    elif data.shape[1] == 3:
        scatter_3d(data, save, c, clu)

    else:
        print("error:over dimension")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnum", default="1")
    parser.add_argument("--clo", default=4)
    parser.add_argument("--save", default="data1.png")
    parser.add_argument("--method", default="minimax")
    args = parser.parse_args()

    main(args)