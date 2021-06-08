import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sk_PCA

from pca import PCA as my_PCA

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def calc_process_time(args, func):
    start = time.perf_counter()
    func(**args)
    end = time.perf_counter()

    return end - start


def main():
    data_size_fix_dim = [
        (100, 2),
        (500, 2),
        (1000, 2),
        (5000, 2),
        (10000, 2),
        (15000, 2),
        (50000, 2),
        (100000, 2),
        (150000, 2),
        (1000000, 2),
    ]
    data_size_fix_num = [
        (10000, 5),
        (10000, 10),
        (10000, 30),
        (10000, 50),
        (10000, 100),
        (10000, 150),
        (10000, 300),
        (10000, 500),
        # (10000, 1000),
        # (10000, 1500),
    ]

    """fix dim"""
    result = []
    for data_size in data_size_fix_dim:
        print(data_size)
        fake_data = np.random.rand(*data_size)

        res = {
            'sk_pca': [],
            'my_pca': []
        }

        for _ in range(10):
            sk_pca = sk_PCA(n_components=fake_data.shape[1])
            sk_res = calc_process_time(
                {'X': fake_data}, sk_pca.fit
            )
            res['sk_pca'].append(sk_res)

            my_pca = my_PCA(dim=data_size[1])
            mine_res = calc_process_time({'X': fake_data}, my_pca.fit)
            res['my_pca'].append(mine_res)

        result.append(res)

    aves = {
        'sk_pca': [],
        'my_pca': []
    }
    for _, res in enumerate(result):
        print('-'*10)
        for k, v in res.items():
            print(k, sum(v)/len(v))
            aves[k].append(sum(v)/len(v))
    print('-'*10)

    x_ax = np.array(data_size_fix_dim)[:, 0]
    plt.plot(x_ax, aves['sk_pca'], marker="o", label='sklearn')
    plt.plot(x_ax, aves['my_pca'], marker="o", label='mine')
    plt.legend(loc=2)
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    plt.savefig('./result/bench-res-fix-dim{}.png'.format(timestamp))
    plt.clf()
    plt.close()

    """fix data num"""
    result = []
    for data_size in data_size_fix_num:
        print(data_size)
        fake_data = np.random.rand(*data_size)

        res = {
            'sk_pca': [],
            'my_pca': []
        }

        for _ in range(10):
            sk_pca = sk_PCA(n_components=fake_data.shape[1])
            sk_res = calc_process_time(
                {'X': fake_data}, sk_pca.fit
            )
            res['sk_pca'].append(sk_res)

            my_pca = my_PCA(dim=data_size[1])
            mine_res = calc_process_time({'X': fake_data}, my_pca.fit)
            res['my_pca'].append(mine_res)

        result.append(res)

    aves = {
        'sk_pca': [],
        'my_pca': []
    }
    for _, res in enumerate(result):
        print('-'*10)
        for k, v in res.items():
            print(k, sum(v)/len(v))
            aves[k].append(sum(v)/len(v))
    print('-'*10)

    x_ax = np.array(data_size_fix_num)[:, 1]
    plt.plot(x_ax, aves['sk_pca'], marker="o", label='sklearn')
    plt.plot(x_ax, aves['my_pca'], marker="o", label='mine')
    plt.legend(loc=2)
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    plt.savefig('./result/bench-res-fix-num{}.png'.format(timestamp))
    plt.clf()
    plt.close()


if __name__ == '__main__':
    main()
