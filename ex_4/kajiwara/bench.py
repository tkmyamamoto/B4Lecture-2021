import time

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from modules.lpc import toeplitz_solver
from modules.autocorrelation import autocorrelation

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def calc_process_time(args, func):
    start = time.perf_counter()
    func(**args)
    end = time.perf_counter()

    return end - start


def main():
    dims = [
        10, 50, 100, 150, 300, 500, 1000, 1500, 3000
    ]
    result = []

    for d in dims:
        print(d)
        fake_data = 2 * np.random.rand(22050*10) - 1
        ac = autocorrelation(fake_data)

        res = {
            'scipy_solver': [],
            'mine_solver': []
        }

        for i in range(10):
            sp_res = calc_process_time(
                {'c_or_cr': (ac[:d][:d-1], ac[:d][:d-1]), 'b': ac[:d][1:]},
                linalg.solve_toeplitz
            )
            res['scipy_solver'].append(sp_res)

            mine_res = calc_process_time({'ac': ac, 'dim': d}, toeplitz_solver)
            res['mine_solver'].append(mine_res)

        result.append(res)

    aves = {
        'scipy_solver': [],
        'mine_solver': []
    }
    for i, res in enumerate(result):
        print('-'*10)
        for k, v in res.items():
            print(k, sum(v)/len(v))
            aves[k].append(sum(v)/len(v))
    print('-'*10)

    plt.plot(dims, aves['scipy_solver'], marker="o", label='scipy')
    plt.plot(dims, aves['mine_solver'], marker="o", label='mine')
    plt.legend(loc=2)
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    plt.savefig('./result/bench-res-{}.png'.format(timestamp))


if __name__ == '__main__':
    main()
