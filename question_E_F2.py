import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from functions import GaussianKernel, log_poisson_likelihood, pcn, plot_3D
from utils import initialise_spatial_dataset
import seaborn as sns
sns.set_style('darkgrid')


subsample_factor = 3
n_iters = 10000

df, xi, yi, xs, data, N, idx, G, c = initialise_spatial_dataset(subsample_factor)

test_ells = np.logspace(-2, 2, 100)
betas = [0.2]

seeds = range(1)

def repeat_for_seeds(_seeds, *args):
    return [[arg for _ in _seeds] for arg in args]

def worker(information):

    K, Kc, N, G, n_iters, beta, seed_num = information

    np.random.seed(seed_num)
    z = np.random.randn(N)
    u0 = Kc @ z

    u_samples, acc_rate = pcn(
        log_poisson_likelihood, 
        u0 = u0, 
        data = c, 
        K = K,
        G = G, 
        n_iters = n_iters,
        beta = beta
    )
    return u_samples

if __name__ == '__main__':

    fig, axs = plt.subplots(1)

    for beta in betas:

        error_means = np.array([])
        error_stds = np.array([])

        for ell in tqdm(test_ells):    

            K = GaussianKernel(xs, ell)
            Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

            pool = multiprocessing.Pool(processes = 5)
            information = list(zip(*repeat_for_seeds(seeds, *[K, Kc, N, G, n_iters, beta]), seeds))

            u_samples = pool.map(worker, information)
            theta_samples = [list(map(np.exp, u)) for u in u_samples]
            expected_counts = [np.mean(thetas, axis = 0) for thetas in theta_samples]

            new_errors = [abs(ec - data).mean() for ec in expected_counts]
            error_means = np.append(error_means, np.mean(new_errors))
            error_stds = np.append(error_stds, np.std(new_errors))

        axs.plot(test_ells, error_means, label = f'$\\beta = {beta}$')
        # plt.fill_between(test_ells, error_means + error_stds, error_means - error_stds, alpha = 0.3)

    axs.set_xscale('log')
    axs.set_xlabel('$\ell^*$')
    axs.set_ylabel('MAE') 
    # plt.legend()
    # plt.show()

    plt.savefig('TEST-question_E_F2.png')