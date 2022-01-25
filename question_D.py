# Grid search each l vs l*
from datetime import datetime
import pickle
from functions import log_probit_likelihood, predict_t, pcn, probit, GaussianKernel
from utils import initialise_chain, initialise_simulation_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing
import seaborn as sns
import sys

import warnings
warnings.filterwarnings("ignore")

sns.set_style('darkgrid')


def repeat_for_seeds(_seeds, *args):
    return [[arg for _ in _seeds] for arg in args]


def worker(information):

    test_K, test_Kc, N, t, G, n_iters, beta, seed_num = information
    
    np.random.seed(seed_num)
    z = np.random.randn(N)
    u0 = test_Kc @ z
    
    X, acc_rate = pcn(log_probit_likelihood, u0 = u0, data = t, K = test_K, G = G, n_iters = n_iters, beta = beta)
    return X, acc_rate


if __name__ == '__main__':

    if sys.argv[1] == 'generate':

        D, subsample_factor, beta, n_iters = 16, 4, 0.2, 10000
        test_ells = np.logspace(-2, 1, 100)
        true_ells = [0.1, 0.2, 0.3, 0.4]
        seeds = range(1)

        results = []

        for true_ell in true_ells:

            xs, indices, x1s, x2s, i1s, i2s, true_K, true_u, subsample_indices, N, M, G, v = \
                initialise_simulation_dataset(D, true_ell, subsample_factor)

            t = probit(v)
            true_t = probit(true_u)

            accept_rates_mean, accuracies_mean, accept_rates_std, accuracies_std = \
                [np.array([]) for _ in range(4)]

            for test_ell in tqdm(test_ells):

                test_K = GaussianKernel(xs, test_ell)
                test_Kc = np.linalg.cholesky(test_K + 1e-6 * np.eye(N))

                pool = multiprocessing.Pool(processes = 5)
                information = list(zip(*repeat_for_seeds(seeds, *[test_K, test_Kc, N, t, G, n_iters, beta]), seeds))
                X_and_accs = pool.map(worker, information)

                round_accept_rates, round_accuracies = np.array([]), np.array([])

                for X, acc_rate in X_and_accs:

                    preds = predict_t(X)
                    binary_preds = (preds > 0.5).astype(int)

                    accuracy = (binary_preds == true_t).sum() / binary_preds.size

                    round_accept_rates = np.append(round_accept_rates, acc_rate)
                    round_accuracies = np.append(round_accuracies, accuracy)

                # accept_rates_mean = np.append(accept_rates_mean, round_accept_rates.mean())
                accuracies_mean = np.append(accuracies_mean, round_accuracies.mean())
                # accept_rates_std = np.append(accept_rates_std, round_accept_rates.std())
                accuracies_std = np.append(accuracies_std, round_accuracies.std())

            # plt.plot(test_ells, accept_rates, label = 'Accept rates')
            results.append(
                {'test_ells': test_ells, 'accuracies_mean': accuracies_mean, 'accuracies_std': accuracies_std, 'true_ell': true_ell})

        
        plt.show()
        # plt.savefig(f'TEST-question_D_{int(sys.argv[1])}.png')

        with open(f'qD_results_{datetime.now().__str__()}.pkl', 'wb') as handle:
            pickle.dump(results, handle, pickle.HIGHEST_PROTOCOL)

    elif sys.argv[1] == 'visualise':

        fname = input('results file name please: ')
        if len(str(fname)) == 0:
            fname = 'qD_results_2022-01-07 17:18:03.451474.pkl'
        
        with open(str(fname), 'rb') as f:
            results = pickle.load(f)

        fig, axs = plt.subplots(1)
        axs.set_xscale('log')

        for col, result in zip(['r', 'b', 'g'], results):
            
            test_ells = result['test_ells']
            accuracies_mean = result['accuracies_mean']
            accuracies_std = result['accuracies_std']
            true_ell = result['true_ell']

            if true_ell == 0.4:
                continue

            import pdb; pdb.set_trace()

            axs.plot(test_ells, accuracies_mean, label = f'$\ell = {true_ell}$', color = col)
            axs.plot([true_ell, true_ell], [0.45, 0.85], linestyle='dashed', color = col)
            #axs.fill_between(test_ells,  accuracies_mean-accuracies_std, accuracies_mean+accuracies_std, alpha = 0.3)

        axs.set_ylim([0.45, 0.85])
        axs.set_xlabel('$\log\ell^*$'); axs.set_ylabel('Accuracy'); plt.legend()
        plt.show()
