from functions import (
    GaussianKernel, log_continuous_likelihood, log_continuous_target, grw, pcn, plot_2D, plot_result
)
from utils import calculate_KL, initialise_simulation_dataset, initialise_chain
from tqdm import tqdm
import sys, pickle, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime




if __name__ == '__main__':

    if sys.argv[1] == 'test':

        # Initial test for acc rate and convergence
        _, _, x1s, x2s, _, _, K, u, subsample_indices, N, _, G, v = \
            initialise_simulation_dataset(D = 16, ell = 0.3, subsample_factor = 4)

        u0 = initialise_chain(N, K)

        X_grw, acceptance_rate_grw = grw(log_target = log_continuous_target, u0 = u0, data = v, K = K, G = G, n_iters = 10000, beta = 0.2)
        X_pcn, acceptance_rate_pcn = pcn(log_likelihood = log_continuous_likelihood, u0 = u0, data = v, K = K, G = G, n_iters = 10000, beta = 0.2)

        print(acceptance_rate_grw, acceptance_rate_pcn)
        import pdb; pdb.set_trace()

        fig = plt.figure(figsize=(15, 5))
        ax1, ax2, ax3 = [fig.add_subplot(1, 3, i, projection='3d') for i in range(1, 4)]

        plot_result(title = "Original data", u = u, data = v, x = x1s, y = x2s, x_d = x1s[subsample_indices],y_d = x2s[subsample_indices], figure = (fig, ax1))
        plot_result(title = "GRW samples mean", u = np.mean(X_grw, axis = 0), data = v, x = x1s, y = x2s, x_d = x1s[subsample_indices],y_d = x2s[subsample_indices],figure = (fig, ax2))
        plot_result(title = "pCN samples mean", u = np.mean(X_pcn, axis = 0), data = v, x = x1s, y = x2s, x_d = x1s[subsample_indices], y_d = x2s[subsample_indices], figure = (fig, ax3))

        plt.show()


    elif sys.argv[1] == 'tabulation_results':

        # Grid search or part b report
        algorithms_dict = {'grw': log_continuous_target, 'pcn': log_continuous_likelihood}
        Ds = [4, 16]
        ells = [0.1, 0.3]
        betas = [0.01, 0.02, 0.1, 0.2, 0.5, 1.0]
        subsample_factor = 4

        n_iters, grid_args, results, data_dict = 10000, [], [], {}
        partb_results = pd.DataFrame(columns=['Algorithm', '$D$', '$\ell$', '$\\beta$', 'Acceptance rate'])

        for D, ell in list(itertools.product(Ds, ells)):
            data_dict[(D, ell)] = initialise_simulation_dataset(D, ell, subsample_factor)
            
            for beta, algorithm in list(itertools.product(betas, algorithms_dict.keys())):
                grid_args.append({'Algorithm': algorithm, '$D$': D, '$\ell$': ell, '$\\beta$': beta})

        for ga in tqdm(grid_args):

            _, _, x1s, x2s, _, _, K, u, subsample_indices, N, M, G, v = \
                data_dict[(ga['$D$'], ga['$\ell$'])]
            
            algorithm, beta = ga['Algorithm'], ga['$\\beta$']
            alg_func, log_method, acc_rates, Xs = eval(algorithm), algorithms_dict[algorithm], [], []

            for _ in range(3):
                u0 = initialise_chain(N, K)
                X, acc_rate = alg_func(log_method, u0 = u0, data = v, K = K, G = G, n_iters = n_iters, beta = beta)
                
                acc_rates.append(acc_rate)
                Xs.append(X)
                # Do error in mean here maybe

            acc_rate_string = str(np.mean(acc_rates).round(2)) + '$\pm$' + str(np.std(acc_rates).round(2))
            partb_results = partb_results.append({**ga, 'Acceptance_rate': acc_rate_string}, ignore_index=True)
            results.append({'config':ga, 'acc_rates': acc_rates.copy(), 'u_true': u, 'Xs': Xs.copy(), 'G': G, 'K': K})  # 'convergence': calculate_W2(X, K, G, v)

        fdate = datetime.now().__str__()

        with open(f'FINAL_qB_results_{fdate}.pkl', 'wb') as handle:
            pickle.dump(results, handle, pickle.HIGHEST_PROTOCOL)

        # partb_results.to_latex(f'FINAL_qB_results_{fdate}.txt', caption='All acceptance rates found')


    elif sys.argv[1] == 'error_field_results':
        # FINAL_qB_results_2022-01-07 14:19:29.086094.pkl

        with open(" ".join(sys.argv[2:]), 'rb') as handle:
            results_dicts = pickle.load(handle)

        D = 16
            
        filter_func = lambda x: (
            x['config']['$D$'] == D and 
            x['config']['$\\ell$'] == 0.1 and
            x['config']['$\\beta$'] == 0.2
        )
        fits = list(filter(filter_func, results_dicts))

        mh_example = list(filter(lambda x: x['config']['Algorithm'] == 'grw', fits))
        pcn_example = list(filter(lambda x: x['config']['Algorithm'] == 'pcn', fits))
        u = mh_example['u_true']
        K = mh_example['K']
        G = mh_example['G']

        N = D**2 # Number of data points
        xs = np.array([[x1, x2] for x1 in np.linspace(0, 1, D) for x2 in np.linspace(0, 1, D)])
        indices = np.array([[i1, i2] for i1 in np.arange(D) for i2 in np.arange(D)])
        x1s, x2s = xs[:,0], xs[:,1]
        i1s, i2s = indices[:,0], indices[:,1]

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        plot_2D(u, i1s, i2s, title='True $u$', figure=(fig, axs[0]))     # Plot true class assignments
        plot_2D(v, G @ i1s, G @ i2s, title='Original data', figure=(fig, axs[1]))     # Plot data
        plot_2D(preds, i1s, i2s, title = '$p(t=1)$', figure=(fig, axs[2]))
        plot_2D(preds>0.5, i1s, i2s, title = '$p(t=1) > 0.5$', figure=(fig, axs[3]))

    

    elif sys.argv[1] == 'convergence_results':

        with open(" ".join(sys.argv[2:]), 'rb') as handle:
            results_dicts = pickle.load(handle)

        example = results_dicts[0]

        _, _, x1s, x2s, _, _, K, u, subsample_indices, N, _, G, v = \
            initialise_simulation_dataset(D = 16, ell = 0.3, subsample_factor = 4)
        calculate_KL(samples=example['Xs'], K=K, G=G, v=v, step = 100)

