from functions import (
    log_continuous_likelihood, log_continuous_target, grw, pcn, plot_2D, plot_result
)
from utils import calculate_KL, calculate_W2, calculate_mean_difference_squareds, initialise_simulation_dataset, initialise_chain
from tqdm import tqdm
import sys, pickle, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable



if __name__ == '__main__':

    if sys.argv[1] in ['convergence_results', 'error_field_results']:

        # Initial test for acc rate and convergence
        _, _, x1s, x2s, i1s, i2s, K, u, subsample_indices, N, _, G, v = \
            initialise_simulation_dataset(D = 16, ell = 0.3, subsample_factor = 4)

        u0 = initialise_chain(N, K)

        X_grw, acceptance_rate_grw = grw(log_target = log_continuous_target, u0 = u0, data = v, K = K, G = G, n_iters = 10000, beta = 0.2)
        X_pcn, acceptance_rate_pcn = pcn(log_likelihood = log_continuous_likelihood, u0 = u0, data = v, K = K, G = G, n_iters = 10000, beta = 0.2)

        print(acceptance_rate_grw, acceptance_rate_pcn)


    if sys.argv[1] == 'error_field_results':
        
        fig = plt.figure(figsize=(15, 10))
        ax1, ax2, ax3 = [fig.add_subplot(2, 3, i, projection='3d') for i in range(1, 4)]
        ax4, ax5, ax6 = [fig.add_subplot(2, 3, i+3) for i in range(1, 4)]

        plot_result(title = "Original data", u = u, data = v, x = x1s, y = x2s, x_d = x1s[subsample_indices],y_d = x2s[subsample_indices], figure = (fig, ax1))
        plot_result(title = "MH samples mean field", u = np.mean(X_grw, axis = 0), data = v, x = x1s, y = x2s, x_d = x1s[subsample_indices],y_d = x2s[subsample_indices],figure = (fig, ax2))
        plot_result(title = "pCN samples mean field", u = np.mean(X_pcn, axis = 0), data = v, x = x1s, y = x2s, x_d = x1s[subsample_indices], y_d = x2s[subsample_indices], figure = (fig, ax3))

        grw_abs_error = abs(u - np.mean(X_grw, axis = 0))
        pcn_abs_error = abs(u - np.mean(X_pcn, axis = 0))

        plot4 = plot_2D(v, i1s[subsample_indices], i2s[subsample_indices], title='Original Data', figure=(fig, ax4))
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(plot4, cax=cax, orientation='vertical')

        plot5 = plot_2D(grw_abs_error, i1s, i2s, title=f'MH samples mean error field - mean = {grw_abs_error.mean().round(3)}', figure=(fig, ax5))
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(plot5, cax=cax, orientation='vertical')

        plot6 = plot_2D(pcn_abs_error, i1s, i2s, title=f'pCN samples mean error field - mean = {pcn_abs_error.mean().round(3)}', figure=(fig, ax6))
        divider = make_axes_locatable(ax6)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(plot6, cax=cax, orientation='vertical')


        plt.show()


    elif sys.argv[1] == 'convergence_results':
        step = 10
        mh_convergance = calculate_mean_difference_squareds(X_grw, K, G, v, step = step)
        pcn_convergance = calculate_mean_difference_squareds(X_pcn, K, G, v, step = step)

        fig, axs = plt.subplots(1)

        axs.plot(range(step, len(X_grw), step), mh_convergance, label = 'MH')
        axs.plot(range(step, len(X_pcn), step), pcn_convergance, label = 'pCN')

        axs.set_xlabel('k')
        axs.set_ylabel('Normalised squared mean error')

        axs.set_title('Convergence to posterior mean for each algorithm')
        axs.legend()

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

