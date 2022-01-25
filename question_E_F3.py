from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from functions import GaussianKernel, log_poisson_likelihood, pcn, plot_2D
from utils import initialise_chain, initialise_spatial_dataset
import sys

ells = list(map(float, sys.argv[1:]))
beta, n_iters, subsample_factor = 0.2, 10000, 3
df, xi, yi, xs, data, N, idx, G, c = initialise_spatial_dataset(subsample_factor)

fig, axs = plt.subplots(1, len(ells), figsize = (len(ells)*5, 5))

for ax, ell in zip(axs, ells):

    K = GaussianKernel(xs, ell)
    u0 = initialise_chain(N, K)

    u_samples, acc_rate = pcn(log_poisson_likelihood, u0 = u0, data = c, K = K, G = G, n_iters = n_iters, beta = beta)
    print(acc_rate)

    thetas = list(map(np.exp, u_samples))
    expected_counts = np.mean(thetas, axis = 0)

    plot = plot_2D(expected_counts, xi, yi, title=f"Expected counts - $\ell = {ell}$", colors='viridis', figure = (fig, ax))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='8%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation='vertical')

plt.show()
