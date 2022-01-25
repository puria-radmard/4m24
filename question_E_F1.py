from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from functions import GaussianKernel, log_poisson_likelihood, pcn, plot_2D
from utils import initialise_chain, initialise_spatial_dataset

ell, beta, n_iters, subsample_factor = 2, 0.2, 10000, 3
df, xi, yi, xs, data, N, idx, G, c = initialise_spatial_dataset(subsample_factor)

K = GaussianKernel(xs, ell)
u0 = initialise_chain(N, K)

u_samples, acc_rate = pcn(log_poisson_likelihood, u0 = u0, data = c, K = K, G = G, n_iters = n_iters, beta = beta)
print(acc_rate)

thetas = list(map(np.exp, u_samples))
expected_counts = np.mean(thetas, axis = 0)

fig, axs = plt.subplots(1, 4, figsize = (20, 5))

plot0 = plot_2D(data, xi, yi, title="True counts", colors='viridis', figure = (fig, axs[0]))
divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='8%', pad=0.05)
fig.colorbar(plot0, cax=cax, orientation='vertical')

plot1 = plot_2D(c, xi[idx], yi[idx], title="Observed counts", colors='viridis', figure = (fig, axs[1]))
divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size='8%', pad=0.05)
fig.colorbar(plot1, cax=cax, orientation='vertical')

plot2 = plot_2D(expected_counts, xi, yi, title=f"Expected counts - $\ell = {ell}$", colors='viridis', figure = (fig, axs[2]))
divider = make_axes_locatable(axs[2])
cax = divider.append_axes('right', size='8%', pad=0.05)
fig.colorbar(plot2, cax=cax, orientation='vertical')

plot3 = plot_2D(abs(expected_counts - data), xi, yi, title="Absolute error", colors='viridis', figure = (fig, axs[3]))
divider = make_axes_locatable(axs[3])
cax = divider.append_axes('right', size='8%', pad=0.05)
fig.colorbar(plot3, cax=cax, orientation='vertical')

plt.show()
