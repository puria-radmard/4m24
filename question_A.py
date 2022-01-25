# Imports
from functions import plot_3D, plot_result, subsample, GaussianKernel, get_G
import numpy as np
from utils import initialise_simulation_dataset
import matplotlib.pyplot as plt

D = 16 # number of samples in each dimension
ells = [0.1, 0.2, 0.3, 0.4]
subsample_factor = 4

fig = plt.figure(figsize=(20, 5))
axs = [
    fig.add_subplot(1, 4, i+1, projection='3d') for i in range(len(ells))
]

for i, ell in enumerate(ells): # length scales of Gaussian priors

    xs, indices, x1s, x2s, i1s, i2s, K, u, subsample_indices, N, M, G, v = \
        initialise_simulation_dataset(D, ell, subsample_factor)

    plot_result(
        u = u, 
        data = v, 
        x = x1s, 
        y = x2s, 
        x_d = x1s[subsample_indices],
        y_d = x2s[subsample_indices],
        title=None,
        figure=(fig, axs[i])
    )
    axs[i].set_title(f'$\ell={ell}$')

plt.show()