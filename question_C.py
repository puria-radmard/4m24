from functions import log_probit_likelihood, log_probit_target, plot_2D, predict_t, pcn, probit
from utils import initialise_chain, initialise_simulation_dataset
import matplotlib.pyplot as plt
import numpy as np

D = 16 # number of samples in each dimension
ell = 0.15 # length scale of Gaussian prior
subsample_factor = 4
beta = 0.2
n_iters = 10000

xs, indices, x1s, x2s, i1s, i2s, K, u, subsample_indices, N, M, G, v = \
    initialise_simulation_dataset(D, ell, subsample_factor)

t = probit(v)
u0 = initialise_chain(N, K)

X, acc_rate = pcn(log_probit_likelihood, u0 = u0, data = t, K = K, G = G, n_iters = n_iters, beta = beta)
preds = predict_t(X)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
plot_2D(probit(u), i1s, i2s, title='Original Data', figure=(fig, axs[0]))     # Plot true class assignments
plot_2D(t, i1s[subsample_indices], i2s[subsample_indices], title='Probit Data', figure=(fig, axs[1]))     # Plot data
plot_2D(preds, i1s, i2s, title = '$p(t=1)$', figure=(fig, axs[2]))
plot_2D(preds>0.5, i1s, i2s, title = '$p(t=1) > 0.5$', figure=(fig, axs[3]))

print("acc_rate", acc_rate)

plt.show()
