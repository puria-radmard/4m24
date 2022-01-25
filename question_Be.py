from functions import (
    log_continuous_likelihood, log_continuous_target, 
    grw, pcn
)
from utils import initialise_simulation_dataset
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
from functions import subsample

# For part b extension report 

M = 64
Ds = np.linspace(16, 32, 9).astype(int)

ell = 0.3
beta = 0.02

n_iters = 10000

subsample_factors = (Ds**2 / M)

for eD, esf in list(zip(Ds, subsample_factors)):
    eN = eD*eD
    sidx = subsample(eN, esf)

    xs, indices, x1s, x2s, i1s, i2s, K, u, subsample_indices, N, M, G, v = \
        initialise_simulation_dataset(eD, ell, esf)

    # Sample u (latent variables - generating function)
    #  from Gaussian prior
    z = np.random.randn(N)
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    u0 = Kc @ z

    X, grw_ac = grw(
        log_continuous_target, 
        u0 = u0, 
        data = v, 
        K = K,
        G = G, 
        n_iters = n_iters,
        beta = beta
    )

    X, pcn_ac = pcn(
        log_continuous_likelihood, 
        u0 = u0, 
        data = v, 
        K = K,
        G = G, 
        n_iters = n_iters,
        beta = beta
    )

    print('D = ', eD, 'MH acceptance:', grw_ac, 'pCN acceptance:', pcn_ac)
