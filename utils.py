# Imports
from functions import subsample, GaussianKernel, get_G
import numpy as np
from tqdm import tqdm
import pandas as pd
import scipy

def initialise_chain(_N, _K):
    # Sample u (latent variables - generating function)
    #  from Gaussian prior
    z = np.random.randn(_N)
    Kc = np.linalg.cholesky(_K + 1e-6 * np.eye(_N))
    u0 = Kc @ z
    return u0


def initialise_simulation_dataset(D, ell, subsample_factor):

    N = D**2 # Number of data points

    # Generate grid and grid indices
    xs = np.array([[x1, x2] for x1 in np.linspace(0, 1, D) for x2 in np.linspace(0, 1, D)])
    indices = np.array([[i1, i2] for i1 in np.arange(D) for i2 in np.arange(D)])
    x1s, x2s = xs[:,0], xs[:,1]
    i1s, i2s = indices[:,0], indices[:,1]

    K = GaussianKernel(xs, ell)

    # Sample u (latent variables - generating function)
    #  from Gaussian prior
    z = np.random.randn(N)
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    u = Kc @ z

    # Subsample prior_samples
    subsample_indices = subsample(N, subsample_factor)
    M = len(subsample_indices) # Number of latent variables

    # Get G matrix (subsampler) and v (noisy data)
    G = get_G(N, subsample_indices)
    eps = np.random.randn(M)
    v = G@u + eps

    return xs, indices, x1s, x2s, i1s, i2s, K, u, subsample_indices, N, M, G, v


def initialise_spatial_dataset(subsample_factor):

    df = pd.read_csv('data.csv')

    ### Generate the arrays needed from the dataframe
    data = np.array(df["bicycle.theft"])
    xi = np.array(df['xi'])
    yi = np.array(df['yi'])
    N = len(data)
    xs = [(xi[i],yi[i]) for i in range(N)]

    ### Subsample the original data set
    idx = subsample(N, subsample_factor, seed=42)
    G = get_G(N,idx)
    c = G @ data

    return df, xi, yi, xs, data, N, idx, G, c


def get_true_posterior(K, G, v):
    true_covariance_inverse = np.linalg.inv(K) + (G.T @ G)
    true_covariance = np.linalg.inv(true_covariance_inverse)
    true_mean = true_covariance @ G.T @ v
    return true_covariance_inverse, true_covariance, true_mean


def calculate_W2(samples, K, G, v, step = 100):
    w2s = []
    true_covariance_inverse, true_covariance, true_mean = get_true_posterior(K, G, v)
    for i in tqdm(range(step, len(samples), step)):
        mean = np.mean(samples[:i], axis = 0)
        cov = np.cov(np.stack(samples[:i]).T)
        sqrt_cov = scipy.linalg.sqrtm(cov)
        mean_term = (mean - true_mean).T @ (mean - true_mean)
        trace_term = np.trace(
            cov + true_covariance 
            - 2 * scipy.linalg.sqrtm(sqrt_cov @ true_covariance @ sqrt_cov)
        )
        w2s.append((mean_term + trace_term)**0.5)
    import pdb; pdb.set_trace()
    return w2s

def calculate_mean_difference_squareds(samples, K, G, v, step = 100):
    mean_diffs_2s = []
    N = K.shape[0]
    true_covariance_inverse, true_covariance, true_mean = get_true_posterior(K, G, v)
    for i in tqdm(range(step, len(samples), step)):
        mean_diff = np.mean(samples[:i], axis = 0) - true_mean
        squared = mean_diff.T @ mean_diff
        mean_diffs_2s.append(squared / N)
    return mean_diffs_2s


def calculate_KL(samples, K, G, v, step = 100):
    kls = []
    true_covariance_inverse, true_covariance, true_mean = get_true_posterior(K, G, v)
    _s, true_covariance_log_det = np.linalg.slogdet(true_covariance)
    d = true_mean.size
    for i in tqdm(range(step, len(samples), step)):
        cov = np.cov(np.stack(samples[:i]).T)
        mean_diff = np.mean(samples[:i], axis = 0) - true_mean
        double = (
            true_covariance_log_det
            - np.linalg.slogdet(cov)[1]
            - d
            + np.trace(true_covariance_inverse @ cov)
            + (mean_diff.T @ true_covariance_inverse @ mean_diff)
        )
        kls.append(0.5 * double)
        import pdb; pdb.set_trace()
    return kls

