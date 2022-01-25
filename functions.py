import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import norm
import matplotlib.cm as cm
import copy


def GaussianKernel(x, l):
    """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function"""
    D = distance_matrix(x, x)
    return np.exp(-pow(D, 2)/(2*pow(l, 2)))


def subsample(N, factor, seed=None):
    assert factor>=1, 'Subsampling factor must be greater than or equal to one.'
    N_sub = int(np.ceil(N / factor))
    if seed: np.random.seed(seed)
    idx = np.random.choice(N, size=N_sub, replace=False)  # Indexes of the randomly sampled points
    return idx


def get_G(N, idx):
    """Generate the observation matrix based on datapoint locations.
    Inputs:
        N - Length of vector to subsample
        idx - Indexes of subsampled coordinates
    Outputs:
        G - Observation matrix"""
    M = len(idx)
    G = np.zeros((M, N))
    for i in range(M):
        G[i,idx[i]] = 1
    return G


def probit(v):
    return np.array([0 if x < 0 else 1 for x in v])

 
def predict_t(samples):
    # pr450
    # Exact: \int p(t^*=1|u)p(u|t)du
    # MC prediction: \sum_{i=1}^N p(t^*|u_i) / N
        # samples = {u_i}i drawn from p(u|t)
    p_t = list(map(norm.cdf, samples))
    mc_preds = np.mean(p_t, axis = 0)
    return mc_preds


###--- Density functions ---###

def log_prior(u, K, K_inverse):
    # Return log p(u)
    # pr450
    N = len(u)
    constant_term = - 0.5 * (N * np.log(2 * np.pi) + np.linalg.slogdet(K)[1])
    log_term = - 0.5 * (u.T @ K_inverse @ u)
    return constant_term + log_term


def log_continuous_likelihood(u, v, G):
    # Return observation likelihood p(v|u)
    # pr450
    M = len(v)
    constant_term = - 0.5 * M * np.log(2*np.pi)
    distance = np.abs(G @ u - v)
    exp_term = - 0.5 * (distance.T @ distance)
    return constant_term + exp_term


def log_probit_likelihood(u, t, G):
    # pr450
    p_1 = norm.cdf(G @ u)
    if (p_1 < 0).sum(): raise Exception('p_1 cannot be < 0')
    return (t*np.log(p_1) + (1-t)*np.log(1-p_1)).sum()


def log_poisson_likelihood(u, c, G):
    # pr450
    subsampled_u = G @ u
    log_term = c * subsampled_u
    return (log_term - np.exp(subsampled_u)).sum()


def log_continuous_target(u, y, K, K_inverse, G):
    return log_prior(u, K, K_inverse) + log_continuous_likelihood(u, y, G)


def log_probit_target(u, t, K, K_inverse, G):
    return log_prior(u, K, K_inverse) + log_probit_likelihood(u, t, G)


def log_poisson_target(u, c, K, K_inverse, G):
    return log_prior(u, K, K_inverse) + log_poisson_likelihood(u, c, G)



###--- MCMC ---###

def grw(log_target, u0, data, K, G, n_iters, beta):
    """ Gaussian random walk Metropolis-Hastings MCMC method
        for sampling from pdf defined by log_target.
    Inputs:
        log_target - log-target density
        u0 - initial sample
        y - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0

    # pr450
    N = len(u0)

    # Inverse computed before the for loop for speed
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    Kc_inverse = np.linalg.inv(Kc)

    # pr450
    K_inverse = Kc_inverse.T @ Kc_inverse

    lt_prev = log_target(u_prev, data, K, K_inverse, G)

    for i in range(n_iters):

        # pr450
        z = np.random.randn(N)
        u_new = u_prev + beta * Kc @ z

        lt_new = log_target(u_new, data, K, K_inverse, G)

        # pr450
        log_alpha = lt_new - lt_prev
        log_uniform = np.log(np.random.random())

        # Accept/Reject
        # pr450
        accept = log_uniform <= log_alpha
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            lt_prev = lt_new
        else:
            X.append(u_prev)

    return X, acc / n_iters


def pcn(log_likelihood, u0, data, K, G, n_iters, beta):
    """ pCN MCMC method for sampling from pdf defined by log_prior and log_likelihood.
    Inputs:
        log_likelihood - log-likelihood function
        u0 - initial sample
        y - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0

    # pr450
    N = len(u0)

    ll_prev = log_likelihood(u_prev, data, G)

    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

    for i in range(n_iters):

        # pr450 - include derivation in appendix!
        z = np.random.randn(N)
        u_new = (np.sqrt(1-beta**2) * u_prev) + (beta * (Kc @ z))

        ll_new = log_likelihood(u_new, data, G)

        # pr450
        log_alpha = ll_new - ll_prev
        log_uniform = np.log(np.random.random())

        # Accept/Reject
        # pr450
        accept = log_uniform < log_alpha
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            ll_prev = ll_new
        else:
            X.append(u_prev)

    return X, acc / n_iters


###--- Plotting ---###

def plot_3D(u, x, y, title=None):
    """Plot the latent variable field u given the list of x,y coordinates"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False)
    if title:  plt.title(title)


def plot_2D(counts, xi, yi, title=None, colors='viridis', figure = None):
    """Visualise count data given the index lists"""
    Z = -np.ones((max(yi) + 1, max(xi) + 1))
    for i in range(len(counts)):
        Z[(yi[i], xi[i])] = counts[i]
    my_cmap = copy.copy(cm.get_cmap(colors))
    my_cmap.set_under('k', alpha=0)
    if figure == None:
        fig, ax = plt.subplots(1)
    else:
        fig, ax = figure
    im = ax.imshow(Z, origin='lower', cmap=my_cmap, clim=[-0.1, np.max(counts)])
    #fig.colorbar(im)
    if title:  ax.set_title(title)
    return im


def plot_result(u, data, x, y, x_d, y_d, title=None, figure = None):
    """Plot the latent variable field u with the observations,
        using the latent variable coordinate lists x,y and the
        data coordinate lists x_d, y_d"""
    if figure == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig, ax = figure
    ax.scatter(x_d, y_d, data, marker='x', color='r')
    ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False, alpha = 0.75)
    ax.scatter(x_d, y_d, data, marker='x', color='r')
    if title:  ax.set_title(title)
