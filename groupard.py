#%% imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinvh # for inverting matrices

# better plot defaults
import matplotlib as mpl
mpl.rcParams['font.size'] = 10
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def get_rsq(Y, Yhat):
    if len(Y.shape) == 1:
        Y = Y[:,None]
    if len(Yhat.shape) == 1:
        Yhat = Yhat[:,None]
    top = Yhat - Y
    bot = Y - Y.mean(axis=0)[None,:]
    return 1 - np.diag(top.T @ top).sum()/np.diag(bot.T @ bot).sum()

def plot_weights(mus, groups, ylabel='$\mu$'):
    # visualize ground-truth weights, colored by group
    for k in range(K):
        muc = [w if g==k else np.nan for g,w in zip(groups, mus)]
        plt.bar(np.arange(len(mu)), muc, label=k)
    plt.xticks([])
    plt.plot(plt.xlim(), np.zeros(2), 'k-', zorder=-1, alpha=0.2)
    plt.xlabel('covariate index (i)')
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()

#%% generate data

# set seed
rng = np.random.RandomState(222)

# sample sizes
N = 200 # number of samples
D = 50 # number of covariates
K = 5 # number of groups

# pick random hyperparameters
alphas = rng.randn(K)**2 # prior inverse variances
alphas[[1,3]] = np.inf # set some prior inv. vars to inf
beta = 0.01 # inverse observation variance (i.e., 1/sigma**2)

# sample groups, weights, and noise
groups = np.sort(rng.randint(low=0, high=K, size=D)) # group assignments
prior_vars = 1/np.array([alphas[g] for g in groups])
mu = np.sqrt(prior_vars) * rng.randn(D) # sample weights from prior
nse = (1/np.sqrt(beta)) * rng.randn(N) # sample random noise

# make data
X = rng.randn(N, D) # sample random covariates
y = X.dot(mu) + nse # create observations

# visualize weights
plot_weights(mu, groups)

#%% model fitting

def fit(X, y, prior='Group ARD', groups=None, alpha_threshold=1e4, alpha_init=1e-5, niters=1000, tol=1e-5):
    """
    tol (float): convergence threshold
    """
    # define group labels used for fitting (based on prior)
    if prior == 'Ridge':
        group_ests = np.zeros(X.shape[-1]).astype(int)
    elif prior == 'ARD':
        group_ests = list(np.arange(X.shape[-1]))
    elif prior == 'Group ARD':
        if groups is None:
            raise Exception("Must provide group labels when using 'Group ARD' prior")
        group_ests = groups.copy()
    else:
        raise Exception("Unrecognized prior. Options: ['Ridge', 'ARD', 'Group ARD']")

    # initialize hyperparameter estimates
    alpha_hats = alpha_init*np.ones(len(np.unique(group_ests)))
    beta_hat = 1/np.var(y)

    mu_hat = np.zeros(X.shape[1])
    keep_alpha = np.ones(X.shape[-1]).astype(bool)
    mu_hat_prev = None
    results = []
    for _ in range(niters):
        # update posterior mean and covariance
        a_hats = np.array([alpha_hats[g] for g in group_ests])[keep_alpha]
        Sigma_hat = pinvh(beta_hat*np.dot(X[:,keep_alpha].T, X[:,keep_alpha]) + np.diag(a_hats))
        mu_hat[keep_alpha] = beta_hat * Sigma_hat.dot(np.dot(X[:,keep_alpha].T, ytr))

        # update hyperparameters
        resid = (ytr - X.dot(mu_hat))[:,None]
        mse = np.diag(resid.T @ resid).sum()
        gamma_hats = 1 - a_hats * np.diag(Sigma_hat)
        beta_hat = (len(X) - sum(gamma_hats)) / mse

        # update inverse prior variances
        if prior == 'Ridge':
            # Ridge update
            top = len(mu_hat) - alpha_hats[0]*np.diag(Sigma_hat).sum()
            bot = (mu_hat**2).sum()
            alpha_hats = (top/bot)*np.ones(len(np.unique(group_ests)))
        elif prior == 'ARD':
            # ARD update
            alpha_hats[keep_alpha] = gamma_hats / (mu_hat[keep_alpha]**2)
        elif prior == 'Group ARD':
            # Group ARD update
            for k in range(len(alpha_hats)):
                ix = (group_ests == k) & keep_alpha
                if ix.sum() == 0:
                    continue
                ixg = (group_ests[np.where(keep_alpha)[0]] == k)
                top = gamma_hats[ixg].sum()
                bot = (mu_hat[ix]**2).sum()
                alpha_hats[k] = top/bot
        if prior != 'Ridge':
            a_hats = np.array([alpha_hats[g] for g in group_ests])
            keep_alpha = a_hats < alpha_threshold
            a_hats[~keep_alpha] = np.inf
            mu_hat[~keep_alpha] = 0

        # check for convergence
        if mu_hat_prev is not None and np.linalg.norm(mu_hat - mu_hat_prev) < tol:
            break
        mu_hat_prev = mu_hat.copy()
        results.append(
            {'mu_hat': mu_hat.copy(),
            'a_hats': a_hats.copy(),
            'beta_hat': beta_hat})

    return mu_hat, results

#%% fit and visualize model

# train-test split
rng = np.random.RandomState(555)
ixTr = np.argsort(rng.rand(len(X))) < int(len(X)/2)
Xtr, ytr = X[ixTr], y[ixTr]
Xte, yte = X[~ixTr], y[~ixTr]

# fit model
mu_hat, results = fit(Xtr, ytr, groups=groups, prior='Group ARD')

# get r-squareds
rsqs = []
for result in results:
    mu_hat = result['mu_hat']
    rsq_tr = get_rsq(ytr, Xtr.dot(mu_hat))
    rsq_te = get_rsq(yte, Xte.dot(mu_hat))
    rsqs.append((rsq_tr, rsq_te))
rsqs = np.vstack(rsqs)

# visualize model fits
ncols = 3; nrows = 2
plt.figure(figsize=(3*ncols,3*nrows)); c = 1

# plot r-squareds
plt.subplot(nrows,ncols,c); c += 1
plt.plot(rsqs[:,0], label='train')
plt.plot(rsqs[:,1], label='test')
plt.legend(fontsize=8)
plt.xlabel('# iterations')
plt.ylabel('$R^2$')
plt.ylim([0,1])

# plot scatter of true and estimated weights
plt.subplot(nrows,ncols,c); c += 1
for k in range(len(groups)):
    plt.plot(mu[groups == k], mu_hat[groups == k], '.')
plt.plot(plt.xlim(), plt.xlim(), 'k-', zorder=-1, alpha=0.2)
plt.axis('equal')
plt.xlabel('$\mu$')
plt.ylabel('$\widehat{\mu}$')

# plot estimated weights
plt.subplot(nrows,ncols,c); c += 1
plot_weights(mu_hat, groups, ylabel='$\widehat{\mu}$')

# plot scatter of true and estimated inverse prior variances
plt.subplot(nrows,ncols,c); c += 1
a_hats = results[-1]['a_hats']
for k in range(len(groups)):
    eps = 1e-9
    plt.plot(prior_vars[groups == k]+eps, 1/a_hats[groups == k]+eps, '.')
plt.plot(plt.xlim(), plt.xlim(), 'k-', zorder=-1, alpha=0.2)
plt.xscale('log'); plt.yscale('log')
plt.xlabel('$\\alpha^{-1}$')
plt.ylabel('$\widehat{\\alpha}^{-1}$')

# plot true weights
plt.subplot(nrows,ncols,c); c += 1
plot_weights(mu, groups)
plt.tight_layout()
