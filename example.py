#%% imports

import numpy as np
import matplotlib.pyplot as plt
from model import fit

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
plt.title('Variance explained')

# plot scatter of true and estimated weights
plt.subplot(nrows,ncols,c); c += 1
for k in range(len(groups)):
    plt.plot(mu[groups == k], mu_hat[groups == k], '.')
plt.plot(plt.xlim(), plt.xlim(), 'k-', zorder=-1, alpha=0.2)
plt.axis('equal')
plt.xlabel('$\mu$')
plt.ylabel('$\widehat{\mu}$')
plt.title('Estimated vs. True weights')

# plot estimated weights
plt.subplot(nrows,ncols,c); c += 1
plot_weights(mu_hat, groups, ylabel='$\widehat{\mu}$')
plt.title('Estimated weights')

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
plt.title('Inverse prior variances')

# plot true weights
plt.subplot(nrows,ncols,c); c += 1
plot_weights(mu, groups)
plt.title('True weights')
plt.tight_layout()
