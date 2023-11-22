import numpy as np
from scipy.linalg import pinvh # for inverting matrices

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
        mu_hat[keep_alpha] = beta_hat * Sigma_hat.dot(np.dot(X[:,keep_alpha].T, y))

        # update hyperparameters
        resid = (y - X.dot(mu_hat))[:,None]
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
