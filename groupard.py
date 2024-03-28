import numpy as np
from scipy.linalg import pinvh 
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model._base import _rescale_data
from sklearn.utils.validation import _check_sample_weight
from sklearn.linear_model._base import LinearModel, _preprocess_data

class GroupARDRegression(RegressorMixin, LinearModel):
    def __init__(self, *, prior='Ridge', groups=None, alpha_threshold=1e4, alpha_init=1e-5, n_iter=500, tol=1.0e-3, 
                 fit_intercept=True, copy_X=True, verbose=False):
        self.prior = prior
        self.groups = groups
        self.alpha_threshold = alpha_threshold
        self.alpha_init = alpha_init
        self.n_iter = n_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        # Implement your fitting logic here
        # X: array-like, shape (n_samples, n_features)
        # y: array-like, shape (n_samples,)
        # Return self
            # define group labels used for fitting (based on prior)
        
        if self.n_iter < 1:
            raise ValueError(
                "n_iter should be greater than or equal to 1. Got {!r}.".format(
                    self.n_iter
                )
            )

        X, y = self._validate_data(X, y, dtype=np.float64, y_numeric=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        self._normalize = False
        X, y, X_offset_, y_offset_, X_scale_ = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_

        if self.prior == 'Ridge':
            group_ests = np.zeros(X.shape[-1]).astype(int)
        elif self.prior == 'ARD':
            group_ests = list(np.arange(X.shape[-1]))
        elif self.prior == 'GroupARD':
            if self.groups is None:
                raise Exception("Must provide group labels when using 'GroupARD' prior")
            group_ests = self.groups.copy()
        else:
            raise Exception("Unrecognized prior. Options: ['Ridge', 'ARD', 'GroupARD']")

        # initialize hyperparameter estimates
        alpha_hats = self.alpha_init*np.ones(len(np.unique(group_ests)))
        beta_hat = 1/np.var(y)

        mu_hat = np.zeros(X.shape[1])
        keep_alpha = np.ones(X.shape[-1]).astype(bool)
        mu_hat_prev = None
        results = []
        for iter_ in range(self.n_iter):
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
            if self.prior == 'Ridge':
                # Ridge update
                top = len(mu_hat) - alpha_hats[0]*np.diag(Sigma_hat).sum()
                bot = (mu_hat**2).sum()
                alpha_hats = (top/bot)*np.ones(len(np.unique(group_ests)))
            elif self.prior == 'ARD':
                # ARD update
                alpha_hats[keep_alpha] = gamma_hats / (mu_hat[keep_alpha]**2)
            elif self.prior == 'GroupARD':
                # Group ARD update
                for k in range(len(alpha_hats)):
                    ix = (group_ests == k) & keep_alpha
                    if ix.sum() == 0:
                        continue
                    ixg = (group_ests[np.where(keep_alpha)[0]] == k)
                    top = gamma_hats[ixg].sum()
                    bot = (mu_hat[ix]**2).sum()
                    alpha_hats[k] = top/bot
            if self.prior != 'Ridge':
                a_hats = np.array([alpha_hats[g] for g in group_ests])
                keep_alpha = a_hats < self.alpha_threshold
                a_hats[~keep_alpha] = np.inf
                mu_hat[~keep_alpha] = 0

            self.coef_ = mu_hat
            # check for convergence
            if mu_hat_prev is not None and np.linalg.norm(mu_hat - mu_hat_prev) < self.tol:
                if self.verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            mu_hat_prev = mu_hat.copy()
            results.append(
                {'mu_hat': mu_hat.copy(),
                'a_hats': a_hats.copy(),
                'beta_hat': beta_hat})
            
        self.n_iter_ = iter_ + 1
        
        self._set_intercept(X_offset_, y_offset_, X_scale_)

        return self

    def predict(self, X):
        # Implement your prediction logic here
        # X: array-like, shape (n_samples, n_features)
        # Return y_pred: array-like, shape (n_samples,)
        return self._decision_function(X)