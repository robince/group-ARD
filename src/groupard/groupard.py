import numpy as np
from scipy.linalg import pinvh
from sklearn.utils.validation import _check_sample_weight, check_X_y
from sklearn.linear_model._base import LinearModel, _preprocess_data, _rescale_data


class GroupARDRegression(LinearModel):
    """Bayesian ARD regression with group sparsity.

    Parameters
    ----------
    prior : {'Ridge', 'ARD', 'GroupARD'}, default='GroupARD'
        Type of prior to use for regularization.

    groups : array-like of shape (n_features,), default=None
        Group labels for features. Required when prior='GroupARD'.

    extend_groups : bool, default=False
        Whether to extend/truncate groups to match number of features.

    alpha_threshold : float, default=1e4
        Threshold for removing features with high precision (low variance).

    alpha_init : float, default=1.0
        Initial value for precision parameters.

    max_iter : int, default=300
        Maximum number of iterations for the optimization.

    tol : float, default=1e-3
        Convergence tolerance.

    alpha_1 : float, default=1e-6
        Shape parameter for Gamma prior on noise precision.

    alpha_2 : float, default=1e-6
        Rate parameter for Gamma prior on noise precision.

    lambda_1 : float, default=1e-6
        Shape parameter for Gamma prior on weight precisions.

    lambda_2 : float, default=1e-6
        Rate parameter for Gamma prior on weight precisions.

    fit_intercept : bool, default=True
        Whether to fit an intercept term.

    copy_X : bool, default=True
        Whether to copy input data.

    verbose : bool, default=False
        Whether to print convergence information.
    """

    def __init__(self, *, prior='GroupARD', groups=None, extend_groups=False,
                 alpha_threshold=1e4, alpha_init=1.0, max_iter=300, tol=1.0e-3,
                 alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6,
                 fit_intercept=True, copy_X=True, verbose=False):
        self.prior = prior
        self.groups = groups
        self.extend_groups = extend_groups
        self.alpha_threshold = alpha_threshold
        self.alpha_init = alpha_init
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        if self.max_iter < 1:
            raise ValueError(
                f"max_iter should be greater than or equal to 1. Got {self.max_iter!r}."
            )

        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)

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
            # _rescale_data return signature varies across sklearn versions.
            try:
                X, y = _rescale_data(X, y, sample_weight)
            except ValueError:
                X, y, *_ = _rescale_data(X, y, sample_weight)

        n_samples, n_features = X.shape

        def _normalize_groups(groups_like, n_features, extend):
            g = np.asarray(groups_like, dtype=int)
            if not extend and g.shape[0] != n_features:
                raise ValueError("group definitions not compatible with data")
            if extend:
                if g.shape[0] > n_features:
                    g = g[:n_features]
                elif g.shape[0] < n_features:
                    new_gid = int(g.max()) + 1 if g.size > 0 else 0
                    g = np.concatenate([g, np.full(n_features - g.shape[0], new_gid, dtype=int)])
            # Map labels to contiguous 0..G-1
            _, inv = np.unique(g, return_inverse=True)
            return inv.astype(int)

        if self.prior == 'Ridge':
            group_ests = np.zeros(n_features, dtype=int)
        elif self.prior == 'ARD':
            group_ests = np.arange(n_features, dtype=int)
        elif self.prior == 'GroupARD':
            if self.groups is None:
                raise ValueError("Must provide group labels when using 'GroupARD' prior")
            group_ests = _normalize_groups(self.groups, n_features, self.extend_groups)
        else:
            raise ValueError("Unrecognized prior. Options: ['Ridge', 'ARD', 'GroupARD']")

        # initialize hyperparameter estimates
        alpha_hats = self.alpha_init * np.ones(len(np.unique(group_ests)))
        beta_hat = 1 / np.var(y)

        mu_hat = np.zeros(n_features)
        keep_alpha = np.ones(n_features, dtype=bool)
        mu_hat_prev = None
        results = []
        for iter_ in range(self.max_iter):
            # update posterior mean and covariance
            a_hats = np.array([alpha_hats[g] for g in group_ests])[keep_alpha]
            Sigma_hat = pinvh(beta_hat * (X[:, keep_alpha].T @ X[:, keep_alpha]) + np.diag(a_hats))
            mu_hat[keep_alpha] = beta_hat * Sigma_hat.dot(X[:, keep_alpha].T @ y)

            # update hyperparameters
            resid = y - X.dot(mu_hat)
            mse = float(resid @ resid)
            gamma_hats = 1 - a_hats * np.diag(Sigma_hat)
            # Noise precision update with Gamma prior
            beta_hat = (len(X) - sum(gamma_hats) + 2.0 * self.alpha_1) / (mse + 2.0 * self.alpha_2)

            # update inverse prior variances with Gamma priors
            if self.prior == 'Ridge':
                top = len(mu_hat) - alpha_hats[0] * np.diag(Sigma_hat).sum() + 2.0 * self.lambda_1
                bot = (mu_hat**2).sum() + 2.0 * self.lambda_2
                alpha_hats = (top / bot) * np.ones(len(np.unique(group_ests)))
            elif self.prior == 'ARD':
                active_idx = np.where(keep_alpha)[0]
                for i, idx in enumerate(active_idx):
                    alpha_hats[idx] = (gamma_hats[i] + 2.0 * self.lambda_1) / (mu_hat[idx]**2 + 2.0 * self.lambda_2)
            elif self.prior == 'GroupARD':
                for k in range(len(alpha_hats)):
                    ix = (group_ests == k) & keep_alpha
                    if ix.sum() == 0:
                        continue
                    ixg = (group_ests[np.where(keep_alpha)[0]] == k)
                    top = gamma_hats[ixg].sum() + 2.0 * self.lambda_1
                    bot = (mu_hat[ix]**2).sum() + 2.0 * self.lambda_2
                    alpha_hats[k] = top / bot
            if self.prior != 'Ridge':
                a_hats_full = np.array([alpha_hats[g] for g in group_ests])
                keep_alpha = a_hats_full < self.alpha_threshold
                mu_hat[~keep_alpha] = 0

            # check for convergence
            if mu_hat_prev is not None and np.sum(np.abs(mu_hat - mu_hat_prev)) < self.tol:
                if self.verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            mu_hat_prev = mu_hat.copy()
            results.append(
                {
                    'iteration': iter_,
                    'mu_hat': mu_hat.copy(),
                    'a_hats': a_hats.copy(),
                    'beta_hat': beta_hat,
                    'mse': mse,
                    'n_active_features': int(keep_alpha.sum()),
                }
            )

        # Learned attributes and bookkeeping
        self.coef_ = mu_hat
        self.n_iter_ = iter_ + 1
        self.convergence_history_ = results

        # Posterior covariance expanded to full feature space
        sigma_full = np.zeros((n_features, n_features), dtype=float)
        active_idx = np.where(keep_alpha)[0]
        if active_idx.size > 0:
            sigma_full[np.ix_(active_idx, active_idx)] = Sigma_hat
        self.sigma_ = sigma_full

        # Noise precision (alpha_) and per-feature weight precisions (lambda_)
        self.alpha_ = float(beta_hat)
        group_precisions = np.asarray([alpha_hats[g] for g in range(len(alpha_hats))], dtype=float)
        self.lambda_ = np.asarray([group_precisions[g] for g in group_ests], dtype=float)

        # Active feature mask and normalized groups
        self.active_mask_ = keep_alpha.copy()
        self.groups_ = group_ests.copy()

        # set intercept and rescale coef if needed
        self._set_intercept(X_offset_, y_offset_, X_scale_)

        return self

    def predict(self, X):
        return self._decision_function(X)

