import numpy as np
from scipy.linalg import pinvh 
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model._base import _rescale_data
from sklearn.utils.validation import _check_sample_weight, check_X_y
from sklearn.linear_model._base import LinearModel, _preprocess_data
from sklearn.linear_model import ARDRegression, BayesianRidge


class GroupARDRegression(RegressorMixin, LinearModel):
    """Bayesian ARD regression with group sparsity.

    Fit the weights of a regression model, using an ARD prior. The weights of
    the regression model are assumed to be in Gaussian distributions.
    Also estimate the parameters lambda (precisions of the distributions of the
    weights) and alpha (precision of the distribution of the noise).
    The estimation is done by iterative procedures (Evidence Maximization).

    The key innovation is the GroupARD prior which allows features to be
    grouped together and share the same precision parameter, enabling
    group-level sparsity where entire groups of features can be 
    simultaneously selected or pruned.

    Parameters
    ----------
    prior : {'Ridge', 'ARD', 'GroupARD'}, default='GroupARD'
        Type of prior to use for regularization:
        - 'Ridge': All features share the same precision (uniform shrinkage)
        - 'ARD': Each feature has its own precision (feature-level sparsity)  
        - 'GroupARD': Features within groups share precision (group-level sparsity)
        
    groups : array-like of shape (n_features,), default=None
        Group labels for features. Required when prior='GroupARD'.
        Each element indicates which group the corresponding feature belongs to.
        Groups should be labeled with integers starting from 0.
        
        Examples:
        - [0, 0, 1, 1, 2, 2] : 6 features in 3 groups of 2 features each
        - [0, 0, 0, 1, 1, 1] : 6 features, first 3 in group 0, last 3 in group 1
        - [0, 1, 0, 1, 2, 2] : Mixed grouping pattern
        
        Typical use cases:
        - Polynomial features: group by degree [0,1,1,2,2,2] for [const,x,y,x²,xy,y²]
        - Multi-categorical: group dummy variables for same categorical feature
        - Domain knowledge: group related measurements (e.g., blood tests, demographics)
        
    extend_groups : bool, default=False
        Whether to extend/truncate groups to match number of features.
        If True and len(groups) < n_features, extra features get new group labels.
        If True and len(groups) > n_features, groups array is truncated.
        
    alpha_threshold : float, default=1e4
        Threshold for removing features with high precision (low variance).
        Features with precision > alpha_threshold are effectively removed.
        
    alpha_init : float, default=1.0
        Initial value for precision parameters. Higher values mean stronger
        initial belief that coefficients should be small.
        
    max_iter : int, default=300
        Maximum number of iterations for the optimization.
        
    tol : float, default=1e-3
        Convergence tolerance. Iteration stops when L2 norm of coefficient 
        changes is less than tol.
        
    alpha_1 : float, default=1e-6
        Shape parameter for Gamma prior on noise precision.
        Controls prior belief about noise variance.
        
    alpha_2 : float, default=1e-6  
        Rate parameter for Gamma prior on noise precision.
        
    lambda_1 : float, default=1e-6
        Shape parameter for Gamma prior on weight precisions.
        Controls prior belief about coefficient magnitudes.
        
    lambda_2 : float, default=1e-6
        Rate parameter for Gamma prior on weight precisions.
        
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
        
    copy_X : bool, default=True
        Whether to copy input data.
        
    verbose : bool, default=False
        Whether to print convergence information.
        
    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of posterior distribution).
        
    intercept_ : float
        Independent term in the linear model.
        
    log_marginal_likelihood_ : list
        Log marginal likelihood at each iteration.
        
    convergence_history_ : list of dict
        Detailed convergence information for each iteration including
        coefficients, precisions, MSE, and number of active features.
        
    max_iter_ : int
        Actual number of iterations performed.
        
    Examples
    --------
    >>> from groupard import GroupARDRegression
    >>> import numpy as np
    >>> X = np.random.randn(100, 6)
    >>> y = X[:, 0] + 2*X[:, 1] + np.random.randn(100) * 0.1
    >>> groups = np.array([0, 0, 1, 1, 2, 2])  # 3 groups of 2 features each
    >>> model = GroupARDRegression(prior='GroupARD', groups=groups)
    >>> model.fit(X, y)
    >>> print(model.coef_)
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
        # Implement your fitting logic here
        # X: array-like, shape (n_samples, n_features)
        # y: array-like, shape (n_samples,)
        # Return self
            # define group labels used for fitting (based on prior)
        
        if self.max_iter < 1:
            raise ValueError(
                "max_iter should be greater than or equal to 1. Got {!r}.".format(
                    self.max_iter
                )
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
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_
        n_samples, n_features = X.shape

        if self.prior == 'Ridge':
            group_ests = np.zeros(X.shape[-1]).astype(int)
        elif self.prior == 'ARD':
            group_ests = list(np.arange(X.shape[-1]))
        elif self.prior == 'GroupARD':
            if self.groups is None:
                raise Exception("Must provide group labels when using 'GroupARD' prior")
            if not self.extend_groups and len(self.groups)!=n_features:
                raise Exception("group definitions not compatible with data")
            if self.extend_groups:
                if len(self.groups)>n_features:
                    group_ests = self.groups[:n_features].copy()
                elif len(self.groups)<n_features:
                    new_group = np.array((n_features-len(self.groups))*[np.max(self.groups)+1])
                    group_ests = np.concatenate([self.groups, new_group])
                else:
                    group_ests = self.groups.copy()
            else:
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
        
        # Store convergence metrics
        self.scores_ = []
        self.log_marginal_likelihood_ = []
        for iter_ in range(self.max_iter):
            # update posterior mean and covariance
            a_hats = np.array([alpha_hats[g] for g in group_ests])[keep_alpha]
            Sigma_hat = pinvh(beta_hat*np.dot(X[:,keep_alpha].T, X[:,keep_alpha]) + np.diag(a_hats))
            mu_hat[keep_alpha] = beta_hat * Sigma_hat.dot(np.dot(X[:,keep_alpha].T, y))

            # update hyperparameters
            resid = (y - X.dot(mu_hat))[:,None]
            mse = np.diag(resid.T @ resid).sum()
            gamma_hats = 1 - a_hats * np.diag(Sigma_hat)
            # Noise precision update with Gamma prior
            beta_hat = (len(X) - sum(gamma_hats) + 2.0 * self.alpha_1) / (mse + 2.0 * self.alpha_2)

            # update inverse prior variances with Gamma priors
            if self.prior == 'Ridge':
                # Ridge update with Gamma prior
                top = len(mu_hat) - alpha_hats[0]*np.diag(Sigma_hat).sum() + 2.0 * self.lambda_1
                bot = (mu_hat**2).sum() + 2.0 * self.lambda_2
                alpha_hats = (top/bot)*np.ones(len(np.unique(group_ests)))
            elif self.prior == 'ARD':
                # ARD update with Gamma prior
                active_idx = np.where(keep_alpha)[0]
                for i, idx in enumerate(active_idx):
                    alpha_hats[idx] = (gamma_hats[i] + 2.0 * self.lambda_1) / (mu_hat[idx]**2 + 2.0 * self.lambda_2)
            elif self.prior == 'GroupARD':
                # Group ARD update with Gamma prior
                for k in range(len(alpha_hats)):
                    ix = (group_ests == k) & keep_alpha
                    if ix.sum() == 0:
                        continue
                    ixg = (group_ests[np.where(keep_alpha)[0]] == k)
                    top = gamma_hats[ixg].sum() + 2.0 * self.lambda_1
                    bot = (mu_hat[ix]**2).sum() + 2.0 * self.lambda_2
                    alpha_hats[k] = top/bot
            if self.prior != 'Ridge':
                a_hats = np.array([alpha_hats[g] for g in group_ests])
                keep_alpha = a_hats < self.alpha_threshold
                a_hats[~keep_alpha] = np.inf
                mu_hat[~keep_alpha] = 0

            self.coef_ = mu_hat
            
            # Compute log marginal likelihood for convergence monitoring
            # Following sklearn's approach
            if keep_alpha.sum() > 0:
                # Only compute for active features
                active_a_hats = a_hats
                active_Sigma = Sigma_hat
                active_mu = mu_hat[keep_alpha]
                
                # Log marginal likelihood computation
                log_det_Sigma = np.sum(np.log(np.diag(np.linalg.cholesky(active_Sigma))))
                log_det_A = np.sum(np.log(active_a_hats))
                
                # Data fit term
                data_fit = -0.5 * beta_hat * mse
                
                # Complexity penalty terms
                complexity = 0.5 * (log_det_Sigma + log_det_A + len(active_mu) * np.log(beta_hat))
                
                # Prior terms (from Gamma priors)
                prior_alpha = (self.alpha_1 - 1.0) * np.log(beta_hat) - self.alpha_2 * beta_hat
                prior_lambda = np.sum((self.lambda_1 - 1.0) * np.log(active_a_hats) - self.lambda_2 * active_a_hats)
                
                log_marginal = data_fit + complexity + prior_alpha + prior_lambda
                self.log_marginal_likelihood_.append(log_marginal)
            else:
                self.log_marginal_likelihood_.append(-np.inf)
            
            # check for convergence
            if mu_hat_prev is not None and np.sum(np.abs(mu_hat - mu_hat_prev)) < self.tol:
                if self.verbose:
                    print("Convergence after ", str(iter_), " iterations")
                    print(f"Final log marginal likelihood: {self.log_marginal_likelihood_[-1]:.4f}")
                break
            mu_hat_prev = mu_hat.copy()
            results.append(
                {'iteration': iter_,
                 'mu_hat': mu_hat.copy(),
                'a_hats': a_hats.copy(),
                'beta_hat': beta_hat,
                'mse': mse,
                'n_active_features': keep_alpha.sum(),
                'log_marginal_likelihood': self.log_marginal_likelihood_[-1] if self.log_marginal_likelihood_ else -np.inf})
            
        self.max_iter_ = iter_ + 1
        self.convergence_history_ = results
        
        self._set_intercept(X_offset_, y_offset_, X_scale_)

        return self

    def predict(self, X):
        # Implement your prediction logic here
        # X: array-like, shape (n_samples, n_features)
        # Return y_pred: array-like, shape (n_samples,)
        return self._decision_function(X)