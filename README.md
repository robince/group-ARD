
Suppose we have covariates $X \in \mathbb{R}^{N \times K}$ and observations $y \in \mathbb{R}^N$, where our observation model is $y_i \sim \mathcal{N}(x_i^\top w, \sigma^2)$. Here, $w \in \mathbb{R}^K$ are unknown weights.

In linear regression we want to find the best estimate of the weights given $X$ and $y$. For example, standard linear regression finds the weights $\widehat{w}$ minimizing the sum of the squared residuals:

$$ \|| y - X \widehat{w} \||_2^2 $$

When $N$ is small or $K$ is large, it's often useful to do Bayesian linear regression. This involves choosing a prior on our weights (see [1] for more details). Some common choices or prior are:

1. __Ridge__: $w_i \sim \mathcal{N}(0, \alpha^{-1})$, where $\alpha \in \mathbb{R}$ is called our "inverse prior variance".
2. __Automatic Relevance Determination (ARD)__: $w_i \sim \mathcal{N}(0, \alpha_i^{-1})$. Note that now each covariate has its own inverse prior variance.

Here we consider a third option in between these two, which I will call "Group ARD" (in analogy to Group Lasso [2]). This prior is relevant when our covariates can be grouped. Specifically, we assume the $i^{th}$ covariate has a known group label $c_i \in \\{ 1, 2, \ldots, G\\}$, where $G$ is the total number of groups. The idea is that every covariate in the same group has the same inverse prior variance. In other words:

3. __Group ARD__: $w \sim \mathcal{N}(0, \alpha_{c_i}^{-1})$

[1] Tipping, Michael E. "Sparse Bayesian learning and the relevance vector machine." Journal of machine learning research 1.Jun (2001): 211-244.
[2] Yuan, Ming, and Yi Lin. "Model selection and estimation in regression with grouped variables." Journal of the Royal Statistical Society Series B: Statistical Methodology 68.1 (2006): 49-67.
