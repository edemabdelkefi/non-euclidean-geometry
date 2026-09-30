# Covariance geometry and synthetic portfolio research

## Positive definite matrices as a statistical manifold

Covariance matrices belong to $\mathcal S_{++}^d$, the symmetric positive definite cone. For symmetric tangent matrices $U,V$ at $A$, the affine-invariant metric is

$$g_A(U,V)=\operatorname{tr}(A^{-1}UA^{-1}V).$$

For zero-mean Gaussian covariance families, the Fisher information metric has the same form with a factor $1/2$. Multiplying the metric by a constant changes distances by a constant but leaves the geodesics and means unchanged. The matrix formulas follow [Pennec, Fillard and Ayache, A Riemannian Framework for Tensor Computing](https://www-sop.inria.fr/asclepios/Publications/Xavier.Pennec/Pennec.IJCV05.pdf).

Symmetric eigendecomposition computes powers, logs and exponentials. Inputs with nonpositive eigenvalues, asymmetry or nonfinite entries are rejected. No silent eigenvalue clipping makes a singular sample appear positive definite.

The distance and geodesic are

$$d(A,B)=\lVert\log(A^{-1/2}BA^{-1/2})\rVert_F,$$

$$\gamma_{A,B}(t)=A^{1/2}(A^{-1/2}BA^{-1/2})^tA^{1/2}.$$

They satisfy $d(CAC^T,CBC^T)=d(A,B)$ for invertible $C$ and

$$\det\gamma_{A,B}(t)=(\det A)^{1-t}(\det B)^t.$$

The logarithmic and exponential maps use the same whitened coordinates:

$$\operatorname{Log}_A(B)=A^{1/2}\log(A^{-1/2}BA^{-1/2})A^{1/2},$$

$$\operatorname{Exp}_A(U)=A^{1/2}\exp(A^{-1/2}UA^{-1/2})A^{1/2}.$$

For $E=A^{1/2}(A^{-1/2}BA^{-1/2})^{1/2}A^{-1/2}$, parallel transport along this geodesic is $U\mapsto EUE^T$. Since $EAE^T=B$, the transported inner product is unchanged. Noncommuting matrix tests verify these properties numerically.

## Mean estimation and convergence

The log-Euclidean mean is $\exp(\sum_iw_i\log A_i)$, with nonnegative weights summing to one. It respects orthogonal transformations but does not generally have full affine congruence invariance.

The affine-invariant Fréchet mean minimizes

$$F(M)=\frac12\sum_iw_i d(M,A_i)^2.$$

At a stationary point, $\sum_iw_i\operatorname{Log}_M(A_i)=0$. The solver starts at the arithmetic mean. It forms $G=\sum_iw_i\log(M^{-1/2}A_iM^{-1/2})$ and takes the geodesic step $M^{1/2}\exp(sG)M^{1/2}$ with Armijo backtracking. It stops when $\lVert G\rVert_F\le10^{-9}$. Failure to meet this condition raises an error. Full objective histories are available from the API; rolling fits save iteration counts and gradient norms.

In the financial comparison, four 63-session sample covariance matrices produce each block mean. Every block contains more observations than assets. This makes positive definiteness plausible under the continuous synthetic distribution; the code still checks it explicitly.

## Data-generating process

Each of 20 independent paths has 12 assets and 1,008 sessions. Three covariance matrices are generated independently for each path:

$$\Sigma_0=LL^T+\operatorname{diag}(\sigma_i^2),\qquad
\Sigma_1=2Q\Sigma_0Q^T,\qquad
\Sigma_2=0.6\Sigma_0+bb^T.$$

$L$ is a random three-factor loading matrix, $Q$ is an orthogonal matrix, idiosyncratic volatilities range from 0.6% to 1.4% per session, and $b$ ranges from -0.8% to 0.8%. Regimes change at sessions 336 and 672.

If $z_t\sim N(0,I)$ and $u_t\sim\chi^2_5$ independently, returns are

$$r_t=\Sigma_{s_t}^{1/2}z_t\sqrt{3/u_t}.$$

This has zero mean and covariance $\Sigma_{s_t}$, while retaining heavy tails. The realized path must have simple returns above -95% for the stated accounting domain; a violation stops the experiment rather than redrawing or truncating observations. All realized returns and true covariance matrices are committed. No market-return prediction claim follows from this synthetic model.

## Chronological estimator comparison

The first 189 sessions fit candidates; the next 63 validate them. Validation chooses $\alpha$ from $\{0,0.05,0.15,0.35,0.65\}$ separately for full-window sample shrinkage, log-Euclidean averaging and affine-invariant averaging. The selected settings are frozen for the entire 756-session holdout.

The covariance input for each later rebalance is the preceding 252 sessions. All estimators use the same history. The sample estimator pools it; the block estimators average four covariances. Final diagonal shrinkage uses $(1-\alpha)\widehat\Sigma+\alpha\operatorname{diag}(\widehat\Sigma)$ for each method. This is validation-selected diagonal shrinkage, not the analytical Ledoit-Wolf estimator.

The score is zero-mean Gaussian quasi-likelihood per asset, with irrelevant constants and the common half factor omitted:

$$\ell_t(\widehat\Sigma)=\frac1d\left[\log\det\widehat\Sigma+r_t^T\widehat\Sigma^{-1}r_t\right].$$

Although innovations are non-Gaussian, the expected score is minimized by their true covariance when their second moment exists. Forecast distance to the known covariance, condition numbers and portfolio variance bias provide complementary diagnostics. Arithmetic pooling and block geometric averaging estimate different summaries during regime mixtures.

## Allocation and accounting

Portfolios solve $\min_w w^T\widehat\Sigma w$ subject to $\mathbf1^Tw=1$ and $0\le w_i\le0.25$, using SLSQP with an analytic gradient. Equal weighting provides a simple baseline. The oracle solves the same constrained problem using the known current covariance. Its regime information is unobservable in the empirical setting.

Every 21 sessions, turnover is $\sum_i|w_i^{new}-w_i^{old}|$. A one-way cost of 5 bp times this turnover is deducted proportionally from wealth before the next return. There is no extra factor of one half because both purchases and sales are charged. All strategies start fully invested at equal weights; the first optimization charges only reallocation from that initial portfolio.

With cost fraction $c_t$ and simple portfolio return $R_t=w_t^Tr_t$,

$$R_t^{net}=(1-c_t)(1+R_t)-1,\qquad
w_{t+1,i}=\frac{w_{t,i}(1+r_{t,i})}{1+R_t}.$$

The cap applies at rebalance dates; holdings can drift past it between rebalances. Net wealth and maximum drawdown follow this self-financing accounting. Annualized volatility is the session standard deviation times $\sqrt{252}$; it is a reporting convention rather than a claim about an exact annual distribution. Empirical 99% shortfall uses the largest eight losses in each 756-session path.

## Monte Carlo uncertainty and interpretation

The primary comparison is $1-\sqrt{\overline{\mathrm{MSE}}_m/\overline{\mathrm{MSE}}_{sample}}$, pooling equal-length path mean squared returns. This differs slightly from comparing the average path standard deviations. A paired bootstrap resamples the 20 complete paths 4,000 times, retaining their validation and dependent portfolio trajectories.

The intervals are pointwise Monte Carlo intervals under this model, not simultaneous guarantees or market evidence. Twenty paths give a limited view of rare losses. The two geometric allocation improvements span zero, and geometric covariance estimates materially understate conditional variance in this design. Shape invariance and numerical convergence alone do not establish forecasting superiority.

The outputs include the fixed protocol, separate validation scores, synthetic returns, true covariances, daily portfolios, covariance diagnostics, path metrics, source hashes and package versions. The original Poincaré, Möbius and Lorentz benchmarks remain in the parent results directory.
