# vennLasso 0.2.0

This release fixes the problems that led to the removal of the package from CRAN and a number of
defects in `oglasso()` and `vennLasso()`. Several of them change the fitted values that existing code
obtains; those are marked **(changes results)**.

## CRAN issues

* The vignette no longer loads `natbib`. Citations are processed by pandoc, and natbib's author-year
  check stopped the vignette from building on CRAN (archived 2023-09-21).
* Removed `LazyData` from `DESCRIPTION` (the package has no data directory).
* Native routines are registered and symbol search is disabled.
* Corrected the citation (Biometrics 74(2), 557-565, 2018) in `inst/CITATION` and the vignette, and
  replaced `citEntry()` by `bibentry()`.

## oglasso()

* Fits with an intercept (the default) for `family = "gaussian"` no longer crash R. This also fixes
  `vennLasso(one.intercept = TRUE)` and the adaptive lasso for nearly separable binomial data.
* The wide solvers (nvars >= 2500 and nobs <= 2 * nvars) now solve the correct linear system when a
  variable belongs to more than one group, for both the gaussian and the binomial family. Before, the
  gaussian solver could diverge and the binomial solver could return a non-optimal fit.
  **(changes results)**
* `standardize = TRUE` (the default) now standardizes `x` for the gaussian and binomial families, as
  documented; it was silently ignored. Coefficients are returned on the original scale.
  **(changes results)**
* Cox models: the survival times are no longer centered and scaled, no intercept is fit, the data no
  longer need to be sorted by time, and tied times are handled as in the Breslow partial likelihood.
  Before, the default settings returned coefficients that were wrong by a large factor.
  **(changes results)**
* The lambda sequence starts at the smallest value at which the penalized coefficients are guaranteed
  to be zero for the problem in which the intercept and unpenalized variables are fit. Before, it
  started too high when an intercept was fit, so many leading values gave the null model.
  **(changes results)**
* The path no longer stops silently once more than nobs coefficients are nonzero. A new argument
  `dfmax` limits the number of nonzero coefficients, with a warning when the path is stopped.
  **(changes results)**
* `penalty.factor.fused` no longer causes valid calls to fail (it is unused).
* `group.weights` may be given with one value per group also when an intercept or ungrouped
  variables are present.
* A group matrix is checked to have one row per variable, as documented (the check compared the
  number of columns).
* Designs with fewer than five columns no longer fail in the eigenvalue computation.
* The gaussian solvers no longer stop early when started from the previous solution on the
  lambda path: the restart state of the accelerated iterations is reset for each lambda (and for
  each Newton step of the binomial and Cox solvers). Before, a fit along a path could stop after a
  few iterations at a non-optimal point (objective up to 2.5% too high), whatever the tolerance.
  **(changes results)**
* Compatibility with Eigen 5: `Eigen::Map<Eigen::SparseMatrix>` replaces the removed
  `Eigen::MappedSparseMatrix`.
* The adaptive lasso no longer crashes for wide designs, and the Cox model no longer crashes for
  nvars >= 2500 (it uses the tall solver).
* A sparse `x` is converted to a dense matrix (the sparse solver did not exist and the call failed).
  The fused penalty (`fused`) is not implemented and now gives an informative error.

## vennLasso() and related functions

* `vennLasso()` inherits the changes to `oglasso()` above. In particular **(changes results)**:
  the lambda sequence now starts at the lambda.max of the problem with unpenalized
  subpopulation intercepts (for the help example, 0.1454 instead of 0.1465; with
  `adaptive.lasso = TRUE`, 0.0693 instead of 0.1465, where before the first 9 fits were the
  null model), so a fit selected by its position in the path (for example `fit$beta[, , 21]`)
  refers to a different lambda; paths are no longer cut short once more than nobs coefficients
  are nonzero; and fits along the path are now converged (see the warm-start fix above).
  The help examples now select fits by BIC or AIC instead of by position.
* The effective number of parameters, and so `df`, `aic`, `bic`, `aicc` and `mbic`, follow
  Section 3.3 of Huling et al. (2018): tr{(H + N lambda A)^(-1) H} over the active set, with
  A_j = sum over the groups g containing j of w_g / ||beta_g||. Before, the group weight was in
  the denominator, the unpenalized subpopulation intercepts counted (about) zero, and a diagonal
  shortcut that assumes x_j'x_j = N was used. The standard errors of the gaussian model use the
  same A. **(changes results)**
* Binomial standard errors and confidence intervals (`compute.se`, `conf.int`) treat the
  subpopulation intercepts as columns of the design: the refit no longer adds a global intercept,
  which had left one intercept with an `NA` standard error and the others with standard errors of
  contrasts. **(changes results)**
* Gaussian refitted coefficients (`beta.refit`, `intercept.refit`, `lower.ci.refit`,
  `upper.ci.refit`) are now returned; they were all zero. The refits are carried along the path
  when the selected set does not change. **(changes results)**
* Variables that are missing for every observation with a condition are removed only from the
  combinations that include that condition, and variables observed only with a condition only
  from the combinations that exclude it. Before, the combinations were matched in reversed
  order. **(changes results when `x` has such missing values)**
* A single stratifying condition (a one-column `groups`) no longer fails.

* The intercepts of the subpopulations are left unpenalized when some variable is constant within a
  subpopulation. Before, the penalty factors were indexed in the wrong order and some intercepts
  could be shrunk to zero. **(changes results)**
* `cv.vennLasso()` fits every fold on the lambda sequence of the full-data fit. Before, each fold used
  its own sequence, so the cross-validation error averaged fits at different values of lambda.
  **(changes results)**
* `predict.vennLasso()` uses the correct coefficients when `newx` does not contain every
  subpopulation of the fit, works when a subpopulation has a single new observation, and
  interpolates correctly when `s` has several values. **(changes results)**
* Degrees of freedom are now also computed for binomial models with `compute.se = TRUE` (they
  were `NA`).
* Documentation: `standardize` in `vennLasso()` scales each subpopulation-specific column over all
  observations; the length and order of `penalty.factor`; the returned `df` and criteria; the
  memory needed by the Cox solver.
* The `plot()` method for `cv.vennLasso` objects counts nonzero coefficients without the intercepts.
* Examples that were wrapped in `\dontrun{}` now use `\donttest{}`.
* Added a `testthat` test suite and an R-hub workflow (`.github/workflows/rhub.yaml`).
