## Reference objective values for the vennLasso tests (not run by R CMD check).
##
## tests/testthat/test-oglasso-gaussian-binomial.R compares oglasso() objective values with
## minima computed once by this script with CVXR (version 1.8.2) and the ECOS solver at
## tolerance 1e-10, for the designs defined in tests/testthat/helper-objectives.R. The values
## are stored in the test file, so CVXR is not needed to check the package.
##
## usage, from the root of the package source:  Rscript inst/reference/make_refs.R
## output (2026-11-13), copied into the tests:
##   wide binomial intercept=FALSE lambda=0.1  obj=0.459483045734 status=optimal
##   wide binomial intercept=FALSE lambda=0.05 obj=0.319640219644 status=optimal_inaccurate
##   wide binomial intercept=TRUE  lambda=0.1  obj=0.453375103437 status=optimal
##   wide binomial intercept=TRUE  lambda=0.05 obj=0.310300615795 status=optimal
##   tall gaussian standardized    lambda=0.2  obj=1.123732911547 status=optimal
##   tall gaussian standardized    lambda=0.05 obj=0.702201422307 status=optimal
## (oglasso reaches these values to within 1e-9 relative, and is at or below them.)
##
## The second part repeats, on a small design, a live comparison of oglasso() with CVXR
## for the gaussian and binomial families (formerly a test skipped on CRAN).

## Reference objective values with CVXR + ECOS for the test designs (helper-objectives.R).
suppressPackageStartupMessages(library(CVXR))
source(file.path("tests", "testthat", "helper-objectives.R"))
solve_ecos <- function(x, y, groups, w, lambda, family, intercept, scale = rep(1, ncol(x))) {
  n <- nrow(x); p <- ncol(x)
  b <- Variable(p); b0 <- Variable(1)
  e <- if (intercept) b0 + x %*% b else x %*% b
  ## penalty on the standardized coefficients scale * b
  pen <- Reduce(`+`, lapply(seq_along(groups), function(j) w[j] * cvxr_norm(scale[groups[[j]]] * b[groups[[j]]], 2)))
  loss <- if (family == "gaussian") sum_squares(y - e) / (2 * n) else sum(logistic(e) - y * e) / n
  pr <- Problem(Minimize(loss + lambda * pen))
  psolve(pr, solver = "ECOS", abstol = 1e-10, reltol = 1e-10, feastol = 1e-10, max_iters = 500L)
  list(obj = value(pr), status = status(pr))
}
d <- design_wide()
for (ic in c(FALSE, TRUE)) for (l in d$lambda) {
  r <- solve_ecos(d$x, d$y, d$groups, d$w, l, "binomial", ic)
  cat(sprintf("wide binomial intercept=%s lambda=%g obj=%.12f status=%s\n", ic, l, r$obj, r$status))
}
d <- design_tall_std()
s <- apply(d$x, 2, function(v) sqrt(mean((v - mean(v))^2)))
for (l in d$lambda) {
  r <- solve_ecos(d$x, d$y, d$groups, d$w, l, "gaussian", TRUE, scale = s)
  cat(sprintf("tall gaussian standardized lambda=%g obj=%.12f status=%s\n", l, r$obj, r$status))
}

## ---- live comparison (needs vennLasso and testthat as well) ----
suppressPackageStartupMessages({library(vennLasso); library(testthat)})
test_that("gaussian and binomial fits agree with CVXR (ECOS) computed live", {
  skip_on_cran()
  skip_if_not_installed("CVXR")
  set.seed(make_seed("cvxr-live"))
  n <- 60; p <- 9
  x <- matrix(rnorm(n * p), n, p)
  y <- 2 + drop(x[, 1:4] %*% c(1, -1, 1, 0.5)) + rnorm(n)
  yb <- rbinom(n, 1, plogis(drop(x[, 1:4] %*% c(1, -1, 1, 0.5))))
  groups <- list(1:3, 3:5, 5:8)  # variable 9 ungrouped (unpenalized)
  w <- sqrt(lengths(groups))
  for (fam in c("gaussian", "binomial")) {
    yy <- if (fam == "gaussian") y else yb
    f <- oglasso(x, yy, group = groups, family = fam, lambda = c(0.2, 0.05), standardize = FALSE,
                 abs.tol = 1e-10, rel.tol = 1e-10, maxit = 50000L, irls.tol = 1e-10)
    for (k in 1:2) {
      b <- CVXR::Variable(p); b0 <- CVXR::Variable(1)
      e <- b0 + x %*% b
      pen <- Reduce(`+`, lapply(seq_along(groups), function(j) w[j] * CVXR::cvxr_norm(b[groups[[j]]], 2)))
      loss <- if (fam == "gaussian") CVXR::sum_squares(yy - e) / (2 * n) else
        sum(CVXR::logistic(e) - yy * e) / n
      pr <- CVXR::Problem(CVXR::Minimize(loss + f$lambda[k] * pen))
      if (exists("psolve", envir = asNamespace("CVXR"))) {
        getExportedValue("CVXR", "psolve")(pr, solver = "ECOS")
        val <- getExportedValue("CVXR", "value")
        b0v <- drop(val(b0)); bv <- drop(val(b))
      } else {
        sol <- solve(pr, solver = "ECOS")
        b0v <- drop(sol$getValue(b0)); bv <- drop(sol$getValue(b))
      }
      ob <- if (fam == "gaussian") obj_gaussian else obj_binomial
      o_ref <- ob(b0v, bv, x, yy, f$lambda[k], groups, w)
      o_fit <- ob(f$beta[1, k], f$beta[-1, k], x, yy, f$lambda[k], groups, w)
      expect_lt((o_fit - o_ref) / abs(o_ref), 1e-6)
    }
  }
})
