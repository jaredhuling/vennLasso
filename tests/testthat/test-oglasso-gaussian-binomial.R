test_that("standardize = TRUE matches glmnet's standardized lasso (singleton groups)", {
  skip_if_not_installed("glmnet")
  set.seed(make_seed("standardize-glmnet"))
  n <- 150; p <- 6
  x <- matrix(rnorm(n * p, mean = 2), n, p) %*% diag(c(1, 10, 0.1, 1, 5, 0.5))
  eta <- drop(x %*% c(1, 0.1, 10, 0, 0, 0))
  y <- 1 + eta + rnorm(n)
  yb <- rbinom(n, 1, plogis(eta - mean(eta)))
  lam <- c(0.3, 0.05)
  for (fam in c("gaussian", "binomial")) {
    yy <- if (fam == "gaussian") y else yb
    f <- oglasso(x, yy, group = as.list(1:p), group.weights = rep(1, p), family = fam, lambda = lam,
                 standardize = TRUE, abs.tol = 1e-11, rel.tol = 1e-11, maxit = 50000L, irls.tol = 1e-11)
    g <- glmnet::glmnet(x, yy, family = fam, lambda = lam, standardize = TRUE, thresh = 1e-16, maxit = 1e7)
    expect_equal(unname(f$beta), unname(as.matrix(rbind(g$a0, g$beta))), tolerance = 1e-5)
    ## and standardization changes the fit
    f0 <- oglasso(x, yy, group = as.list(1:p), group.weights = rep(1, p), family = fam, lambda = lam,
                  standardize = FALSE, abs.tol = 1e-11, rel.tol = 1e-11, maxit = 50000L, irls.tol = 1e-11)
    expect_gt(max(abs(f$beta - f0$beta)), 0.1)
  }
})

test_that("standardized fits with an intercept and overlapping groups reach the reference objective", {
  ## reference: CVXR + ECOS minimum of (1/2n)||y - b0 - X b||^2 + lambda sum_g w_g ||(s * b)_g||
  ref <- c(1.123732911547, 0.702201422307)
  d <- design_tall_std()
  s <- apply(d$x, 2, function(v) sqrt(mean((v - mean(v))^2)))
  f <- oglasso(d$x, d$y, group = d$groups, lambda = d$lambda, abs.tol = 1e-10, rel.tol = 1e-10, maxit = 50000L)
  for (k in 1:2) {
    o <- sum((d$y - f$beta[1, k] - d$x %*% f$beta[-1, k])^2) / (2 * nrow(d$x)) +
      f$lambda[k] * grp_pen(s * f$beta[-1, k], d$groups, d$w)
    expect_lt((o - ref[k]) / ref[k], 1e-8)
  }
})

test_that("the wide logistic solver reaches the reference objective with overlapping groups", {
  ## reference: CVXR + ECOS; rows intercept FALSE / TRUE, columns lambda 0.1 / 0.05
  ref <- rbind(c(0.459483045734, 0.319640219644), c(0.453375103437, 0.310300615795))
  d <- design_wide()
  for (ic in c(FALSE, TRUE)) {
    f <- oglasso(d$x, d$y, group = d$groups, family = "binomial", lambda = d$lambda, intercept = ic,
                 standardize = FALSE, group.weights = d$w, abs.tol = 1e-9, rel.tol = 1e-9, maxit = 10000L,
                 irls.tol = 1e-9)
    expect_equal(length(f$lambda), 2L)
    for (k in 1:2) {
      expect_true(all(f$beta[4:11, k] != 0))  # overlapped variables are active
      o <- obj_binomial(f$beta[1, k], f$beta[-1, k], d$x, d$y, f$lambda[k], d$groups, d$w)
      expect_lt((o - ref[ic + 1, k]) / ref[ic + 1, k], 1e-8)
    }
  }
})

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
