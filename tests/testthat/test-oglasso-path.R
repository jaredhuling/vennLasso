test_that("the path is not truncated silently; dfmax stops it with a warning", {
  set.seed(make_seed("oglasso-path", 1))
  n <- 30; p <- 200
  x <- matrix(rnorm(n * p), n, p)
  y <- drop(x[, 1:5] %*% rep(1, 5)) + rnorm(n)
  lam <- exp(seq(log(2), log(0.05), length.out = 20))
  groups <- list(1:100, 101:200)
  expect_no_warning(f <- oglasso(x, y, group = groups, lambda = lam))
  expect_equal(length(f$lambda), 20L)
  expect_equal(ncol(f$beta), 20L)
  expect_true(any(colSums(f$beta[-1, ] != 0) > n))
  expect_warning(f2 <- oglasso(x, y, group = groups, lambda = lam, dfmax = n), "dfmax")
  expect_lt(length(f2$lambda), 20L)
  expect_true(all(colSums(f2$beta[-1, , drop = FALSE] != 0) <= n))
})

test_that("lambda.max is computed for the problem with an unpenalized intercept", {
  set.seed(make_seed("oglasso-path", 2))
  n <- 200; p <- 30
  x <- matrix(rnorm(n * p), n, p)
  y <- 6 + drop(x[, 1:3] %*% c(1, -1, 0.5)) + rnorm(n)
  groups <- c(list(1:3, 3:6), as.list(7:30))
  f <- oglasso(x, y, group = groups, nlambda = 20, abs.tol = 1e-10, rel.tol = 1e-10, maxit = 50000L)
  ## lambda.max is the boundary value: zero up to the solver tolerance there, exactly zero
  ## slightly above it, and nonzero at the next value of the grid
  expect_lt(max(abs(f$beta[-1, 1])), 1e-6)
  f.above <- oglasso(x, y, group = groups, lambda = f$lambda[1] * 1.001, abs.tol = 1e-10, rel.tol = 1e-10,
                     maxit = 50000L)
  expect_true(all(f.above$beta[-1, 1] == 0))
  expect_gt(sum(f$beta[-1, 2] != 0), 0)
  expect_equal(unname(f$beta[1, 1]), mean(y), tolerance = 1e-8)
  ## singleton groups (the lasso): lambda.max = max |x_j'(y - ybar)| / n on standardized x
  xs <- scale(x) * sqrt(n / (n - 1))
  f1 <- oglasso(x, y, group = as.list(1:p), nlambda = 3)
  expect_equal(f1$lambda[1], max(abs(crossprod(xs, y - mean(y)))) / n, tolerance = 1e-10)
})

test_that("valid calls are not rejected and unsupported inputs are handled", {
  set.seed(make_seed("oglasso-path", 3))
  x <- matrix(rnorm(100 * 5), 100, 5); y <- rnorm(100)
  ## variable 5 in no group, no intercept (failed the penalty.factor.fused length check)
  expect_no_error(f <- oglasso(x, y, group = list(1:2, 3:4), intercept = FALSE, nlambda = 5))
  ## group.weights of length equal to the number of user groups, with an intercept
  expect_no_error(oglasso(x, y, group = list(1:2, 3:4), group.weights = c(1, 2), nlambda = 5))
  expect_error(oglasso(x, y, group = list(1:2, 3:4), group.weights = 1:5, nlambda = 5), "group.weights")
  ## a group matrix with variables in rows and fewer groups than variables
  gm <- cbind(c(1, 1, 0, 0, 0), c(0, 1, 1, 0, 0))
  expect_equal(oglasso(x, y, group = gm, nlambda = 5)$beta,
               oglasso(x, y, group = list(1:2, 2:3), nlambda = 5)$beta)
  ## a sparse x is fit as dense
  expect_equal(oglasso(Matrix::Matrix(x, sparse = TRUE), y, group = list(1:2, 3:4), nlambda = 5)$beta,
               oglasso(x, y, group = list(1:2, 3:4), nlambda = 5)$beta)
  expect_error(oglasso(x, y, group = list(1:2), fused = diag(5)), "not implemented")
  expect_error(oglasso(x, rbinom(100, 1, 0.5) + 1, group = list(1:2), family = "binomial"), "0 and 1")
})

test_that("adaptive lasso runs for the wide solver", {
  set.seed(make_seed("oglasso-path", 4))
  n <- 40; p <- 2500
  x <- matrix(rnorm(n * p), n, p); y <- drop(x[, 1:3] %*% c(1, -1, 1)) + rnorm(n)
  f <- oglasso(x, y, group = as.list(1:p), adaptive.lasso = TRUE, nlambda = 3)
  expect_equal(dim(f$beta), c(p + 1L, 3L))
  expect_true(all(is.finite(f$beta)))
})

test_that("designs with fewer than five columns can be fit", {
  set.seed(make_seed("oglasso-path", 5))
  x <- matrix(rnorm(80 * 3), 80, 3); y <- drop(x %*% c(1, -1, 0)) + rnorm(80)
  yb <- rbinom(80, 1, plogis(drop(x %*% c(1, -1, 0))))
  expect_no_error(f <- oglasso(x, y, group = list(1:2, 2:3), intercept = FALSE, nlambda = 5))
  expect_no_error(oglasso(x, yb, group = list(1:2, 2:3), family = "binomial", nlambda = 5))
  ## no penalty at the smallest lambda: least squares
  f <- oglasso(x, y, group = list(1:2, 2:3), lambda = c(0.1, 1e-9), intercept = FALSE, standardize = FALSE,
               abs.tol = 1e-12, rel.tol = 1e-12, maxit = 50000L)
  expect_equal(unname(f$beta[-1, 2]), unname(coef(lm(y ~ x - 1))), tolerance = 1e-6)
})

test_that("warm-started gaussian paths reach the optimum at every lambda (glmnet reference)", {
  skip_if_not_installed("glmnet")
  for (r in 1:3) {
    set.seed(make_seed("warm-start-path", r))
    n <- 200; p <- 20
    x <- matrix(rnorm(n * p), n, p)
    y <- drop(x[, 1:5] %*% c(2, -1, 1, 0.5, -0.5)) + rnorm(n)
    g <- glmnet::glmnet(x, y, standardize = FALSE, thresh = 1e-16, maxit = 1e7, nlambda = 30)
    f <- oglasso(x, y, group = as.list(1:p), group.weights = rep(1, p), lambda = g$lambda,
                 standardize = FALSE, abs.tol = 1e-9, rel.tol = 1e-9, maxit = 100000L)
    ob <- function(a0, b, l) sum((y - a0 - drop(x %*% b))^2) / (2 * n) + l * sum(abs(b))
    gap <- sapply(seq_along(g$lambda), function(k)
      (ob(f$beta[1, k], f$beta[-1, k], g$lambda[k]) - ob(g$a0[k], g$beta[, k], g$lambda[k])) /
        ob(g$a0[k], g$beta[, k], g$lambda[k]))
    expect_lt(max(gap), 1e-9)
  }
  ## the case where the previous restart state stopped the second fit after 4 iterations
  set.seed(make_seed("warm-start-case"))
  n <- 150; p <- 8
  x <- matrix(rnorm(n * p, mean = 2), n, p) %*% diag(c(1, 10, .1, 1, 5, .5, 2, 3))
  y <- 3 + drop(x %*% c(1, .1, 10, 0, 0, 0, .3, 0)) + rnorm(n)
  xs <- scale(x, center = FALSE, scale = sqrt(colMeans(sweep(x, 2, colMeans(x))^2)))
  cold <- oglasso(xs, y, group = as.list(1:p), group.weights = rep(1, p), lambda = 0.3, standardize = FALSE,
                  intercept = FALSE, abs.tol = 1e-11, rel.tol = 1e-11, maxit = 100000L)
  warm <- oglasso(xs, y, group = as.list(1:p), group.weights = rep(1, p), lambda = c(1, 0.3), standardize = FALSE,
                  intercept = FALSE, abs.tol = 1e-11, rel.tol = 1e-11, maxit = 100000L)
  ob <- function(b) sum((y - xs %*% b)^2) / (2 * n) + 0.3 * sum(abs(b))
  expect_lt(abs(ob(warm$beta[-1, 2]) - ob(cold$beta[-1, 1])), 1e-9)
})
