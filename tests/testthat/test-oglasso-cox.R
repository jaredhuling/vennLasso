cox_data <- function(n, p, seed, ties = FALSE) {
  set.seed(seed)
  x <- matrix(rnorm(n * p), n, p)
  tt <- rexp(n, exp(0.8 * x[, 1] - 0.5 * x[, 2])); cc <- rexp(n, 0.2)
  time <- pmin(tt, cc) * 10
  if (ties) time <- ceiling(time)
  list(x = x, time = time, status = as.integer(tt <= cc))
}

test_that("Cox fits match survival::coxph with nearly no penalty, for any standardize/intercept and row order", {
  d <- cox_data(300, 5, make_seed("cox_data", 11))
  ref <- coef(survival::coxph(survival::Surv(d$time, d$status) ~ d$x, ties = "breslow"))
  for (st in c(TRUE, FALSE)) for (ic in c(TRUE, FALSE)) {
    f <- oglasso(d$x, d$time, delta = d$status, group = as.list(1:5), family = "coxph",
                 lambda = c(1e-3, 1e-8), standardize = st, intercept = ic,
                 abs.tol = 1e-10, rel.tol = 1e-10, maxit = 20000L)
    expect_equal(unname(f$beta[1, 2]), 0)
    expect_equal(unname(f$beta[-1, 2]), unname(ref), tolerance = 1e-5)
  }
})

test_that("Cox fits handle tied times as Breslow does", {
  d <- cox_data(300, 4, make_seed("cox_data", 12), ties = TRUE)
  expect_true(any(duplicated(d$time[d$status == 1])))
  ref <- coef(survival::coxph(survival::Surv(d$time, d$status) ~ d$x, ties = "breslow"))
  f <- oglasso(d$x, d$time, delta = d$status, group = as.list(1:4), family = "coxph",
               lambda = c(1e-3, 1e-8), standardize = FALSE, abs.tol = 1e-10, rel.tol = 1e-10, maxit = 20000L)
  expect_equal(unname(f$beta[-1, 2]), unname(ref), tolerance = 1e-5)
})

test_that("penalized Cox fits match glmnet (lasso as singleton groups)", {
  skip_if_not_installed("glmnet")
  d <- cox_data(300, 8, make_seed("cox_data", 13))
  lam <- c(0.05, 0.02)
  f <- oglasso(d$x, d$time, delta = d$status, group = as.list(1:8), group.weights = rep(1, 8),
               family = "coxph", lambda = lam, standardize = FALSE,
               abs.tol = 1e-10, rel.tol = 1e-10, maxit = 20000L, irls.tol = 1e-10)
  g <- glmnet::glmnet(d$x, survival::Surv(d$time, d$status), family = "cox", lambda = lam,
                      standardize = FALSE, thresh = 1e-14)
  expect_equal(unname(f$beta[-1, ]), unname(as.matrix(g$beta)), tolerance = 1e-4)
})

test_that("Cox lambda.max gives the zero solution", {
  d <- cox_data(200, 6, make_seed("cox_data", 14))
  f <- oglasso(d$x, d$time, delta = d$status, group = as.list(1:6), family = "coxph", nlambda = 5,
               abs.tol = 1e-10, rel.tol = 1e-10, maxit = 20000L)
  expect_lt(max(abs(f$beta[, 1])), 1e-6)
  f.above <- oglasso(d$x, d$time, delta = d$status, group = as.list(1:6), family = "coxph",
                     lambda = f$lambda[1] * 1.001, abs.tol = 1e-10, rel.tol = 1e-10, maxit = 20000L)
  expect_true(all(f.above$beta[, 1] == 0))
  expect_true(any(f$beta[, 2] != 0))
})

test_that("the Cox model with p >= 2500 and n <= 2p runs (no wide Cox solver exists)", {
  skip_on_cran()  # about 20 seconds
  set.seed(make_seed("oglasso-cox", 1))
  n <- 60; p <- 2500
  x <- matrix(rnorm(n * p), n, p); tt <- rexp(n); st <- rbinom(n, 1, 0.8)
  fw <- oglasso(x, tt, delta = st, group = as.list(1:p), family = "coxph", nlambda = 2,
                irls.maxit = 3L, maxit = 100L)
  expect_equal(dim(fw$beta), c(p + 1L, 2L))
})
