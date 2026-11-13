venn_data <- function(seed, nvars = 8, nobs = 200, family = "gaussian") {
  set.seed(seed)
  genHierSparseData(ncats = 2, nvars = nvars, nobs = nobs, hier.sparsity.param = 0.5,
                    prop.zero.vars = 0.5, family = family)
}

test_that("stratum intercepts stay unpenalized when a variable is constant in a stratum", {
  dat <- venn_data(make_seed("venn_data", 71), nvars = 6)
  x <- dat$x; pat <- apply(dat$group.ind, 1, paste, collapse = "")
  x[pat == "00", 1] <- 0
  f <- vennLasso(x, dat$y, groups = dat$group.ind, nlambda = 5)
  ints <- f$beta[, "(Intercept)", 1]
  ymeans <- tapply(dat$y, pat, mean)[sub(",", "", rownames(f$beta))]
  expect_lt(max(abs(as.vector(ints) - as.vector(ymeans))), 1e-4)
})

test_that("predict uses the right coefficients for subsets of strata and single observations", {
  dat <- venn_data(make_seed("venn_data", 61), nvars = 10, nobs = 150)
  f <- vennLasso(dat$x, dat$y, groups = dat$group.ind, nlambda = 10)
  pat <- apply(dat$group.ind, 1, paste, collapse = "")
  full <- predict(f, newx = dat$x, group.mat = dat$group.ind, s = f$lambda[c(4, 7)])
  for (idx in list(which(pat == "01")[1:3], c(which(pat == "10")[1:2], which(pat == "01")[1]))) {
    sub <- predict(f, newx = dat$x[idx, ], group.mat = dat$group.ind[idx, ], s = f$lambda[c(4, 7)])
    expect_equal(sub, full[idx, , drop = FALSE])
  }
  ## at the lambda values themselves the interpolation is exact
  expect_equal(predict(f, newx = dat$x, group.mat = dat$group.ind, s = f$lambda[c(4, 7)]),
               predict(f, newx = dat$x, group.mat = dat$group.ind)[, c(4, 7)], ignore_attr = TRUE)
})

test_that("cross-validation folds use the lambda sequence of the full fit", {
  dat <- venn_data(make_seed("venn_data", 61), nvars = 10, nobs = 150)
  foldid <- rep(1:4, length.out = nrow(dat$x))
  cvf <- cv.vennLasso(dat$x, dat$y, groups = dat$group.ind, nlambda = 8, foldid = foldid)
  lam <- cvf$vennLasso.fit$lambda
  pred <- matrix(NA, nrow(dat$x), length(lam))
  for (k in 1:4) {
    fk <- vennLasso(dat$x[foldid != k, ], dat$y[foldid != k], groups = dat$group.ind[foldid != k, ], lambda = lam)
    pred[foldid == k, ] <- predict(fk, dat$x[foldid == k, ], group.mat = dat$group.ind[foldid == k, ])
  }
  fold.mse <- sapply(1:4, function(k) colMeans((dat$y[foldid == k] - pred[foldid == k, , drop = FALSE])^2))
  cvm <- rowMeans(fold.mse)
  expect_equal(cvf$cvm, cvm, tolerance = 1e-10)
})

## independent construction of the vennLasso groups for complete data: column j of the
## expanded design is (stratum c, variable i), stratum-major; group (i, c') contains (i, c)
## when c' is c or lies above c in the hierarchy (the null stratum is in its own group only)
venn_groups <- function(combin, P) {
  M <- nrow(combin)
  above <- function(c) if (all(combin[c, ] == 0)) c else which(apply(combin, 1, function(r) all(r >= combin[c, ])))
  members <- vector("list", M * P); k <- 0
  for (i in 1:P) for (cp in 1:M) {
    k <- k + 1
    members[[k]] <- (which(sapply(1:M, function(c) cp %in% above(c))) - 1) * P + i
  }
  members
}

## effective number of parameters from the hat matrix, tr{X_S (X_S'W X_S + N lambda A_S)^{-1} X_S'W}
df_reference <- function(X, b, members, w, lambda, wts = rep(1, nrow(X))) {
  N <- nrow(X)
  nrm <- sapply(members, function(g) sqrt(sum(b[g]^2)))
  A <- numeric(ncol(X))
  for (k in seq_along(members)) if (w[k] > 0) A[members[[k]]] <- A[members[[k]]] + w[k] / nrm[k]
  S <- which((b != 0 & is.finite(A)) | A == 0)
  XS <- X[, S, drop = FALSE]
  hat <- XS %*% solve(crossprod(XS, wts * XS) + N * lambda * diag(A[S], length(S)), t(XS * wts))
  sum(diag(hat))
}

expanded_coef <- function(fit, l) as.vector(t(fit$beta[, , l]))  # stratum-major, complete data

test_that("vennLasso df and AIC follow the paper (exact trace over the active set)", {
  dat <- venn_data(make_seed("venn_data", 81), nvars = 5, nobs = 160)
  for (fam in c("gaussian", "binomial")) {
    y <- if (fam == "gaussian") dat$y else rbinom(length(dat$y), 1, plogis(dat$y - mean(dat$y)))
    f <- vennLasso(dat$x, y, groups = dat$group.ind, family = fam, nlambda = 12, model.matrix = TRUE,
                   abs.tol = 1e-9, rel.tol = 1e-9, maxit = 20000L)
    X <- f$model.matrix; N <- nrow(X); M <- nrow(f$beta); P <- ncol(f$beta)
    expect_false(anyNA(f$beta))
    members <- venn_groups(f$condition.combinations, P)
    w <- sqrt(lengths(members)) * rep(c(0, rep(1, P - 1)), each = M)   # intercept groups unpenalized
    edf <- N - f$df
    expect_equal(edf[1], M, tolerance = 1e-6)                    # stratum means only
    for (l in c(1, 4, 8, 12)) {
      b <- expanded_coef(f, l)
      wts <- if (fam == "binomial") { mu <- plogis(drop(X %*% b)); mu * (1 - mu) } else rep(1, N)
      expect_equal(edf[l], df_reference(X, b, members, w, f$lambda[l], wts), tolerance = 1e-6)
    }
    ll <- if (fam == "gaussian") {
      rss <- sapply(seq_along(f$lambda), function(l) sum((y - X %*% expanded_coef(f, l))^2))
      -0.5 * N * (log(2 * pi) - log(N) + log(rss)) - 0.5 * N
    } else sapply(seq_along(f$lambda), function(l) {
      e <- drop(X %*% expanded_coef(f, l)); sum(y * e - log1p(exp(e))) })
    expect_equal(f$aic, 2 * edf - 2 * ll, tolerance = 1e-4)
    expect_equal(f$bic, log(N) * edf - 2 * ll, tolerance = 1e-4)
  }
  ## with almost no penalty every column counts
  f <- vennLasso(dat$x, dat$y, groups = dat$group.ind, lambda = c(0.1, 1e-9), abs.tol = 1e-12,
                 rel.tol = 1e-12, maxit = 100000L, model.matrix = TRUE)
  expect_equal(nrow(dat$x) - f$df[2], ncol(f$model.matrix), tolerance = 1e-5)
  ## a single shared intercept counts once at lambda.max
  f1 <- vennLasso(dat$x, dat$y + 5, groups = dat$group.ind, nlambda = 5, one.intercept = TRUE)
  expect_equal(nrow(dat$x) - f1$df[1], 1, tolerance = 1e-6)
  expect_lt(abs(f1$intercept[1] - mean(dat$y + 5)), 1e-4)
})

test_that("vennLasso standardize = TRUE matches glmnet on the expanded design (one condition)", {
  skip_if_not_installed("glmnet")
  ## with one binary condition every group is a single coefficient, so vennLasso is a lasso on the
  ## expanded design with unpenalized stratum intercepts
  set.seed(make_seed("venn-standardize"))
  n <- 200; p <- 4
  x <- matrix(rnorm(n * p), n, p) %*% diag(c(1, 10, 0.1, 3))
  grp <- matrix(rbinom(n, 1, 0.5), ncol = 1)
  y <- drop(x %*% c(1, 0.1, 5, 0)) * (1 + grp[, 1]) + rnorm(n)
  lam <- c(0.2, 0.02)
  f <- vennLasso(x, y, groups = grp, lambda = lam, standardize = TRUE, model.matrix = TRUE,
                 abs.tol = 1e-11, rel.tol = 1e-11, maxit = 100000L)
  X <- f$model.matrix
  s <- apply(X, 2, function(v) sqrt(mean((v - mean(v))^2)))
  pf <- rep(c(0, rep(1, p)), times = nrow(f$beta))   # stratum intercepts unpenalized
  cfac <- sum(pf) / length(pf)                          # glmnet rescales penalty.factor to sum to nvars
  g <- glmnet::glmnet(sweep(X, 2, s, "/"), y, intercept = FALSE, standardize = FALSE, penalty.factor = pf,
                      lambda = lam * cfac, thresh = 1e-16, maxit = 1e7)
  for (k in 1:2) expect_equal(expanded_coef(f, k), unname(as.matrix(g$beta)[, k]) / s, tolerance = 1e-5)
  fn <- vennLasso(x, y, groups = grp, lambda = lam, standardize = FALSE)
  expect_gt(max(abs(f$beta - fn$beta)), 1e-3)
})

test_that("binomial confidence intervals give every stratum intercept its own standard error", {
  dat <- venn_data(make_seed("venn_data", 63), nvars = 6, nobs = 300, family = "binomial")
  f <- vennLasso(dat$x, dat$y, groups = dat$group.ind, family = "binomial", nlambda = 5, conf.int = 0.95,
                 model.matrix = TRUE)
  expect_equal(dim(f$beta)[3], 5L)
  expect_true(all(is.finite(f$aic)))
  l <- 3
  X <- f$model.matrix
  b <- expanded_coef(f, l)
  nz <- abs(b) >= sqrt(.Machine$double.eps)
  ref <- glm(dat$y ~ X[, nz] - 1, family = binomial())
  int.cols <- which(nz)[which(nz) %in% ((seq_len(nrow(f$beta)) - 1) * ncol(f$beta) + 1)]
  se.int <- f$se[, "(Intercept)", l]
  expect_false(anyNA(se.int))
  expect_equal(unname(se.int), unname(sqrt(diag(vcov(ref)))[match(int.cols, which(nz))]), tolerance = 1e-6)
  expect_equal(unname(f$beta.refit[, "(Intercept)", l]), unname(coef(ref)[match(int.cols, which(nz))]),
               tolerance = 1e-6)
})

test_that("variables missing for a whole condition are dropped only where they are missing", {
  dat <- venn_data(make_seed("venn_data", 91), nvars = 5, nobs = 150)
  g <- dat$group.ind
  x <- dat$x; x[g[, 1] == 1, 2] <- NA          # not measured when condition 1 is present
  f <- vennLasso(x, dat$y, groups = g, nlambda = 5)
  has1 <- f$condition.combinations[, 1] == 1
  expect_true(all(is.na(f$beta[has1, "V2", 5])))
  expect_false(any(is.na(f$beta[!has1, "V2", 5])))
  x <- dat$x; x[g[, 1] == 0, 2] <- NA          # measured only when condition 1 is present
  f <- vennLasso(x, dat$y, groups = g, nlambda = 5)
  expect_true(all(is.na(f$beta[!has1, "V2", 5])))
  expect_false(any(is.na(f$beta[has1, "V2", 5])))
})

test_that("a single condition can be used", {
  set.seed(make_seed("one-condition"))
  x <- matrix(rnorm(200 * 3), 200, 3); g <- matrix(rbinom(200, 1, 0.5), ncol = 1)
  y <- x[, 1] * (1 + g[, 1]) + rnorm(200)
  f <- vennLasso(x, y, groups = g, nlambda = 5)
  expect_equal(dim(f$beta), c(2L, 4L, 5L))
  expect_equal(dim(predict(f, x, g)), c(200L, 5L))
})

test_that("gaussian refitted coefficients and their standard errors are returned", {
  dat <- venn_data(make_seed("venn_data", 95), nvars = 6, nobs = 200)
  f <- vennLasso(dat$x, dat$y, groups = dat$group.ind, nlambda = 6, conf.int = 0.9, model.matrix = TRUE)
  X <- f$model.matrix
  for (l in c(3, 6)) {
    b <- expanded_coef(f, l)
    nz <- abs(b) >= sqrt(.Machine$double.eps)
    ref <- lm(dat$y ~ X[, nz] - 1)
    refit <- as.vector(t(f$beta.refit[, , l]))
    expect_equal(refit[nz], unname(coef(ref)), tolerance = 1e-6)
    expect_true(all(refit[!nz] == 0))
  }
})
