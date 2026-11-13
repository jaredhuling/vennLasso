## seeds are derived from a tag (and an index) rather than chosen by hand
make_seed <- function(tag, k = 0L) {
  v <- utf8ToInt(tag)
  as.integer((sum(v * seq_along(v)) * 7919 + k * 104729) %% .Machine$integer.max)
}

## objective functions used to check optimality, on the scale used by oglasso():
##   loss / n + lambda * sum_g w_g ||beta_g||
grp_pen <- function(b, groups, w) sum(w * vapply(groups, function(g) sqrt(sum(b[g]^2)), 0))

obj_gaussian <- function(b0, b, x, y, lambda, groups, w)
  sum((y - b0 - drop(x %*% b))^2) / (2 * nrow(x)) + lambda * grp_pen(b, groups, w)

obj_binomial <- function(b0, b, x, y, lambda, groups, w) {
  e <- b0 + drop(x %*% b)
  mean(log1p(exp(e)) - y * e) + lambda * grp_pen(b, groups, w)
}

## Test designs. The reference objective values used in the tests were computed once for
## these designs with CVXR 1.8.2 and the ECOS solver (tolerances 1e-10) by
## inst/reference/make_refs.R; CVXR is not needed to run the tests.

## wide design (p = 2500 > n, so the wide solvers are used); variables 3 to 10 are in two
## groups, and the small window weights make them active
design_wide <- function() {
  set.seed(make_seed("wide-logistic"))
  n <- 50; p <- 2500
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, plogis(0.5 + drop(x[, 3:8] %*% c(4, -4, 3, 3, -2, 2))))
  groups <- c(list(1:4, 3:6, 5:8, 7:10, 9:30), split(31:p, ceiling(seq_along(31:p) / 10)))
  w <- c(rep(0.4, 5), sqrt(lengths(groups))[-(1:5)])
  list(x = x, y = y, groups = groups, w = w, lambda = c(0.1, 0.05))
}

## tall design with an intercept, columns far from mean zero and unequal scales,
## overlapping groups; the problem is fit with standardize = TRUE
design_tall_std <- function() {
  set.seed(make_seed("tall-standardized"))
  n <- 120; p <- 10
  x <- matrix(rnorm(n * p, mean = 3), n, p) %*% diag(c(1, 10, 0.1, 1, 5, 0.5, 2, 3, 1, 1))
  y <- 5 + drop(x[, 1:3] %*% c(1, -0.1, 5)) + rnorm(n)
  groups <- list(1:3, 3:5, 6:10)
  list(x = x, y = y, groups = groups, w = sqrt(lengths(groups)), lambda = c(0.2, 0.05))
}
