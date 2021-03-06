% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/vennLasso.R
\name{vennLasso}
\alias{vennLasso}
\title{Fitting vennLasso models}
\usage{
vennLasso(
  x,
  y,
  groups,
  family = c("gaussian", "binomial"),
  nlambda = 100L,
  lambda = NULL,
  lambda.min.ratio = NULL,
  lambda.fused = NULL,
  penalty.factor = NULL,
  group.weights = NULL,
  adaptive.lasso = FALSE,
  adaptive.fused = FALSE,
  gamma = 1,
  standardize = FALSE,
  intercept = TRUE,
  one.intercept = FALSE,
  compute.se = FALSE,
  conf.int = NULL,
  rho = NULL,
  dynamic.rho = TRUE,
  maxit = 500L,
  abs.tol = 1e-05,
  rel.tol = 1e-05,
  irls.tol = 1e-05,
  irls.maxit = 100L,
  model.matrix = FALSE,
  ...
)
}
\arguments{
\item{x}{input matrix of dimension nobs by nvars. Each row is an observation,
each column corresponds to a covariate}

\item{y}{numeric response vector of length nobs}

\item{groups}{A list of length equal to the number of groups containing vectors of integers
indicating the variable IDs for each group. For example, \code{groups = list(c(1,2), c(2,3), c(3,4,5))} specifies
that Group 1 contains variables 1 and 2, Group 2 contains variables 2 and 3, and Group 3 contains
variables 3, 4, and 5. Can also be a matrix of 0s and 1s with the number of columns equal to the
number of groups and the number of rows equal to the number of variables. A value of 1 in row i and
column j indicates that variable i is in group j and 0 indicates that variable i is not in group j.}

\item{family}{\code{"gaussian"} for least squares problems, \code{"binomial"} for binary response, 
and \code{"coxph"} for time-to-event outcomes (not yet available)}

\item{nlambda}{The number of lambda values. Default is 100.}

\item{lambda}{A user-specified sequence of lambda values. Left unspecified, the a sequence of lambda values is
automatically computed, ranging uniformly on the log scale over the relevant range of lambda values.}

\item{lambda.min.ratio}{Smallest value for lambda, as a fraction of \code{lambda.max}, the (data derived) entry
value (i.e. the smallest value for which all parameter estimates are zero). The default
depends on the sample size \code{nobs} relative to the number of variables \code{nvars}. If
\code{nobs > nvars}, the default is 0.0001, close to zero. If \code{nobs < nvars}, the default
is 0.01. A very small value of \code{lambda.min.ratio} can lead to a saturated fit
when \code{nobs < nvars}.}

\item{lambda.fused}{tuning parameter for fused lasso penalty}

\item{penalty.factor}{vector of weights to be multiplied to the tuning parameter for the
group lasso penalty. A vector of length equal to the number of groups}

\item{group.weights}{A vector of values representing multiplicative factors by which each group's penalty is to
be multiplied. Often, this is a function (such as the square root) of the number of predictors in each group.
The default is to use the square root of group size for the group selection methods.}

\item{adaptive.lasso}{Flag indicating whether or not to use adaptive lasso weights. If set to \code{TRUE} and
\code{group.weights} is unspecified, then this will override \code{group.weights}. If a vector is supplied to group.weights,
then the \code{adaptive.lasso} weights will be multiplied by the \code{group.weights} vector}

\item{adaptive.fused}{Flag indicating whether or not to use adaptive fused lasso weights.}

\item{gamma}{power to raise the MLE estimated weights by for the adaptive lasso. defaults to 1}

\item{standardize}{Should the data be standardized? Defaults to \code{FALSE}.}

\item{intercept}{Should an intercept be fit? Defaults to \code{TRUE}}

\item{one.intercept}{Should a single intercept be fit for all subpopulations instead of one
for each subpopulation? Defaults to \code{FALSE}.}

\item{compute.se}{Should standard errors be computed? If \code{TRUE}, then models are re-fit with no penalization and the standard
errors are computed from the refit models. These standard errors are only theoretically valid for the
adaptive lasso (when \code{adaptive.lasso} is set to \code{TRUE})}

\item{conf.int}{level for confidence intervals. Defaults to \code{NULL} (no confidence intervals). Should be a value between 0 and 1. If confidence
intervals are to be computed, compute.se will be automatically set to \code{TRUE}}

\item{rho}{ADMM parameter. must be a strictly positive value. By default, an appropriate value is automatically chosen}

\item{dynamic.rho}{\code{TRUE}/\code{FALSE} indicating whether or not the rho value should be updated throughout the course of the ADMM iterations}

\item{maxit}{integer. Maximum number of ADMM iterations. Default is 500.}

\item{abs.tol}{absolute convergence tolerance for ADMM iterations for the relative dual and primal residuals.
Default is \code{10^{-5}}, which is typically adequate.}

\item{rel.tol}{relative convergence tolerance for ADMM iterations for the relative dual and primal residuals.
Default is \code{10^{-5}}, which is typically adequate.}

\item{irls.tol}{convergence tolerance for IRLS iterations. Only used if \code{family != "gaussian"}. Default is 10^{-5}.}

\item{irls.maxit}{integer. Maximum number of IRLS iterations. Only used if \code{family != "gaussian"}. Default is 100.}

\item{model.matrix}{logical flag. Should the design matrix used be returned?}

\item{...}{not used}
}
\value{
An object with S3 class "vennLasso"
}
\description{
Fitting vennLasso models
}
\examples{
library(Matrix)

# first simulate heterogeneous data using
# genHierSparseData
set.seed(123)
dat.sim <- genHierSparseData(ncats = 2, nvars = 25,
                             nobs = 200, 
                             hier.sparsity.param = 0.5,
                             prop.zero.vars = 0.5,
                             family = "gaussian")

x          <- dat.sim$x
conditions <- dat.sim$group.ind
y          <- dat.sim$y

true.beta.mat <- dat.sim$beta.mat

fit <- vennLasso(x = x, y = y, groups = conditions)

(true.coef <- true.beta.mat[match(dimnames(fit$beta)[[1]], rownames(true.beta.mat)),])
round(fit$beta[,,21], 2)

## fit adaptive version and compute confidence intervals
afit <- vennLasso(x = x, y = y, groups = conditions, conf.int = 0.95, adaptive.lasso = TRUE)

(true.coef <- true.beta.mat[match(dimnames(fit$beta)[[1]], rownames(true.beta.mat)),])[,1:10]
round(afit$beta[,1:10,28], 2)
round(afit$lower.ci[,1:10,28], 2)
round(afit$upper.ci[,1:10,28], 2)

aic.idx <- which.min(afit$aic)
bic.idx <- which.min(afit$bic)

# actual coverage
# actual coverage
mean(true.coef[afit$beta[,-1,aic.idx] != 0] >= 
             afit$lower.ci[,-1,aic.idx][afit$beta[,-1,aic.idx] != 0] &
         true.coef[afit$beta[,-1,aic.idx] != 0] <= 
             afit$upper.ci[,-1,aic.idx][afit$beta[,-1,aic.idx] != 0])

(covered <- true.coef >= afit$lower.ci[,-1,aic.idx] & true.coef <= afit$upper.ci[,-1,aic.idx])
mean(covered)


# logistic regression example
\dontrun{
set.seed(123)
dat.sim <- genHierSparseData(ncats = 2, nvars = 25,
                             nobs = 200, 
                             hier.sparsity.param = 0.5,
                             prop.zero.vars = 0.5,
                             family = "binomial",
                             effect.size.max = 0.5) # don't make any 
                                                    # coefficients too big

x           <- dat.sim$x
conditions  <- dat.sim$group.ind
y           <- dat.sim$y
true.beta.b <- dat.sim$beta.mat

bfit <- vennLasso(x = x, y = y, groups = conditions, family = "binomial")

(true.coef.b <- -true.beta.b[match(dimnames(fit$beta)[[1]], rownames(true.beta.b)),])
round(bfit$beta[,,20], 2)
}

}
