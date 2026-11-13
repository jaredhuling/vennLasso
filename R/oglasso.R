
#-------------------------------------------------------------------------------
# Estimation of the overlapping group lasso for
# least squares and generalized linear models
#
# August 2015
# Jared Huling
#-------------------------------------------------------------------------------



#' Overlapping Group Lasso (OGLasso)
#'
#' Fits the overlapping group lasso, which minimizes
#' \deqn{\ell(\beta_0, \beta)/n + \lambda \sum_{g} w_g \|\beta_{G_g}\|_2}{loss/n + lambda * sum_g w_g ||beta_{G_g}||_2}
#' over a path of values of \eqn{\lambda}, where the groups \eqn{G_g} may overlap, \eqn{\ell} is the
#' residual sum of squares divided by 2 (\code{family = "gaussian"}), the negative log-likelihood
#' (\code{family = "binomial"}), or the negative Breslow partial log-likelihood (\code{family = "coxph"}),
#' and \eqn{w_g} is the product of \code{group.weights} and \code{penalty.factor} (and of the
#' adaptive weights when \code{adaptive.lasso = TRUE}).
#'
#' @param x input matrix of dimension nobs by nvars. Each row is an observation,
#' each column corresponds to a covariate. A sparse matrix is converted to a dense matrix.
#' @param y numeric response vector of length nobs. For \code{family = "binomial"} it must take the values 0 and 1;
#' for \code{family = "coxph"} it holds the observed times.
#' @param delta vector of length equal to the number of observations with values in 1 and 0, where a 1 indicates the
#' observed time is a death and a 0 indicates the observed time is a censoring event. Required for \code{family = "coxph"}.
#' @param group A list of length equal to the number of groups containing vectors of integers
#' indicating the variable IDs for each group. For example, \code{group = list(c(1,2), c(2,3), c(3,4,5))} specifies
#' that Group 1 contains variables 1 and 2, Group 2 contains variables 2 and 3, and Group 3 contains
#' variables 3, 4, and 5. Can also be a matrix of 0s and 1s with the number of columns equal to the
#' number of groups and the number of rows equal to the number of variables. A value of 1 in row i and
#' column j indicates that variable i is in group j and 0 indicates that variable i is not in group j.
#' Variables that belong to no group are left unpenalized.
#' @param fused not yet implemented; must be \code{NULL}.
#' @param family \code{"gaussian"} for least squares problems, \code{"binomial"} for binary response,
#' and \code{"coxph"} for time-to-event outcomes (Cox proportional hazards model with the Breslow
#' treatment of tied times). The Cox solver forms the nvars x nvars matrix \eqn{X'WX}, which needs
#' 8 * nvars^2 bytes (3.2 GB for 20000 variables)
#' @param nlambda The number of lambda values. Default is 100.
#' @param lambda A user-specified sequence of lambda values. Left unspecified, the a sequence of lambda values is
#' automatically computed, ranging uniformly on the log scale over the relevant range of lambda values.
#' @param lambda.min.ratio Smallest value for lambda, as a fraction of lambda.max, the (data derived) entry
#' value. lambda.max is computed for the problem in which the intercept and any unpenalized variables are
#' fit, and every coefficient in a penalized group is zero at lambda.max. With overlapping groups,
#' lambda.max is a value at which the zero solution is guaranteed, which can be somewhat larger than the
#' smallest such value. The default
#' depends on the sample size nobs relative to the number of variables nvars. If
#' \code{nobs > nvars}, the default is 0.0001, close to zero. If \code{nobs < nvars}, the default
#' is 0.05. A very small value of \code{lambda.min.ratio} will lead to a saturated fit in
#' the \code{nobs < nvars} case.
#' @param lambda.fused not used (the fused penalty is not yet implemented)
#' @param alpha currently not used.
#' @param penalty.factor vector of weights to be multiplied to the tuning parameter for the
#' group lasso penalty. A vector of length equal to the number of groups
#' @param penalty.factor.fused not used (the fused penalty is not yet implemented)
#' @param group.weights A vector of values representing multiplicative factors by which each group's penalty is to
#' be multiplied, one value per group. Often, this is a function (such as the square root) of the number of predictors in each group.
#' The default is to use the square root of group size for the group selection methods.
#' @param adaptive.lasso Flag indicating whether or not to use adaptive lasso weights. If set to \code{TRUE} and
#' \code{group.weights} is unspecified, then this will override \code{group.weights}. If a vector is supplied to group.weights,
#' then the \code{adaptive.lasso} weights will be multiplied by the \code{group.weights} vector
#' @param adaptive.fused not used (the fused penalty is not yet implemented)
#' @param gamma power to raise the MLE estimated weights by for the adaptive lasso. defaults to 1
#' @param standardize Logical flag for \code{x} variable standardization, prior to fitting the models.
#' Each column of \code{x} is divided by its standard deviation (computed with divisor nobs) and,
#' when an intercept is fit, centered at its mean; constant columns are not scaled.
#' The coefficients are always returned on the original scale. Default is \code{standardize = TRUE}. If
#' variables are in the same units already, you might not wish to standardize. The response is never
#' centered or scaled.
#' @param intercept Should intercept(s) be fitted (\code{default = TRUE}) or set to zero (\code{FALSE}).
#' Ignored (no intercept) for \code{family = "coxph"}.
#' @param compute.se not used by \code{oglasso()}; see \code{\link{vennLasso}}
#' @param rho ADMM parameter. must be a strictly positive value. By default, an appropriate value is automatically chosen
#' @param dynamic.rho \code{TRUE}/\code{FALSE} indicating whether or not the rho value should be updated throughout the course of the ADMM iterations
#' @param maxit integer. Maximum number of ADMM iterations. Default is 500.
#' @param abs.tol absolute convergence tolerance for ADMM iterations for the relative dual and primal residuals.
#' Default is \code{10^{-5}}, which is typically adequate.
#' @param rel.tol relative convergence tolerance for ADMM iterations for the relative dual and primal residuals.
#' Default is \code{10^{-5}}, which is typically adequate.
#' @param irls.maxit integer. Maximum number of IRLS iterations. Only used if \code{family != "gaussian"}. Default is 100.
#' @param irls.tol convergence tolerance for IRLS iterations. Only used if \code{family != "gaussian"}. Default is \code{10^{-5}}.
#' @param dfmax limit on the number of nonzero coefficients (not counting the intercept). The path is
#' stopped, with a warning, at the first lambda at which more than \code{dfmax} coefficients are nonzero,
#' and only the solutions for the larger values of lambda are returned. The default \code{NULL} places
#' no limit, so the whole path is computed.
#' @return An object with S3 class "oglasso", a list including
#' \item{beta}{matrix of coefficients on the original scale of \code{x}, with one row for the intercept
#' (zero when no intercept is fit) followed by one row per variable, and one column per value of lambda}
#' \item{lambda}{the values of lambda used}
#' \item{loss}{the loss (residual sum of squares for gaussian, negative log-likelihood for binomial) at each lambda}
#' \item{niter}{number of iterations used for each value of lambda}
#' \item{group.weights}{the group weights used, including penalty factors and adaptive weights}
#' @export
#' @examples
#' library(vennLasso)
#'
#' set.seed(123)
#' n.obs <- 200
#' n.vars <- 50
#'
#' true.beta <- c(rep(0,2), 1, -1, rep(0, 8), 0.5, -0.5, 1, rep(0, 35))
#'
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 3) + drop(x %*% true.beta)
#'
#' groups <- c(list(c(1,2), c(2,3), c(3,4,5), 5:10, 6:12, 7:15), lapply(16:50, function(x) x))
#'
#' fit <- oglasso(x = x, y = y, group = groups, nlambda = 25)
#' round(fit$beta[1:16, c(1, 10, 25)], 3)
oglasso <- function(x, y,
                    delta                = NULL,
                    group,
                    fused                = NULL,
                    family               = c("gaussian", "binomial", "coxph"),
                    nlambda              = 100L,
                    lambda               = NULL,
                    lambda.min.ratio     = NULL,
                    lambda.fused         = 0,
                    alpha                = NULL, # not used currently
                    group.weights        = NULL,
                    adaptive.lasso       = FALSE,
                    adaptive.fused       = FALSE,
                    penalty.factor       = NULL,
                    penalty.factor.fused = NULL,
                    gamma                = 1,
                    standardize          = TRUE,
                    intercept            = TRUE,
                    compute.se           = FALSE,
                    rho                  = NULL,
                    dynamic.rho          = TRUE,
                    maxit                = 500L,
                    abs.tol              = 1e-5,
                    rel.tol              = 1e-5,
                    irls.tol             = 1e-5,
                    irls.maxit           = 100L,
                    dfmax                = NULL)
{

    family <- match.arg(family)
    this.call = match.call()

    if (!is.null(fused))
    {
        stop("the fused lasso penalty ('fused') is not implemented in this version of vennLasso")
    }

    ## there is no sparse-matrix solver; work with a dense copy
    if (inherits(x, "Matrix")) x <- as.matrix(x)
    x <- as.matrix(x)
    if (!is.numeric(x)) stop("x must be a numeric matrix")
    storage.mode(x) <- "double"
    if (anyNA(x)) stop("x contains missing values")

    nvars <- ncol(x)
    nobs <- nrow(x)
    y <- drop(y)
    dimy <- dim(y)
    leny <- ifelse(is.null(dimy), length(y), dimy[1])
    stopifnot(leny == nobs)
    if (!is.null(dimy)) stop("y must be a vector")
    y <- as.double(y)
    if (anyNA(y)) stop("y contains missing values")

    if (family == "binomial" && !all(y %in% c(0, 1)))
    {
        stop("for family = 'binomial', y must take the values 0 and 1")
    }

    if (family == "coxph")
    {
        if (is.null(delta)) stop("delta must be provided for coxph")
        delta <- as.double(drop(delta))
        if (length(delta) != nobs) stop("delta must have one value per observation")
        if (anyNA(delta) || !all(delta %in% c(0, 1))) stop("delta must take the values 0 and 1")
        if (sum(delta) < 1) stop("all observations are censored; not permitted for coxph")
        ## the Cox model has no intercept
        intercept <- FALSE

        ## the solver needs the observations sorted by time; the
        ## coefficients do not depend on the order of the rows
        yorder <- order(y)
        x      <- x[yorder, , drop = FALSE]
        y      <- y[yorder]
        delta  <- delta[yorder]
        ## first row (0-based) tied with each row: Breslow risk sets
        tiestart <- as.integer(match(y, y) - 1L)
    } else
    {
        delta    <- rep(0, nobs)
        tiestart <- integer(nobs)
    }

    if(isTRUE(rho <= 0))
    {
        stop("rho should be positive")
    }

    if (missing(group)) {
        stop("Must specify group structure.")
    } else if (is.matrix(group) || inherits(group, "Matrix")) {
        if (nrow(group) != nvars) {
            stop("group matrix must have one row per variable, i.e. nrow(group) = ncol(x).")
        }
        group <- asDgCMatrix(group)
    } else if (is.list(group) && !is.data.frame(group)) {
        grp.vars <- unlist(group)
        if (length(grp.vars) && (!is.numeric(grp.vars) || any(grp.vars != round(grp.vars)) ||
                                 any(grp.vars < 1) || any(grp.vars > nvars)))
        {
            stop("the elements of group must be variable indices between 1 and ncol(x)")
        }
        tmp.vc <- numeric(nvars)
        group <- asDgCMatrix(vapply(group, function(x) {tmp.vc[x] <- 1; return(tmp.vc) }, tmp.vc))
    } else {
        stop("Supply either a list or matrix. No data frames allowed.")
    }
    group <- Matrix::drop0(group)
    if (length(group@x) && any(group@x != 1))
    {
        stop("group matrix must contain only 0s and 1s")
    }
    ngroups.user <- ncol(group)

    # add unpenalized group if intercept wanted
    if (intercept)
    {
        group  <- asDgCMatrix(rbind(0, group))
    }
    add <- 1L * intercept

    # Seek for variables which were not
    # included in any group and add them
    # to a final group which will be
    # unpenalized.
    rSg <- Matrix::rowSums(group) == 0
    addZeroGroup <- any(rSg)
    if (addZeroGroup) {
        group <- asDgCMatrix(cbind(group, (1 * rSg)))
    }
    group.idx <- as.integer(c(0, cumsum(Matrix::colSums(group))))
    ngroups <- ncol(group)

    if (!is.null(group.weights)) {
        group.weights <- as.double(group.weights)
        if (length(group.weights) == ngroups.user) {
            if (addZeroGroup) group.weights <- c(group.weights, 0)
        } else if (!(addZeroGroup && length(group.weights) == ngroups)) {
            stop("group.weights must have one value per group")
        }
        if (anyNA(group.weights) || any(group.weights < 0)) stop("group.weights must be non-negative")
    } else {
        group.weights <- sqrt(Matrix::colSums(group))
    }

    if (is.null(penalty.factor))
    {
        penalty.factor <- rep(1, ngroups.user)
    }
    penalty.factor <- as.double(penalty.factor)
    if (length(penalty.factor) != ngroups.user)
        stop("penalty.factor must have length equal to the number of groups")
    if (anyNA(penalty.factor) || any(penalty.factor < 0)) stop("penalty.factor must be non-negative")

    if (addZeroGroup)
    {
        # Do not penalize variables in last group
        # which represents the group of variables
        # that were not put into any groups.
        # this will override any adaptive lasso
        # weighting later on (which is what we want)
        group.weights[ngroups] <- 0
        penalty.factor <- c(penalty.factor, 0)
    }

    varnames <- colnames(x)
    if (is.null(varnames)) {
        varnames <- paste("V", seq(nvars), sep = "")
    }

    ## standardization: scale each column by its sd (divisor n) and, when an
    ## intercept is fit, center it; the coefficients are mapped back below
    std.center <- rep(0, nvars)
    std.scale  <- rep(1, nvars)
    if (standardize)
    {
        cm <- colMeans(x)
        std.scale <- sqrt(colMeans(sweep(x, 2, cm) ^ 2))
        const.col <- !(std.scale > 1e-12 * pmax(1, abs(cm)))
        std.scale[const.col] <- 1
        if (intercept) std.center <- cm
        x <- sweep(sweep(x, 2, std.center), 2, std.scale, "/")
    }

    if(is.null(lambda.min.ratio))
    {
        lambda.min.ratio <- ifelse(nrow(x) < ncol(x), 0.05, 0.0001)
    }

    if (is.null(dfmax))
    {
        dfmax <- -1L
    } else
    {
        dfmax <- as.integer(dfmax)[1]
        if (is.na(dfmax) || dfmax < 0) stop("dfmax must be a non-negative integer")
    }

    gamma       <- as.double(gamma)
    irls.tol    <- as.double(irls.tol)
    abs.tol     <- as.double(abs.tol)
    rel.tol     <- as.double(rel.tol)
    dynamic.rho <- as.logical(dynamic.rho)
    irls.maxit  <- as.integer(irls.maxit)
    maxit       <- as.integer(maxit)
    rho         <- if(is.null(rho))  -1.0  else  as.numeric(rho)

    adaptive.lasso <- as.logical(adaptive.lasso)
    adaptive.fused <- as.logical(adaptive.fused)

    if (is.null(lambda)) {
        if (lambda.min.ratio >= 1 | lambda.min.ratio <= 0) {
            stop("lambda.min.ratio should be less than 1 and greater than 0")
        }
        lambda.min.ratio <- as.double(lambda.min.ratio)
        nlambda <- as.integer(nlambda)[1]
        if (nlambda < 1) stop("nlambda must be at least 1")
        lambda <- numeric(0L)
    } else {
        if (any(lambda < 0)) {
            stop("lambdas should be non-negative")
        }
        lambda <- as.double(rev(sort(lambda)))
        nlambda <- as.integer(length(lambda))
    }

    ## gradient of the per-observation loss at the fit of the unpenalized part
    ## (intercept, ungrouped variables, zero-weight groups); used for lambda.max
    xdesign  <- if (intercept) cbind(1, x) else x
    eff.wts  <- group.weights * penalty.factor
    unpen    <- which(Matrix::rowSums(group[, eff.wts > 0, drop = FALSE]) == 0)
    null.grad <- nullGradient(xdesign, y, delta, tiestart, unpen, family)

    opts <- list(maxit       = maxit,
                 eps_abs     = abs.tol,
                 eps_rel     = rel.tol,
                 rho         = rho,
                 dynamic_rho = dynamic.rho,
                 irls_maxit  = irls.maxit,
                 irls_tol    = irls.tol)

    fit <- .Call("admm_oglasso_dense",
                 x,
                 y,
                 delta,
                 group,
                 family,
                 nlambda,
                 lambda,
                 lambda.min.ratio,
                 group.weights,
                 adaptive.lasso,
                 penalty.factor,
                 gamma,
                 group.idx,
                 ngroups,
                 FALSE,       # standardization is done above
                 intercept,
                 compute.se,
                 opts,
                 as.double(null.grad),
                 dfmax,
                 tiestart,
                 PACKAGE = "vennLasso")

    nkeep <- fit$nkeep
    if (nkeep < length(fit$lambda))
    {
        warning(sprintf(paste0("the lambda path was stopped after %d of %d values because more than ",
                               "dfmax = %d coefficients are nonzero at lambda = %.4g"),
                        nkeep, length(fit$lambda), dfmax, fit$lambda[nkeep + 1]), call. = FALSE)
    }

    fit$beta    <- fit$beta[, seq_len(nkeep), drop = FALSE]
    fit$lambda  <- fit$lambda[seq_len(nkeep)]
    fit$loss    <- fit$loss[seq_len(nkeep)]
    fit$niter   <- fit$niter[seq_len(nkeep)]
    fit$nlambda <- nkeep

    ## map coefficients (and X'X, X'y) back to the original scale of x
    if (standardize)
    {
        fit$beta[-1, ] <- fit$beta[-1, , drop = FALSE] / std.scale
        if (intercept)
        {
            fit$beta[1, ] <- fit$beta[1, ] - colSums(fit$beta[-1, , drop = FALSE] * std.center)
        }
        if (adaptive.lasso)
        {
            bu <- fit$beta.unpenalized
            bu[add + seq_len(nvars)] <- bu[add + seq_len(nvars)] / std.scale
            if (intercept) bu[1] <- bu[1] - sum(bu[-1] * std.center)
            fit$beta.unpenalized <- bu
        }
        ## x_original = x_standardized %*% B
        B <- diag(std.scale, nrow = nvars)
        if (intercept) B <- rbind(c(1, std.center), cbind(0, B))
        if (is.matrix(fit$XtX) && nrow(fit$XtX) == nvars + add && ncol(fit$XtX) == nvars + add)
        {
            fit$XtX <- crossprod(B, fit$XtX %*% B)
            fit$XtY <- drop(crossprod(B, fit$XtY))
        }
    }
    rownames(fit$beta) <- c("(Intercept)", varnames)

    fit$fused        <- NULL
    fit$standardized <- standardize
    fit$intercept    <- intercept
    fit$dfmax        <- if (dfmax < 0) NULL else dfmax
    fit$family       <- family
    fit$call         <- this.call
    class(fit)       <- "oglasso"
    fit
}

## coerce a matrix or Matrix to a general numeric sparse matrix (dgCMatrix)
asDgCMatrix <- function(m)
{
    m <- methods::as(m, "CsparseMatrix")
    m <- methods::as(m, "generalMatrix")
    methods::as(m, "dMatrix")
}

## gradient of the per-observation loss with respect to the coefficients of the
## columns of x, evaluated at the fit that uses only the columns 'unpen'
## (all other coefficients zero). For coxph the rows are sorted by time and
## tiestart gives the first (0-based) row tied with each row.
nullGradient <- function(x, y, delta, tiestart, unpen, family)
{
    nobs <- nrow(x)
    if (family == "gaussian")
    {
        resid <- if (length(unpen)) qr.resid(qr(x[, unpen, drop = FALSE]), y) else y
        score <- resid
    } else if (family == "binomial")
    {
        if (length(unpen))
        {
            mu <- suppressWarnings(stats::glm.fit(x[, unpen, drop = FALSE], y,
                                                  family = stats::binomial(),
                                                  intercept = FALSE)$fitted.values)
        } else
        {
            mu <- rep(0.5, nobs)
        }
        score <- y - mu
    } else
    {
        eta <- rep(0, nobs)
        if (length(unpen))
        {
            xu  <- x[, unpen, drop = FALSE]
            cf  <- suppressWarnings(stats::coef(survival::coxph(survival::Surv(y, delta) ~ xu,
                                                                ties = "breslow")))
            cf[is.na(cf)] <- 0
            eta <- drop(xu %*% cf)
        }
        w     <- exp(eta - max(eta))
        rsum  <- rev(cumsum(rev(w)))          # sum of w over rows >= i
        ev    <- which(delta == 1)
        start <- tiestart[ev] + 1L
        inc   <- numeric(nobs)
        agg   <- rowsum(1 / rsum[start], start)
        inc[as.integer(rownames(agg))] <- agg[, 1]
        score <- delta - w * cumsum(inc)
    }
    drop(crossprod(x, score)) / nobs
}
