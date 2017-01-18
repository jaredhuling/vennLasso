

#define EIGEN_DONT_PARALLELIZE

#include "ADMMogLassoFusedTall.h"
#include "ADMMogLassoLogisticFusedTall.h"
#include "ADMMogLassoCoxPHFusedTall.h"
//#include "ADMMogLassoWide.h"
#include "DataStd.h"
#include <RcppNumerical.h>

using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::VectorXf;
using Eigen::VectorXd;
using Eigen::ArrayXf;
using Eigen::ArrayXd;
using Eigen::ArrayXXf;
using Eigen::Map;

using Rcpp::wrap;
using Rcpp::as;
using Rcpp::List;
using Rcpp::Named;
using Rcpp::IntegerVector;
using Rcpp::CharacterVector;

using namespace Numer;

typedef Map<VectorXd> MapVecd;
typedef Map<Eigen::MatrixXd> MapMatd;
typedef Eigen::MappedSparseMatrix<double> MSpMat;
typedef Eigen::SparseVector<float> SpVecf;
typedef Eigen::SparseMatrix<float> SpMatf;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;

inline void write_beta_matrix(SpMat &betas, int col, double beta0, SpVec &coef)
{
    betas.insert(0, col) = beta0;

    for(SpVec::InnerIterator iter(coef); iter; ++iter)
    {
        betas.insert(iter.index() + 1, col) = iter.value();
    }
}


class LogisticRegUnivar: public MFuncGrad
{
private:
    const VectorXd X;
    const VectorXd Y;
    const int n;
    VectorXd xbeta;  // contains X*beta
    VectorXd prob;   // contains log(1+exp(X*beta)) and 1/(1+exp(-X*beta))
public:
    LogisticRegUnivar(const VectorXd x_, const VectorXd y_) :
    X(x_),
    Y(y_),
    n(X.size()),
    xbeta(n),
    prob(n)
    {}

    double f_grad(Constvec& beta, Refvec grad)
    {
        // Negative log likelihood
        //   sum(log(1 + exp(X * beta))) - y' * X * beta


        Eigen::VectorXd xbeta = X * beta(1);
        xbeta.array() += beta(0);
        const double yxbeta = Y.dot(xbeta);
        // X * beta => exp(X * beta)
        xbeta = xbeta.array().exp();
        const double f = (xbeta.array() + 1.0).log().sum() - yxbeta;

        // Gradient
        //   X' * (p - y), p = exp(X * beta) / (1 + exp(X * beta))

        // exp(X * beta) => p
        xbeta.array() /= (xbeta.array() + 1.0);
        VectorXd resid = xbeta - Y;
        grad(0) = resid.sum();
        grad(1) = X.dot(resid);

        return f;
    }
    Eigen::VectorXd current_xb() const { return xbeta; }
    Eigen::VectorXd current_p()  const { return prob; }
};


RcppExport SEXP admm_oglasso_fused_dense(SEXP x_,
                                         SEXP y_,
                                         SEXP delta_,
                                         SEXP group_,
                                         SEXP fused_,
                                         SEXP family_,
                                         SEXP nlambda_,
                                         SEXP lambda_,
                                         SEXP lambda_fused_,
                                         SEXP lambda_min_ratio_,
                                         SEXP group_weights_,
                                         SEXP adaptive_lasso_,
                                         SEXP adaptive_fused_,
                                         SEXP penalty_factor_,
                                         SEXP penalty_factor_fused_,
                                         SEXP gamma_,
                                         SEXP group_idx_,
                                         SEXP ngroups_,
                                         SEXP standardize_,
                                         SEXP intercept_,
                                         SEXP compute_se_,
                                         SEXP opts_)
{
    BEGIN_RCPP

    //Rcpp::NumericMatrix xx(x_);
    //Rcpp::NumericVector yy(y_);

    Rcpp::NumericMatrix xx(x_);
    Rcpp::NumericVector yy(y_);
    Rcpp::NumericVector ddelta(delta_);

    const int n = xx.rows();
    const int p = xx.cols();

    MatrixXd datX(n, p);
    VectorXd datY(n);
    VectorXi delta(n);

    // Copy data and convert type from double to float
    std::copy(xx.begin(), xx.end(), datX.data());
    std::copy(yy.begin(), yy.end(), datY.data());
    std::copy(ddelta.begin(), ddelta.end(), delta.data());

    //MatrixXd datX(as<MatrixXd>(x_));
    //VectorXd datY(as<VectorXd>(y_));

    //const int n = datX.rows();
    //const int p = datX.cols();

    //MatrixXf datX(n, p);
    //VectorXf datY(n);

    // Copy data and convert type from double to float
    //std::copy(xx.begin(), xx.end(), datX.data());
    //std::copy(yy.begin(), yy.end(), datY.data());

    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    ArrayXd lambda(as<ArrayXd>(lambda_));
    int nlambda = lambda.size();
    VectorXd lambda_fused(as<VectorXd>(lambda_fused_));
    int nlambda_fused = lambda_fused.size();

    List opts(opts_);
    const int maxit           = as<int>(opts["maxit"]);
    const int irls_maxit      = as<int>(opts["irls_maxit"]);
    const double irls_tol     = as<double>(opts["irls_tol"]);
    const double eps_abs      = as<double>(opts["eps_abs"]);
    const double eps_rel      = as<double>(opts["eps_rel"]);
    const double rho          = as<double>(opts["rho"]);
    const double dynamic_rho  = as<double>(opts["dynamic_rho"]);
    const double gamma        = as<double>(gamma_);   //power factor for adaptive lasso
    bool standardize          = as<bool>(standardize_);
    bool intercept            = as<bool>(intercept_);
    bool intercept_bin        = intercept;
    bool adaptive_lasso       = as<bool>(adaptive_lasso_);
    const bool adaptive_fused = as<bool>(adaptive_fused_);
    bool compute_se           = compute_se_;


    const SpMat group(as<MSpMat>(group_));
    const SpMat fused(as<MSpMat>(fused_));
    CharacterVector family(as<CharacterVector>(family_));
    // group_weights not MapVec or const, because
    // we need to change it later if agaptive_lasso == true
    VectorXd group_weights(as<VectorXd>(group_weights_));
    VectorXd penalty_factor(as<VectorXd>(penalty_factor_));
    VectorXd penalty_factor_fused(as<VectorXd>(penalty_factor_fused_));
    IntegerVector group_idx(group_idx_);

    const int ngroups(as<int>(ngroups_));
    VectorXd adaptive_weights(ngroups);
    adaptive_weights.setZero();

    // don't standardize if not linear model.
    // fit intercept the dumb way if it is wanted
    bool fullbetamat = false;
    int add = 0;
    if (family(0) != "coxph") // family(0) != "gaussian" &
    {
        standardize = false;
        intercept = false;

        // this will do the dumb thing
        // and attach a column of 1s
        // for the intercept
        if (intercept_bin)
        {
            fullbetamat = true;
            add = 1;

            VectorXd v(n);
            v.fill(1);
            MatrixXd datX_tmp(n, p+1);

            datX_tmp << v, datX;
            datX.swap(datX_tmp);

            datX_tmp.resize(0,0);
        }
    }



    // total size of all groups
    const int M(group.sum());

    // number of fused penalties
    const int nfused(fused.rows());

    VectorXd adaptive_fused_weights(nfused);
    adaptive_fused_weights.fill(1);

    // create C matrix
    //   C_{i,j} = 1 if y_i is a replicate of x_j
    //           = 0 otherwise
    Eigen::SparseMatrix<double,Eigen::RowMajor> C(Eigen::SparseMatrix<double,Eigen::RowMajor>(M, p + add));
    C.reserve(VectorXi::Constant(M,1));
    createC(C, group, M);


    DataStd<double> datstd(n, p + add, standardize, intercept);
    datstd.standardize(datX, datY);

    FADMMBaseTwo<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> *solver_tall = NULL; // obj doesn't point to anything yet
    FADMMBaseTwo<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> *solver_tall_2 = NULL; // obj doesn't point to anything yet

    //ADMMogLassoTall *solver_tall;
    //ADMMogLassoWide *solver_wide;


    for (int g = 0; g < ngroups; ++g)
    {
        // multiply group weights by penalty factor
        group_weights(g) *= penalty_factor(g);
    }
    for (int ff = 0; ff < nfused; ++ff)
    {
        // multiply fused weights by penalty factor
        adaptive_fused_weights(ff) *= penalty_factor_fused(ff);
    }



    if(n > 2 * p || p < 4500)
    {

        if (family(0) == "gaussian")
        {
            solver_tall = new ADMMogLassoFusedTall(datX, datY, C, fused, n, p, M,
                                                   ngroups, nfused,
                                                   family,
                                                   group_weights,
                                                   adaptive_fused_weights,
                                                   group_idx, dynamic_rho,
                                                   irls_tol, irls_maxit,
                                                   eps_abs, eps_rel);
        } else if (family(0) == "binomial")
        {
            solver_tall = new ADMMogLassoLogisticFusedTall(datX, datY, C, fused, n, p + add, M,
                                                           ngroups, nfused,
                                                           family,
                                                           group_weights,
                                                           adaptive_fused_weights,
                                                           group_idx, dynamic_rho,
                                                           irls_tol, irls_maxit,
                                                           eps_abs, eps_rel);
        } else if (family(0) == "coxph")
        {
            solver_tall = new ADMMogLassoCoxPHFusedTall(datX, datY, delta, C, fused, n, p, M,
                                                        ngroups, nfused,
                                                        family,
                                                        group_weights,
                                                        adaptive_fused_weights,
                                                        group_idx, dynamic_rho,
                                                        irls_tol, irls_maxit,
                                                        eps_abs, eps_rel);
        }
    }
    else
    {

        /*
        if (family(0) == "gaussian")
        {
        solver_wide = new ADMMogLassoTallWide(datX, datY, C, n, p, M, ngroups,
                                              family, group_weights, group_idx,
        dynamic_rho, irls_tol, irls_maxit,
        eps_abs, eps_rel);
        } else if (family(0) == "binomial")
        {
        solver_wide = new ADMMogLassoLogisticWide(datX, datY, C, n, p, M, ngroups,
                                                  family, group_weights, group_idx,
        dynamic_rho, irls_tol, irls_maxit,
        eps_abs, eps_rel);
        }
        */
    }


    if(nlambda < 1)
    {
        double lmax = 0.0;
        if(n > 2 * p || p < 4500)
        {
            lmax = solver_tall->get_lambda_zero() / n * datstd.get_scaleY();
        }
        else
        {
            //lmax = solver_wide->get_lambda_zero() / n * datstd.get_scaleY();
        }

        double lmin = as<double>(lambda_min_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }

    VectorXd loss(nlambda);

    MatrixXd beta(p + 1, nlambda);
    IntegerVector niter(nlambda);
    double ilambda  = 0.0;
    double ilambdaF = 0.0;

    // if adaptive_lasso is specified, then call the solver
    // and estimate a model with no penalty applied

    VectorXd beta_unpen;

    if (adaptive_lasso || adaptive_fused)
    {
        VectorXd d, fu;
        ilambda = 1e-10 * lambda[0] * n / datstd.get_scaleY();
        if(n > 2 * p || p < 4500)
        {
            if (p <= n)
            {
                solver_tall->init(ilambda, rho, ilambda);
                int itertmp = solver_tall->solve(maxit);
                //VectorXd res = solver_tall->get_aux_gamma();
                VectorXd restmp = solver_tall->get_gamma();

                // if the design matrix includes the intercept
                // then don't back into the intercept with
                // datastd and include it to beta directly.
                VectorXd adaptive_weights(ngroups);
                if (fullbetamat)
                {
                    double beta0;
                    //datstd.recover(beta0, res);
                    //d = res.tail(p);

                    //datstd.recover(beta0, restmp);
                    beta_unpen = restmp; //.tail(p);
                } else
                {
                    double beta0;
                    //datstd.recover(beta0, res);
                    //d = res;
                    //datstd.recover(beta0, restmp);
                    beta_unpen = restmp;
                }
                double maxbetaval = beta_unpen.array().abs().maxCoeff();

                // if the MLEs are divergent then use
                // a linear probability model
                if (maxbetaval > 15)
                {
                    solver_tall_2 = new ADMMogLassoFusedTall(datX, datY, C, fused, n, p, M,
                                                             ngroups, nfused,
                                                             family,
                                                             group_weights,
                                                             adaptive_fused_weights,
                                                             group_idx, dynamic_rho,
                                                             irls_tol, irls_maxit,
                                                             eps_abs, eps_rel);

                    solver_tall_2->init(ilambda, rho, ilambda);
                    int itertmp = solver_tall_2->solve(maxit);
                    //VectorXd res = solver_tall->get_aux_gamma();
                    beta_unpen = solver_tall_2->get_gamma();

                    delete solver_tall_2;
                }


            } else
            {
                // marginal regression for adaptive weights
                if (family(0) == "gaussian")
                {
                    double ysum = datY.sum();
                    VectorXd xsums  = datX.colwise().sum();
                    VectorXd xsumsq = datX.array().square().colwise().sum();
                    for (int jj = 0; jj < datX.cols(); ++jj)
                    {
                        double xysum = (datX.col(jj).array() * datY.array()).sum();
                        double xsqdiff = (xsumsq(jj) - std::pow(xsums(jj), 2) /n );
                        if (xsqdiff <= 1e-14 && xsqdiff >= -1e-14)
                        {
                            beta_unpen(jj) = 0;
                        } else
                        {
                            beta_unpen(jj) = (xysum - xsums(jj) * ysum / n) / xsqdiff;
                        }
                    }
                } else if (family(0) == "binomial")
                {
                    for (int jj = 0; jj < datX.cols(); ++jj)
                    {
                        LogisticRegUnivar nll(datX.col(jj), datY);

                        VectorXd betaunivar(2);
                        betaunivar.fill(0.5);
                        int maxitunivar = 50;
                        double eps_f = 1e-8;
                        double eps_g = 1e-5;

                        double fopt;
                        int status = optim_lbfgs(nll, betaunivar, fopt, maxitunivar, eps_f, eps_g);
                        if(status < 0)
                            Rcpp::warning("algorithm did not converge");

                        beta_unpen(jj) = betaunivar(1);

                    }
                } else if (family(0) == "coxph")
                {

                }
            }

            if (adaptive_lasso)
            {
                d  = C * beta_unpen;
            }
            if (adaptive_fused)
            {
                fu = (fused * beta_unpen).array().abs();
            }

        } else
        {
            // reserved for wide data case

            // marginal regression for adaptive weights
            if (family(0) == "gaussian")
            {
                double ysum = datY.sum();
                VectorXd xsums  = datX.colwise().sum();
                VectorXd xsumsq = datX.array().square().colwise().sum();
                for (int jj = 0; jj < datX.cols(); ++jj)
                {
                    double xysum = (datX.col(jj).array() * datY.array()).sum();

                    double xsqdiff = (xsumsq(jj) - std::pow(xsums(jj), 2) /n );
                    if (xsqdiff <= 1e-14 && xsqdiff >= -1e-14)
                    {
                        beta_unpen(jj) = 0;
                    } else
                    {
                        beta_unpen(jj) = (xysum - xsums(jj) * ysum / n) / xsqdiff;
                    }
                }

            } else if (family(0) == "binomial")
            {
                for (int j = 0; j < datX.cols(); ++j)
                {
                    LogisticRegUnivar nll(datX.col(j), datY);

                    VectorXd betaunivar(2);
                    betaunivar.fill(0.5);
                    int maxitunivar = 50;
                    double eps_f = 1e-8;
                    double eps_g = 1e-5;

                    double fopt;
                    int status = optim_lbfgs(nll, betaunivar, fopt, maxitunivar, eps_f, eps_g);
                    if(status < 0)
                        Rcpp::warning("algorithm did not converge");

                    beta_unpen(j) = betaunivar(1);
                }
            } else if (family(0) == "coxph")
            {

            }
        }

        if (adaptive_lasso)
        {
            for (int g = 0; g < ngroups; ++g)
            {
                double norm_group = std::pow( (d.segment(group_idx(g), group_idx(g+1) - group_idx(g))).norm(), gamma );
                if (norm_group < 1e-5)
                {
                    norm_group = 1e-5;
                } else if (norm_group > 1e5)
                {
                    norm_group = 1e5;
                }
                double ada_wt = 1 / norm_group;
                adaptive_weights(g) = ada_wt;
                group_weights(g) *= ada_wt; // multiply weights by adaptive lasso weights
            }
            if(n > 2 * p || p < 4500)
            {
                solver_tall->update_adaptive_group_weights(group_weights);
            }

        }
        if (adaptive_fused)
        {
            for (int ff = 0; ff < nfused; ++ff)
            {
                double abs_diff = std::pow( fu(ff), gamma );
                if (abs_diff < 1e-5)
                {
                    abs_diff = 1e-5;
                } else if (abs_diff > 1e5)
                {
                    abs_diff = 1e5;
                }
                double ada_wt = 1 / abs_diff;
                adaptive_fused_weights(ff) *= ada_wt;
            }

            if(n > 2 * p || p < 4500)
            {
                solver_tall->update_adaptive_fused_weights(adaptive_fused_weights);
            }
        }

    }

    //   =======================================   //
    //         End Adaptive Lasso Section
    //   =======================================   //

    for (int j = 0; j < nlambda_fused; j++)
    {
        ilambdaF = lambda_fused[j] * n / datstd.get_scaleY();
        for(int i = 0; i < nlambda; i++)
        {
            ilambda  = lambda[i] * n / datstd.get_scaleY();

            if(n > 2 * p || p < 4500)
            {
                if(i == 0)
                    solver_tall->init(ilambda, rho, ilambdaF);
                else
                    solver_tall->init_warm(ilambda, ilambdaF);

                niter[i] = solver_tall->solve(maxit);

                // get computed beta
                VectorXd res = solver_tall->get_gamma();

                // get associated loss
                loss(i) = solver_tall->get_loss();

                double beta0 = 0.0;

                // if the design matrix includes the intercept
                // then don't back into the intercept with
                // datastd and include it to beta directly.
                if (fullbetamat)
                {
                    datstd.recover(beta0, res);
                    beta.block(0, i, p+1, 1) = res;
                } else
                {
                    datstd.recover(beta0, res);
                    beta(0,i) = beta0;
                    beta.block(1, i, p, 1) = res;
                }



            } else {
                /*
                if(i == 0)
                solver_wide->init(ilambda, rho);
                else
                solver_wide->init_warm(ilambda);

                niter[i] = solver_wide->solve(maxit);
                SpVec res = solver_wide->get_x();
                double beta0 = 0.0;
                if (!fullbetamat)
                {
                datstd.recover(beta0, res);
                }
                write_beta_matrix(beta, i, beta0, res);
                */
            }
        }
    }

    //   =======================================   //
    //           End Estimation Section
    //   =======================================   //

    MatrixXd XX; // hessian (XtX or XtWX)
    VectorXd XY; // XtY

    // need to deallocate dynamic object
    if(n > 2 * p || p < 4500)
    {
        XX = solver_tall->get_hessian();
        XY = solver_tall->get_xty();
        delete solver_tall;
    }
    else
    {
        //delete solver_wide;
    }


    return List::create(Named("lambda")                 = lambda,
                        Named("nlambda")                = nlambda,
                        Named("beta")                   = beta,
                        Named("niter")                  = niter,
                        Named("adaptive.weights")       = adaptive_weights,
                        Named("group.weights")          = group_weights,
                        Named("adaptive.fused.weights") = adaptive_fused_weights,
                        Named("beta.unpenalized")       = beta_unpen,
                        Named("loss")                   = loss,
                        Named("XtX")                    = XX,
                        Named("XtY")                    = XY);

    END_RCPP
}






