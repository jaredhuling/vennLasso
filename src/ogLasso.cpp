

#define EIGEN_DONT_PARALLELIZE

#include "ADMMogLassoTall.h"
#include "ADMMogLassoWide.h"
#include "ADMMogLassoLogisticTall.h"
#include "ADMMogLassoLogisticWide.h"
#include "ADMMogLassoCoxPHTall.h"
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
typedef Map<MatrixXd> MapMatd;
typedef Eigen::Map<Eigen::SparseMatrix<double> > MSpMat;  // MappedSparseMatrix was removed in Eigen 5
typedef Eigen::SparseVector<float> SpVecf;
typedef Eigen::SparseMatrix<float> SpMatf;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;


//typedef Eigen::Ref<Eigen::VectorXd>             Refvec;
//typedef const Eigen::Ref<const Eigen::VectorXd> Constvec;


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


// Smallest lambda (on the scale of the per-observation loss) at which a zero
// solution is guaranteed for the penalty  sum_g w_g ||beta_{G_g}||  with
// possibly overlapping groups. 'grad' is the gradient of the per-observation
// loss at the fit with only the unpenalized variables (w_g = 0 groups only).
//
// Zero is optimal iff grad = lambda * sum_g w_g u_g with supp(u_g) in G_g and
// ||u_g|| <= 1. Two feasible splits of grad_j over the groups containing j
// give two sufficient values; their minimum is returned:
//   (a) u_g,j = grad_j / (lambda * W_j),        W_j = sum_{g: j in G_g, w_g > 0} w_g
//       => lambda >= max_g || (grad_j / W_j)_{j in G_g} ||
//   (b) u_g,j = grad_j / (lambda * c_j * w_g),  c_j = #{g: j in G_g, w_g > 0}
//       => lambda >= max_g || (grad_j / c_j)_{j in G_g} || / w_g
// Both equal the exact value  max_g ||grad_{G_g}|| / w_g  for non-overlapping groups.
static double lambda_max_overlap(const VectorXd &grad,
                                 const Eigen::SparseMatrix<double,Eigen::RowMajor> &C,
                                 const IntegerVector &group_idx,
                                 const VectorXd &weights,
                                 const int ngroups)
{
    const int nv = grad.size();
    VectorXi rowvar(C.rows());
    for (int r = 0; r < C.rows(); ++r)
    {
        rowvar(r) = 0;
        for (Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(C, r); it; ++it)
            rowvar(r) = it.col();
    }
    VectorXd wsum = VectorXd::Zero(nv);
    VectorXd cnt  = VectorXd::Zero(nv);
    for (int g = 0; g < ngroups; ++g)
    {
        if (weights(g) <= 0) continue;
        for (int r = group_idx(g); r < group_idx(g + 1); ++r)
        {
            wsum(rowvar(r)) += weights(g);
            cnt(rowvar(r))  += 1.0;
        }
    }
    double ua = 0.0, ub = 0.0;
    for (int g = 0; g < ngroups; ++g)
    {
        if (weights(g) <= 0) continue;
        double sa = 0.0, sb = 0.0;
        for (int r = group_idx(g); r < group_idx(g + 1); ++r)
        {
            const int j = rowvar(r);
            sa += std::pow(grad(j) / wsum(j), 2);
            sb += std::pow(grad(j) / cnt(j), 2);
        }
        ua = std::max(ua, std::sqrt(sa));
        ub = std::max(ub, std::sqrt(sb) / weights(g));
    }
    return std::min(ua, ub);
}


RcppExport SEXP admm_oglasso_dense(SEXP x_,
                                   SEXP y_,
                                   SEXP delta_,
                                   SEXP group_,
                                   SEXP family_,
                                   SEXP nlambda_,
                                   SEXP lambda_,
                                   SEXP lambda_min_ratio_,
                                   SEXP group_weights_,
                                   SEXP adaptive_lasso_,
                                   SEXP penalty_factor_,
                                   SEXP gamma_,
                                   SEXP group_idx_,
                                   SEXP ngroups_,
                                   SEXP standardize_,
                                   SEXP intercept_,
                                   SEXP compute_se_,
                                   SEXP opts_,
                                   SEXP null_grad_,
                                   SEXP dfmax_,
                                   SEXP tiestart_)
{
    BEGIN_RCPP

    //Rcpp::NumericMatrix xx(x_);
    //Rcpp::NumericVector yy(y_);

    Rcpp::NumericMatrix xx(x_);
    Rcpp::NumericVector yy(y_);
    Rcpp::NumericVector ddelta(delta_);
    Rcpp::IntegerVector ttiestart(tiestart_);

    const int n = xx.rows();
    const int p = xx.cols();

    MatrixXd datX(n, p);
    VectorXd datY(n);
    VectorXi delta(n);
    VectorXi tiestart(n);

    // Copy data and convert type from double to float
    std::copy(xx.begin(), xx.end(), datX.data());
    std::copy(yy.begin(), yy.end(), datY.data());
    std::copy(ddelta.begin(), ddelta.end(), delta.data());
    std::copy(ttiestart.begin(), ttiestart.end(), tiestart.data());

    // old stuff 
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
    

    List opts(opts_);
    const int maxit          = as<int>(opts["maxit"]);
    const int irls_maxit     = as<int>(opts["irls_maxit"]);
    const double irls_tol    = as<double>(opts["irls_tol"]);
    const double eps_abs     = as<double>(opts["eps_abs"]);
    const double eps_rel     = as<double>(opts["eps_rel"]);
    const double rho         = as<double>(opts["rho"]);
    const bool dynamic_rho   = as<bool>(opts["dynamic_rho"]);
    const double gamma       = as<double>(gamma_);   //power factor for adaptive lasso
    bool standardize         = as<bool>(standardize_);
    bool intercept           = as<bool>(intercept_);
    bool intercept_bin       = intercept;
    bool adaptive_lasso      = as<bool>(adaptive_lasso_);
    const int dfmax          = as<int>(dfmax_);  // negative means no limit
    VectorXd null_grad(as<VectorXd>(null_grad_));
    const bool lambda_auto   = (nlambda < 1);
    //bool compute_se     = compute_se_; // not used
    
    // only use wide version of solver if 
    // p >> n and p is very large
    // the Cox solver has no wide version
    bool tall_condition = n > 2 * p || p < 2500 || Rcpp::as<std::string>(family_) == "coxph";
    
    

    const SpMat group(as<MSpMat>(group_));
    CharacterVector family(as<CharacterVector>(family_));
    // group_weights not MapVec or const, because
    // we need to change it later if agaptive_lasso == true
    VectorXd group_weights(as<VectorXd>(group_weights_));
    VectorXd penalty_factor(as<VectorXd>(penalty_factor_));
    IntegerVector group_idx(group_idx_);

    const int ngroups(as<int>(ngroups_));
    VectorXd adaptive_weights(ngroups);
    adaptive_weights.setZero();

    // Standardization and centering are done in R (oglasso()), which maps
    // the coefficients back to the original scale, so DataStd is a no-op
    // here for every family. In particular the survival times of a Cox model
    // are never centered or scaled.
    // An intercept is fit as an explicit column of ones in an unpenalized group.
    standardize = false;
    intercept = false;
    bool fullbetamat = false;
    int add = 0;
    if (family(0) != "coxph")
    {

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

    // create C matrix
    //   C_{i,j} = 1 if y_i is a replicate of x_j
    //           = 0 otherwise
    Eigen::SparseMatrix<double,Eigen::RowMajor> C(Eigen::SparseMatrix<double,Eigen::RowMajor>(M, p + add));
    C.reserve(VectorXi::Constant(M,1));
    createC(C, group, M);


    DataStd<double> datstd(n, p + add, standardize, intercept);
    datstd.standardize(datX, datY);

    FADMMBase<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> *solver_tall = NULL; // obj doesn't point to anything yet
    FADMMBase<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> *solver_tall_2 = NULL;
    FADMMBase<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> *solver_wide = NULL; // obj doesn't point to anything yet
    //ADMMogLassoTall *solver_tall;
    //ADMMogLassoWide *solver_wide;


    for (int g = 0; g < ngroups; ++g)
    {
        group_weights(g) *= penalty_factor(g); // multiply weights by adaptive lasso weights
    }


    if(tall_condition)
    {

        if (family(0) == "gaussian")
        {
            solver_tall = new ADMMogLassoTall(datX, datY, C, n, p + add, M, ngroups,
                                              family, group_weights, group_idx,
                                              dynamic_rho, irls_tol, irls_maxit,
                                              eps_abs, eps_rel);
        } else if (family(0) == "binomial")
        {
            solver_tall = new ADMMogLassoLogisticTall(datX, datY, C, n, p + add, M, ngroups,
                                                      family, group_weights, group_idx,
                                                      dynamic_rho, irls_tol, irls_maxit,
                                                      eps_abs, eps_rel);
        } else if (family(0) == "coxph")
        {
            solver_tall = new ADMMogLassoCoxPHTall(datX, datY, delta, tiestart, C, n, p, M, ngroups,
                                                   family, group_weights, group_idx,
                                                   dynamic_rho, irls_tol, irls_maxit,
                                                   eps_abs, eps_rel);
        }
    }
    else
    {
        
        if (family(0) == "gaussian")
        {
            solver_wide = new ADMMogLassoWide(datX, datY, C, n, p + add, M, ngroups,
                                              family, group_weights, group_idx,
                                              dynamic_rho, irls_tol, irls_maxit,
                                              eps_abs, eps_rel);
        } else if (family(0) == "binomial")
        {
            solver_wide = new ADMMogLassoLogisticWide(datX, datY, C, n, p + add, M, ngroups,
                                                      family, group_weights, group_idx,
                                                      dynamic_rho, irls_tol, irls_maxit,
                                                      eps_abs, eps_rel);
        }
         
    }


    if(lambda_auto)
    {
        // lambda_max for the problem whose unpenalized part (intercept and
        // zero-weight groups) is fit; null_grad is computed in R at that fit
        double lmax = lambda_max_overlap(null_grad, C, group_idx, group_weights, ngroups);
        if (!(lmax > 0)) lmax = 1e-6;

        double lmin = as<double>(lambda_min_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }
    
    int nkeep = nlambda;

    VectorXd loss(nlambda);

    MatrixXd beta(p + 1, nlambda);
    IntegerVector niter(nlambda);
    double ilambda = 0.0;

    // if adaptive_lasso is specified, then call the solver
    // and estimate a model with no penalty applied

    VectorXd beta_unpen(p + add);

    if (adaptive_lasso)
    {

        VectorXd d;
        ilambda = 1e-2 * lambda[nlambda - 1] * n / datstd.get_scaleY();
        if(tall_condition)
        {
            if (p <= n || family(0) == "coxph")
            {
                solver_tall->init(ilambda, rho);
                solver_tall->solve(maxit);
                //VectorXd res = solver_tall->get_aux_gamma();
                VectorXd restmp = solver_tall->get_gamma();

                // if the design matrix includes the intercept
                // then don't back into the intercept with
                // datastd and include it to beta directly.
                VectorXd adaptive_weights(ngroups);
                if (fullbetamat)
                {
                    //double beta0;
                    //datstd.recover(beta0, res);
                    //d = res.tail(p);

                    //datstd.recover(beta0, restmp);
                    beta_unpen = restmp; //.tail(p);
                } else
                {
                    //double beta0;
                    //datstd.recover(beta0, res);
                    //d = res;
                    //datstd.recover(beta0, restmp);
                    beta_unpen = restmp;
                }
                double maxbetaval = beta_unpen.array().abs().maxCoeff();

                // if the MLEs are divergent then use
                // a linear probability model
                if (maxbetaval > 15 && family(0) != "coxph")
                {
                    solver_tall_2 = new ADMMogLassoTall(datX, datY, C, n, p + add, M, ngroups,
                                                        family, group_weights, group_idx,
                                                        dynamic_rho, irls_tol, irls_maxit,
                                                        eps_abs, eps_rel);

                    solver_tall_2->init(ilambda, rho);
                    solver_tall_2->solve(maxit);
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


            d = C * beta_unpen;

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
            }
            // reserved for wide data case

            d = C * beta_unpen;
        }


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

        if(tall_condition)
        {
            solver_tall->update_adaptive_group_weights(group_weights);
        } else 
        {
            solver_wide->update_adaptive_group_weights(group_weights);
        }

        // the adaptive weights change lambda_max
        if (lambda_auto)
        {
            double lmax = lambda_max_overlap(null_grad, C, group_idx, group_weights, ngroups);
            if (!(lmax > 0)) lmax = 1e-6;
            double lmin = as<double>(lambda_min_ratio_) * lmax;
            lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
            lambda = lambda.exp();
        }

    }

    //   =======================================   //
    //         End Adaptive Lasso Section
    //   =======================================   //

    for(int i = 0; i < nlambda; i++)
    {
        ilambda = lambda[i] * n / datstd.get_scaleY();

        if(tall_condition)
        {
            if(i == 0)
                solver_tall->init(ilambda, rho);
            else
                solver_tall->init_warm(ilambda);

            niter[i] = solver_tall->solve(maxit);

            // get computed beta
            VectorXd res = solver_tall->get_gamma();
            
            // number of nonzero coefficients, not counting the intercept column
            int nselected = solver_tall->get_nselected(res) - ((add == 1 && res(0) != 0) ? 1 : 0);
            
            if (dfmax < 0 || nselected <= dfmax || i < 1)
            {
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
            } else
            {
                nkeep = i;
                break;
            }
            
        } else 
        {
            if(i == 0)
                solver_wide->init(ilambda, rho);
            else
                solver_wide->init_warm(ilambda);
            
            niter[i] = solver_wide->solve(maxit);
            
            // get computed beta
            VectorXd res = solver_wide->get_gamma();
            
            // number of nonzero coefficients, not counting the intercept column
            int nselected = solver_wide->get_nselected(res) - ((add == 1 && res(0) != 0) ? 1 : 0);
            
            if (dfmax < 0 || nselected <= dfmax || i < 1)
            {
                // get associated loss
                loss(i) = solver_wide->get_loss();
                
                
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
            } else
            {
                nkeep = i;
                break;
            }
        }
    }

    //   =======================================   //
    //           End Estimation Section
    //   =======================================   //

    MatrixXd XX; // hessian (XtX or XtWX)
    VectorXd XY; // XtY

    // need to deallocate dynamic object
    if(tall_condition)
    {
        XX = solver_tall->get_hessian();
        XY = solver_tall->get_xty();
        delete solver_tall;
    } else 
    {
        XX = solver_wide->get_hessian();
        XY = solver_wide->get_xty();
        delete solver_wide;
    }


    return List::create(Named("lambda")           = lambda,
                        Named("nlambda")          = nlambda,
                        Named("beta")             = beta,
                        Named("niter")            = niter,
                        Named("adaptive.weights") = adaptive_weights,
                        Named("group.weights")    = group_weights,
                        Named("penalty.factor")   = penalty_factor,
                        Named("beta.unpenalized") = beta_unpen,
                        Named("loss")             = loss,
                        Named("XtX")              = XX,
                        Named("XtY")              = XY,
                        Named("C")                = C,
                        Named("group.idx")        = group_idx,
                        Named("nkeep")            = nkeep);

    END_RCPP
}






