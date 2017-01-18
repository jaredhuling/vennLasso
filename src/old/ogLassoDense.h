#ifndef OGLASSODENSE_H
#define OGLASSODENSE_H

#include <algorithm> 
#include "ogLassoDenseBase.h"


// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. Ax + z = c
//
// x => beta
// z => -X * beta
// A => X
// b => y
// c => 0
// f(x) => lambda * ||x||_1
// g(z) => 1/2 * ||z + b||^2
class ogLassoDense: public ogLassoDenseBase
{
protected:
 
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::Map<const MatrixXd> MapMat;
    typedef Eigen::Map<const VectorXd> MapVec;
    typedef Eigen::MappedSparseMatrix<double> MSpMat;
    typedef Eigen::MappedSparseMatrix<double, Eigen::RowMajor> MSpMatR;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatR;
    typedef Eigen::SparseMatrix<int, Eigen::RowMajor> SpMatIntR;
    typedef Eigen::SparseVector<double> SparseVector;

    const MapMat x;                   // pointer to data matrix
    const MapVec y;                   // pointer response vector
    const SpMatR C;                   // pointer to C matrix
    //const MapVec D;                 // pointer D vector
    VectorXd D;
    double eig_mult;                  // spectral radius of X'X
    double lambda;                    // L1 penalty
    double lambda0;                   // minimum lambda to make coefficients all zero
    double lambda2;                   // mixing term for ridge penalty. 1 means no ridge, 0 means all ridge
    double gamma_mcp;                 // gamma parameter for MCP penalty
    double stepsize;                  // step size for gradient-based methods
    double largest_eig;

    int iter_counter;                 // which iteration are we in?

    VectorXd cache_xty;               // cache X'y
    MatrixXd cache_xcrossprod;        // cache X'X or XDX'
    MatrixXd RHS_sub;                 // needed for nvars > nobs case
    MatrixXd LHS;                     // matrix for LHS of linear system
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> one_over_D_diag; // diag(1/D)
    SparseMatrix<double,Eigen::ColMajor> CC;
    
    
    virtual void block_soft_threshold(VectorXd &gammavec, VectorXd &d, 
                                      const double &lam, const double &step_size) 
    {
        // This thresholding function is for the most
        // basic overlapping group penalty, the 
        // l1/l2 norm penalty, ie 
        //     lambda * sqrt(beta_1 ^ 2 + beta_2 ^ 2 + ...)
        
        // d is the vector to be thresholded
        
        int itrs = 0;
        
        for (int g = 0; g < ngroups; ++g) 
        {
            double ds_norm = (d.segment(group_idx(g), group_idx(g+1) - group_idx(g))).norm();
            double thresh_factor = std::max(0.0, 1 - step_size * lam * group_weights(g) / (ds_norm) );
            
            for (int gr = group_idx(g); gr < group_idx(g+1); ++gr) 
            {
                gammavec(itrs) = thresh_factor * d(gr);
                ++itrs;
            }
        }
    }
    
    virtual void block_mcp_threshold(VectorXd &gammavec, VectorXd &d, const double &lam, 
                                     const double &gam, const double &step_size) 
    {
        // This thresholding function is for the most
        // basic overlapping group penalty, the 
        // l1/l2 norm penalty, ie 
        //     lambda * sqrt(beta_1 ^ 2 + beta_2 ^ 2 + ...)
        // VectorXd d = (C * betavec).matrix() - ((1/rho) * nu.array()).matrix();
        int itrs = 0;
        
        for (int g = 0; g < ngroups; ++g) 
        {
            double ds_norm = (d.segment(group_idx(g), group_idx(g+1) - group_idx(g))).norm();
            //double thresh_factor = std::max(0.0, 1 - lam * group_weights(g) / (rho * ds_norm) );
            double thresh_factor;
            
            if (ds_norm <= step_size * lam * group_weights(g)) 
            {
                thresh_factor = 0;
            } else if (ds_norm <= step_size * gam * lam * group_weights(g))
            {
                thresh_factor = (1 / (1 - 1/gam)) * ( 1 - step_size * lam * group_weights(g) / (ds_norm) );
            } else 
            {
                thresh_factor = 1;
            }
            
            for (int gr = group_idx(g); gr < group_idx(g+1); ++gr) 
            {
                gammavec(itrs) = thresh_factor * d(gr);
                ++itrs;
            }
        }
    }
    
    virtual void block_scad_threshold(VectorXd &gammavec, VectorXd &d, const double &lam, 
                                      const double &gam, const double &step_size) 
    {
        // This thresholding function is for the most
        // basic overlapping group penalty, the 
        // l1/l2 norm penalty, ie 
        //     lambda * sqrt(beta_1 ^ 2 + beta_2 ^ 2 + ...)
        // VectorXd d = (C * betavec).matrix() - ((1/rho) * nu.array()).matrix();
        int itrs = 0;
        
        for (int g = 0; g < ngroups; ++g) 
        {
            double ds_norm = (d.segment(group_idx(g), group_idx(g+1) - group_idx(g))).norm();
            //double thresh_factor = std::max(0.0, 1 - lam * group_weights(g) / (rho * ds_norm) );
            double thresh_factor;
            
            if (ds_norm <= step_size * lam * group_weights(g)) 
            {
                thresh_factor = 0;
            } else if (ds_norm <= 2 * step_size * lam * group_weights(g))
            {
                thresh_factor = ( 1 - step_size * lam * group_weights(g) / (ds_norm) );
            } else if (ds_norm <= step_size * gam * lam * group_weights(g))
            {
                thresh_factor = (1 / (1 - 1 / (gam - 1))) * ( 1 - step_size * gam * lam * group_weights(g) / 
                                (ds_norm * (gam - 1)) );
            } else 
            {
                thresh_factor = 1;
            }
            
            for (int gr = group_idx(g); gr < group_idx(g+1); ++gr) 
            {
                gammavec(itrs) = thresh_factor * d(gr);
                ++itrs;
            }
        }
    }
    /*
    static void soft_threshold(SparseVector &res, VectorXd &vec, const double &lam)
    {
        res.setZero();
        res.reserve(vec.size() / 2);

        double *ptr = vec.data();
        for(int i = 0; i < vec.size(); i++)
        {
            if(ptr[i] > lam)
                res.insertBack(i) = ptr[i] - lam;
            else if(ptr[i] < -lam)
                res.insertBack(i) = ptr[i] + lam;
        }
    }*/
    
    virtual void rho_changed_action() 
    {
        if (method(0) == "admm") {
          
            if (cond) {
                //one_over_D__diag = ( (1/rho) * (1/D.array()).array() ).matrix().asDiagonal();
                LHS = MatrixXd::Identity(nobs, nobs) + ((1/rho) * cache_xcrossprod.array()).matrix();
            } else {
                LHS = cache_xcrossprod;
                LHS += (rho * D.array()).matrix().asDiagonal();
            }
            // precompute factorization
            ldlt.compute(LHS);
            //Rcpp::Rcout << "rho = " << rho << std::endl;
            if (cond) {
                Eigen::DiagonalMatrix<double, Eigen::Dynamic> rho_diag_d(((1/rho) * one_over_D_diag));
                RHS_sub =  ( (1/pow(rho, 2)) * one_over_D_diag ) * x.adjoint() * ldlt.solve(x * one_over_D_diag);
                RHS_sub.array() *= -1;
                RHS_sub += rho_diag_d;
            }
        }
    }
    
    virtual void update_gradient() 
    {
        if (family(0) == "gaussian") 
        {
            if (cond) 
            {
                grad_beta = x.adjoint() * (x * beta) - cache_xty - C.adjoint() * nu + 
                            (rho * ( (D.array() * beta.array()).matrix() - C.adjoint() * gamma ).array()).matrix();
            } 
            else
            {
                grad_beta = cache_xcrossprod * beta - cache_xty - C.adjoint() * nu + 
                            (rho * ( (D.array() * beta.array()).matrix() - C.adjoint() * gamma ).array()).matrix();
            }
            
            grad_gamma = nu - rho * (C * beta - gamma);
        }
    }
      

    // calculating the spectral radius of X'X
    // in this case it is the largest eigenvalue of X'X
    virtual double eigs(const MatrixXd &X)
    {
        Rcpp::NumericMatrix mat = Rcpp::wrap(X);
        
        Rcpp::Environment stats = Rcpp::Environment::namespace_env("oglasso");
        if (method(0) == "admm") 
        {
            Rcpp::Function largestEig = stats["computeLargestSmallestEigenvalueProd"];
    
            return Rcpp::as<double>(largestEig(Named("xpx", mat)));
        } 
        else if (nobs >= nvars || nvars <= 3500) 
        {
            Rcpp::Function largestEig = stats["computeLargestSmallestEigenvalue"];
            VectorXd leigseig(Rcpp::as<VectorXd>(largestEig(Named("xpx", mat))));
            largest_eig = leigseig(0);
            if (family(0) == "binomial")
            {
                largest_eig /= 4; // condition for fista to converge smaller if logistic reg
            }
            stepsize = 0.995 / largest_eig;
            
            return leigseig(0) * leigseig(1);
        }
        else if (method(0) == "fista" && cond)
        {
            Rcpp::Function largestEig = stats["computeLargestEigenvalueSVD"];
            largest_eig = Rcpp::as<double>(largestEig(Named("x", mat)));
            stepsize = 0.995 / largest_eig;
            
            return largest_eig;
        }
    }

public:
    ogLassoDense(const MatrixXd &x_, const VectorXd &y_, 
                 const SpMatR &C_,// const VectorXd &D_, 
                 int nobs_, int nvars_, int M_,
                 int ngroups_,
                 Rcpp::CharacterVector family_,
                 Rcpp::CharacterVector penalty_, 
                 Rcpp::CharacterVector method_,
                 VectorXd group_weights_,
                 Rcpp::IntegerVector group_idx_,
                 double eps_, bool dynamic_rho_,
                 double newton_tol_, double inner_tol_) :
        ogLassoDenseBase(nobs_, nvars_, M_,
                         ngroups_,
                         family_,
                         penalty_,
                         method_,
                         group_weights_,
                         group_idx_,
                         eps_, dynamic_rho_,
                         newton_tol_, inner_tol_),
        x(x_.data(), x_.rows(), x_.cols()),
        y(y_.data(), y_.size()),
        C(C_), 
        D(nvars_),
        cache_xty(nvars_),
        cache_xcrossprod(std::min(nvars_, nobs_), std::min(nvars_, nobs_)),
        RHS_sub(nvars_, nvars_),
        one_over_D_diag(nvars_),
        CC(Eigen::SparseMatrix<double>(M_, nvars_))
    {
        
        // store ColMajor version of C
        CC = C;
        
        // create vector D, whose elements are the number of times
        // each variable is in a group
        for (int k=0; k < CC.outerSize(); ++k)
        {
            double tmp_val = 0;
            for (SparseMatrix<double>::InnerIterator it(CC,k); it; ++it)
            { 
                tmp_val += it.value();
            }
            D(k) = tmp_val;
        }
        
        one_over_D_diag = (1/D.array()).matrix().asDiagonal();
        
        cache_xty = x.adjoint() * y;
        
        lambda0 = cache_xty.array().abs().maxCoeff();
        
        if (nobs_ < nvars_ && nvars_ > 3500)
        {
            cond = true;
        } 
        else 
        {
            cond = false;
        }
        if (cond) {
            if (method(0) == "admm") {
                cache_xcrossprod = XWXt(x, (1/D.array()).matrix());
            }
        } else {
            cache_xcrossprod = XtX(x);
        }
        if (method(0) == "fista" && cond) 
        {
            eig_mult = eigs(x);
        } 
        else 
        {
            if (nvars > nobs && !cond) 
            {
                eig_mult = eigs(XXt(x));
            } 
            else
            {
                eig_mult = eigs(cache_xcrossprod);
            }
        }
    }
    /*
    ogLassoGaussianDense(const double *x_, const double *y_,
                         const SpMatR *C_, const VectorXd *D_, 
                         int nobs_, int nvars_, int M_,
                         int ngroups_,
                         Rcpp::CharacterVector penalty_, 
                         Rcpp::CharacterVector method_,
                         VectorXd group_weights_,
                         Rcpp::IntegerVector group_idx_,
                         double eps_ = 1e-4, bool dynamic_rho_ = true) :
        ogLassoDense(nobs_, nvars_, M_,
                     ngroups_,
                     penalty_,
                     method_,
                     group_weights_,
                     group_idx_,
                     eps_, dynamic_rho_),
        x(x_, nobs_, nvars_),
        y(y_, nobs_),
        C(C_), //as<MSpMatR>(C_)
        D(D_)        
    {
        int min_dimension = 0;
        if (nobs_ < nvars_) {
          min_dimension = nobs_;
        } else {
          min_dimension = nvars_;
        }
        cache_xcrossprod.resize(min_dimension, min_dimension);
        cache_xty.resize(nvars_);
        cache_xty = x.transpose() * y;
        lambda0 = cache_xty.array().abs().maxCoeff();
        if (nobs_ < nvars_) {
          cache_xcrossprod = XtWX(x, (1/D.array()).array().sqrt().matrix());
        } else {
          cache_xcrossprod = XtX(x);
        }
        MatrixXd X = x;
        eig_mult = spectral_radius(X);
    }
    */

    virtual double get_lambda_zero() { return lambda0; }

    // init() is a cold start for the first lambda
    virtual void init(double lambda_, double lambda2_, double gamma_mcp_)
    {
        beta.setZero();
        gamma.fill(1.0);
        nu.setZero();
        lambda = lambda_;
        lambda2 = lambda2_;
        gamma_mcp = gamma_mcp_;
        if (cond) {
            if (method(0) == "fista") 
            {
                rho = eig_mult / 20;
            }
            else 
            {
                rho = sqrt(eig_mult / std::pow(D.sum()/D.size(), 2) ) / 20;
            }
        } else {
            rho = sqrt(eig_mult) / 20;
        }
        //rho = lambda_ / (0.1 * eig_mult);
        primal_resid = 1e10;
        dual_resid = 1e10;
        rel_primal_resid = 1e10;
        rel_dual_resid = 1e10;
        iter_counter = 0;
        ts_c = 0;
  
        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    virtual void init_warm(double lambda_, VectorXd beta_, double lambda2_, double gamma_mcp_)
    {
        beta = beta_;
        lambda = lambda_;
        lambda2 = lambda2_;
        gamma_mcp = gamma_mcp_;
        primal_resid = 1e10;
        dual_resid = 1e10;
        rel_primal_resid = 1e10;
        rel_dual_resid = 1e10;
        iter_counter = 0;
        ts_c = 0;
    }
    
    virtual void update_beta()
    {
      beta_prev = beta;
      
      if (method(0) == "admm") {
          if (cond) {
              beta = ( RHS_sub ).matrix() * 
              (cache_xty + C.adjoint() * nu + (rho *  (C.adjoint() * gamma)).matrix()).matrix();
          } else {
              beta = ldlt.solve(cache_xty + C.adjoint() * nu + (rho *  (C.adjoint() * gamma).array()).matrix() );
          }
      } else if (method(0) == "fista") {
              
          update_gradient();
          
          beta = beta - stepsize * grad_beta;
                    
      }
      
    }
    virtual void update_gamma()
    {

        gamma_prev = gamma;
        VectorXd d(M);
        
        if (method(0) == "admm")
        {
            d = (C * beta).matrix() - ((1/rho) * nu.array()).matrix();
            stepsize = 1 / rho;
        }
        else if (method(0) == "fista")
        {
            d = gamma - stepsize * grad_gamma;
        }
        
        if (penalty(0) == "gr.lasso") 
        {
            block_soft_threshold(gamma, d, lambda, stepsize);
        } 
        else if (penalty(0) == "gr.mcp") 
        {
            block_mcp_threshold(gamma, d, lambda, gamma_mcp, stepsize);
        } 
        else if (penalty(0) == "gr.scad")
        {
            block_scad_threshold(gamma, d, lambda, gamma_mcp, stepsize);
        }
        
    }
    virtual void update_nu()
    {
      
        nu_prev = nu;
        Cbeta = C * beta;
        nu -= ((rho) * (Cbeta - gamma).array()).matrix();
        
      
    }
    virtual void update_residuals()
    {
        //ts_p = ts_c;
        primal_resid = (Cbeta - gamma).norm();
        dual_resid = (C.adjoint() * (gamma - gamma_prev)).norm();
        rel_primal_denom = std::max(Cbeta.norm(), gamma.norm());
        rel_dual_denom = (C.adjoint() * gamma_prev).norm();
        
        if (rel_primal_denom <= 1e-14) {
            rel_primal_denom = 1e-5;
        }
        if (rel_dual_denom <= 1e-14) {
            rel_dual_denom = 1e-5;
        }
        
        rel_dual_resid = dual_resid / rel_dual_denom;
        rel_primal_resid = primal_resid / rel_primal_denom;

        //ts_c = 0.5 * (1 + sqrt(1 + 4 * pow(ts_p, 2)));
        
        //gamma = gamma + ((ts_p-1)/ts_c) * (gamma - gamma_prev);
        //nu = nu + ((ts_p-1)/ts_c) * (nu - nu_prev);
        
        //if ( (cache_xcrossprod * beta - cache_xty).adjoint() * (return_beta() - beta_prev) > 0 ) {
        //  ts_c = 0;
          //beta = return_beta();
          
        //}
      
    }
    virtual VectorXd return_beta() { 
        VectorXd beta_return(nvars);
        for (int k=0; k < CC.outerSize(); ++k)
        {
            int rowidx;
            bool current_zero = false;
            bool already_idx = false;
            for (SparseMatrix<double>::InnerIterator it(CC,k); it; ++it)
            {
                
                if (gamma(it.row()) == 0.0 && !current_zero)
                {
                    rowidx = it.row();
                    current_zero = true;
                } else if (!current_zero && !already_idx)
                {
                    rowidx = it.row();
                    already_idx = true;
                }
                
              
            }
            beta_return(k) = gamma(rowidx);
        }
        return beta_return; 
      }

};



#endif