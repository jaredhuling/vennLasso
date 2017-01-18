#ifndef OGLASSODENSEBASE_H
#define OGLASSODENSEBASE_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector> 
#include <functional> 
#include <algorithm> 
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <float.h>
#include <string>
#include "utils.h"

using namespace Rcpp;
using Eigen::LDLT;
using Rcpp::IntegerVector;
using Rcpp::CharacterVector;

// #define ADMM_PROFILE 2

// General problem setting
//   minimize f(x) + g(z)
//   s.t. Ax + Bz = c
//
// x(n, 1), z(m, 1), A(p, n), B(p, m), c(p, 1)
//
//template<typename VecTypeX, typename VecTypeZ>
class ogLassoDenseBase
{
protected:
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::MatrixXd MatrixXd;

    int nobs;                 // number of observations
    int nvars;                // number of variables
    int ngroups;              // number of groups
    int M;                    // length of nu (total size of all groups)

    VectorXd beta;            // parameters to be optimized
    VectorXd gamma;           // auxiliary parameters
    VectorXd nu;              // Lagrangian multiplier
    VectorXd beta_prev;       // parameters to be optimized
    VectorXd gamma_prev;      // auxiliary parameters
    VectorXd nu_prev;         // Lagrangian multiplier
    IntegerVector group_idx;  // indices of groups
    VectorXd Cbeta;           // vector C * beta 
    VectorXd grad_beta;       // gradient wrt beta
    VectorXd grad_gamma;      // gradient wrt gamma
    
    CharacterVector family;   // model family (gaussian, binomial, or Cox PH)
    CharacterVector penalty;  // penalty type
    CharacterVector method;   // algorithmic method
    VectorXd group_weights;   // group weight multipliers
    
    Eigen::LDLT<MatrixXd> ldlt;
    

    double rho;               // augmented Lagrangian parameter
    double eps;               // relative tolerance
    double newton_tol;        // tolerance for newton iterations
    double inner_tol;         // tolerance for inner iterations
    

    double primal_resid;      // primal residual
    double dual_resid;        // dual residual
    double rel_primal_resid;  // relative primal residual
    double rel_dual_resid;    // relative dual residual
    double rel_primal_denom;  // relative primal residual denominator
    double rel_dual_denom;    // relative dual residual denominator
    double ts_c;
    double ts_p;
    bool cond;
    

    
    bool dynamic_rho;
    int number_consec_zero_gammas;
    
    // action when rho is changed, e.g. re-factorize matrices
    virtual void rho_changed_action() {}
    
    // increase or decrease rho in iterations
    virtual void update_rho()
    {
        // If the dual and primal residuals are
        // too far apart, change rho in a way
        // that will induce them to be more 
        // balanced.
        if (dynamic_rho) {
            if (primal_resid > 10 * dual_resid)
            {
                rho *= 2;
                rho_changed_action();
            } else if (dual_resid > 10 * primal_resid)
            {
                rho *= 0.5;
                rho_changed_action();
            }
        }
    }
    
    static bool stopRule(const VectorXd& cur, const VectorXd& prev, 
                         const VectorXd& cur2, const VectorXd& prev2, const double& tolerance) 
    {
        for (unsigned i = 0; i < cur.rows(); i++) {
          if ( (cur(i) != 0 && prev(i) == 0) || (cur(i) == 0 && prev(i) != 0) ) {
            return 0;
          }
          if (cur(i) != 0 && prev(i) != 0 && 
              std::abs( (cur(i) - prev(i)) / prev(i)) > tolerance) {
            return 0;
          }
        }
        for (unsigned i = 0; i < cur2.rows(); i++) {
          if (cur2(i) != 0 && prev2(i) != 0 && 
              std::abs( (cur2(i) - prev2(i)) / prev2(i)) > tolerance) {
            return 0;
          }
          if ( (cur2(i) != 0 && prev2(i) == 0) || (cur2(i) == 0 && prev2(i) != 0) ) {
            return 0;
          }
        }
        return 1;
    }

public:
    ogLassoDenseBase(int nobs_, int nvars_, int M_,
                     int ngroups_,
                     CharacterVector family_,
                     CharacterVector penalty_,
                     CharacterVector method_,
                     VectorXd group_weights_,
                     IntegerVector group_idx_,
                     double eps_, bool dynamic_rho_,
                     double newton_tol_, double inner_tol_) :
        nobs(nobs_), nvars(nvars_), M(M_),
        ngroups(ngroups_), 
        family(family_), penalty(penalty_),
        method(method_), group_weights(group_weights_),
        group_idx(group_idx_),
        beta(nvars_), gamma(M_), nu(M_), Cbeta(M_),     // allocate space but do not set values
        beta_prev(nvars_), gamma_prev(M_), nu_prev(M_), // allocate space but do not set values
        grad_beta(nvars_), grad_gamma(M_),              // allocate space but do not set values
        eps(eps_), dynamic_rho(dynamic_rho_),
        newton_tol(newton_tol_), inner_tol(inner_tol_)
    {}

    virtual void update_beta() {}
    virtual void update_gamma() {}
    virtual void update_nu() {}
    virtual void update_residuals() {}

    virtual void debuginfo()
    {
        Rcpp::Rcout << "eps = " << eps << std::endl;
        Rcpp::Rcout << "primal_resid = " << primal_resid << std::endl;
        Rcpp::Rcout << "rel_primal_resid = " << rel_primal_resid << std::endl;
        Rcpp::Rcout << "dual_resid = " << dual_resid << std::endl;
        Rcpp::Rcout << "rel_dual_resid = " << rel_dual_resid << std::endl;
        Rcpp::Rcout << "rho = " << rho << std::endl;
    }

    virtual bool converged(int cur_iter)
    {
        // 'converged' checks if both dual and primal
        // residuals have converged, or if the 'gamma'
        // variable (the one that is thresholded) remains
        // zero for a long time and all of 'beta' is 
        // very small (as beta should be converging to 
        // gamma due to the constrained optimization that 
        // C * beta = gamma)
        if ((rel_primal_resid < eps) &&
               (rel_dual_resid < eps))
        {
            return true;
        }
        else 
        {
            if (cur_iter > 2)
            {
                if (gamma.array().abs().matrix().sum() <= 1e-15)
                {
                    ++number_consec_zero_gammas;
                }
                
                if ((number_consec_zero_gammas > 100) && (beta.array().abs().matrix().sum() <= 1e-10))
                {
                    return true;
                } else 
                {
                    return false;
                }
                
            } else
            {
                number_consec_zero_gammas = 0;
                return false;
            }
        }
    }

    virtual int fit(int maxit, int newton_maxit, int inner_maxit)
    {
        // ==================================================== //
        // This version of the 'fit' method is for the most     //
        // basic model: linear regression with the overlapping  //
        // group penalty (ie, method = "gaussian").             //
        //                                                      //
        // As this method is virtual, the logistic and cox PH   //
        // methods will need to overwrite the 'fit' method in   //
        // favor of one which performs a Newton-like procedure. //
        // ==================================================== //
        int i;
        
        if (family(0) == "gaussian" && method(0) == "admm") 
        {
            // if family is gaussian, use one set of outer iterations,
            // no Newton iterations required. Else, if the method is
            // fista, then no Newton iteratons are required.

            for(i = 0; i < maxit; i++)
            {
                
                update_beta();
        
                update_gamma();
        
                update_nu();
                
                update_residuals();
                
                update_rho();
                
        
                // debuginfo();
                if(converged(i)) {
                    break;
                }
            }
        } else if (method(0) == "fista")
        {
            // if method == fista, we need to iterate between updating beta
            // and gamma until convergence before updating nu. This approach
            // only requires the gradient at each beta/gamma update so it 
            // is done the same way for all families (gaussian, binomial, and cox)
            
            for(i = 0; i < maxit; i++)
            {
                double tt = 0;
                double tt_prev;
                for (int ii = 0; ii < inner_maxit; ii++)
                {
                    tt_prev = tt;
                    
                    update_beta();
                    
                    update_gamma();
                    
                    
                    // update FISTA step-length
                    tt = 0.5 * (1 + sqrt(1 + 4 * pow(tt_prev, 2)));
                    
                    // FISTA updates
                    //beta = beta_prev + ((tt_prev - 1) / tt) * (beta - beta_prev);
                    //gamma = gamma_prev + ((tt_prev - 1) / tt) * (gamma - gamma_prev);
                    
                    if (stopRule(gamma, gamma_prev, beta, beta_prev, inner_tol))
                    {
                        break;
                    }
                    
                }
        
                update_nu();
                
                update_residuals();
                
                update_rho();
        
                // debuginfo();
                if(converged(i)) {
                    break;
                }
            }
        } // need to add else if here
        else if (method(0) == "admm" || method(0) == "fista.newton")
        {
            // if the family is not gaussian and the method is admm,
            // use a Newton-like procedure. Else if method is the Newton
            // method with fista iterations for the weighted least squares
            // problems, then use Newton iterations.
                        
            for (i = 0; i < newton_maxit; i++)
            {
                int j;
                
                for(j = 0; j < maxit; j++)
                {
                    update_beta();
            
                    update_gamma();
            
                    update_nu();
                    
                    update_residuals();
                    
                    update_rho();
            
                    // debuginfo();
                    if(converged(j)) {
                        break;
                    }
                }
            }
            

        }
  
          return i;
    }

    virtual VectorXd return_beta() { return beta; }
    virtual VectorXd return_gamma() { return gamma; }
    virtual VectorXd return_nu() { return nu; }
};



#endif