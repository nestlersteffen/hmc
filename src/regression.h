#ifndef REGRESSION_H
#define REGRESSION_H

#include <RcppEigen.h>
#include <cppad/cppad.hpp>
#include "model_types.h"

// *** analytic functions

inline ModelResult regression( const Eigen::VectorXd& theta, 
	const Eigen::VectorXd& y, const Eigen::MatrixXd& X,
	const double lambda2, const double a, const double b )
{
 	int p = theta.size();
	int n = X.rows();
	// make parms:
	Eigen::VectorXd beta = theta.head(p-1);
	double phi = theta(p-1);
	double sigma2 = std::exp( 2.0*phi );
	// make parms:
	Eigen::VectorXd ey = y - X*beta;
	// compute nll:
	double sum_ey_sq = ey.squaredNorm();
	double sum_beta_sq = beta.squaredNorm();
	double ll = -n * phi - sum_ey_sq/( 2.0*sigma2 );
    double lp_b = -sum_beta_sq/( 2.0*lambda2 );
    double lp_s = -2.0*(a+1)*phi - b/sigma2;
    double jac = 2.0*phi;
    // compute gradient:
    Eigen::VectorXd g_beta = (1.0/sigma2) * (X.transpose() * ey) - (beta/lambda2);
	double g_phi = -n + sum_ey_sq/sigma2 - 2.0*(a + 1.0) + (2.0*b/sigma2) + 2.0;
	// make gradient:
	Eigen::VectorXd gr(p);
	gr.head(p-1) = g_beta; 
    gr(p-1) = g_phi; 
    // output
    ModelResult res;
    res.fn = -1*(ll + lp_b + lp_s + jac);
    res.gr = -1*gr;
    return res;
}

// *** CppAD function

inline CppAD::ADFun<double> regression_tape(
    const Eigen::VectorXd& theta, 
    const Eigen::VectorXd& y, const Eigen::MatrixXd& X, 
    const double a, const double b, const double lambda2 ) 
{

    int p = theta.size(); // Anzahl der Parameter
    int d = X.cols();
    int n = y.size();
    
    // Schritt 1: Tape starten
    std::vector<CppAD::AD<double>> ad_theta(p);
    for (int i = 0; i < p; i++) {
        ad_theta[i] = theta(i);
    }
    CppAD::Independent(ad_theta); // ab hier wird aufgezeichnet

    // Schritt 2: Zuweisung der Paramater
    std::vector<CppAD::AD<double>> beta(d);
    for (int i = 0; i < d; i++) {
        beta[i] = ad_theta[i];
    }
    CppAD::AD<double> phi = ad_theta[p-1];
    CppAD::AD<double> sigma2 = CppAD::exp(2*phi);

    // Schritt 3: Berechnung der Posterior

    // 3.1 die log-likelihood
    CppAD::AD<double> ll = 0.0;
    for (int i = 0; i < n; i++) {
        CppAD::AD<double> mu = 0.0;
        for ( int j = 0; j < d; j++ ) {
            mu += X(i, j) * beta[j];
        }
        CppAD::AD<double> resid = y[i] - mu;
        ll += -phi - (resid * resid) / (2*sigma2);
    }

    // 3.2 die log-Priors inkl. Jacobian
    CppAD::AD<double> lp_beta = 0.0;
    for ( int j = 0; j < d; j++ ) {
        lp_beta += -( beta[j]*beta[j] )/( 2*lambda2 );
    }

    CppAD::AD<double> lp_phi = -2*(a+1)*phi - b/sigma2;
    CppAD::AD<double> jac = 2*phi;

    // finales outcome definieren und damit das "Tape beenden"
    std::vector<CppAD::AD<double>> ad_loglik(1);
    ad_loglik[0] = -1*(ll + lp_beta + lp_phi + jac);
    CppAD::ADFun<double> f(ad_theta, ad_loglik);

    return f;
    
}

#endif // REGRESSION_H