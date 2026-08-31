#ifndef DDM4_H
#define DDM4_H

#include <RcppEigen.h>
#include <cppad/cppad.hpp>
#include "model_types.h"

// inline constexpr double DDM_PI = 3.14159265358979323846;

inline CppAD::AD<double> ddm4_dstand( 
    const CppAD::AD<double>& t, 
    const CppAD::AD<double>& a, const CppAD::AD<double>& w, const CppAD::AD<double>& v, 
    const double& s2, const int& kmax )
{

    // CppAD::AD<double> DDM_PI = 4.0 * CppAD::atan(CppAD::AD<double>(1.0));

    // Step 1: compute multiplicative term:
    CppAD::AD<double> log_mult = CppAD::log( s2*DDM_PI ) - 2*CppAD::log(a) + ( -w*a*v - 0.5*v*v*t )/s2;

    // Step 1: we approximate the infinite sum:
    CppAD::AD<double> tmp_sum = 0.0;

    for ( int k = 1; k < kmax; k++ ){

        //* compute elements of the sum:
        CppAD::AD<double> pt_1 = CppAD::sin( DDM_PI*k*w );
        CppAD::AD<double> log_pt_2 = (-0.5*DDM_PI*DDM_PI*k*k*s2*t)/(a*a);

        //* compute the sum and save:
        CppAD::AD<double> log_hist = CppAD::log( k ) + log_pt_2;
        tmp_sum +=  pt_1*CppAD::exp( log_hist );
            
    }

    return CppAD::CondExpGt( tmp_sum, CppAD::AD<double>(0.0), tmp_sum*CppAD::exp( log_mult ), CppAD::AD<double>(1e-29));

}

// function that builds the tape

inline CppAD::ADFun<double> ddm4_tape(
    const Eigen::VectorXd& theta, 
    const Eigen::VectorXd& rts, const Eigen::VectorXd& xs,
    const Eigen::VectorXd& muPrior_sp, const Eigen::VectorXd& sdPrior_sp,
    const double min_rt,
    const double s2, const int kmax ) 
{

    int p = theta.size(); // Anzahl der Parameter
    int n = rts.size();
    
    // Schritt 1: Tape starten
    std::vector<CppAD::AD<double>> ad_theta(p);
    for (int i = 0; i < p; i++) {
        ad_theta[i] = theta(i);
    }
    CppAD::Independent(ad_theta); // ab hier wird aufgezeichnet

    // Schritt 2: Zuweisung der Paramater
    CppAD::AD<double> v  = ad_theta[0]; 
    CppAD::AD<double> a  = CppAD::exp( ad_theta[1] );

    // model t0:
    //CppAD::AD<double> t0 = CppAD::exp( ad_theta[3] );
    CppAD::AD<double> t0 = min_rt / (1.0 + CppAD::exp(-ad_theta[3]));

    // model z or w:
    //CppAD::AD<double> z  = CppAD::exp( ad_theta[2] );
    //a+=z;
    //CppAD::AD<double> w  = z/a;
    CppAD::AD<double> w  = 1.0 / (1.0 + CppAD::exp( -ad_theta[2] ));

    // Schritt 3: Berechnung der Posterior
    
    // 3.1 log-Likelihood der Daten

    CppAD::AD<double> ll = 0.0;
    for (int i = 0; i < n; i++) {
    
        // get reaction time and response:
        CppAD::AD<double> t = rts(i) - t0;
        // CppAD::AD<double> x = xs(i);

        // make a safe reaction time:
        CppAD::AD<double> t_safe = CppAD::CondExpGt( t, CppAD::AD<double>(1e-10), t, CppAD::AD<double>(1e-10) );

        // we now compute the ddm-density:
        CppAD::AD<double> tmp = 0.0;
        if ( xs(i) == 0 ) tmp = ddm4_dstand( t_safe, a, w, v, s2, kmax );
        else              tmp = ddm4_dstand( t_safe, a, 1-w, -v, s2, kmax );

        // a sanity check:
        tmp = CppAD::CondExpGt( t, CppAD::AD<double>(0.0), tmp, CppAD::AD<double>(1e-29) );

        // add term:
        ll += CppAD::log( tmp );

    }

    // 3.2 Priors (hier Normalverteilung)

    CppAD::AD<double> ll_p = 0.0;
    for (int i = 0; i < p; i++) {
        CppAD::AD<double> mu = muPrior_sp(i);
        CppAD::AD<double> sd = sdPrior_sp(i);
        CppAD::AD<double> resid = ad_theta[i] - mu;
        ll_p += -CppAD::log(sd) - (resid*resid)/(2*sd*sd);
    }

    // finales outcome definieren und damit das "Tape beenden"
    std::vector<CppAD::AD<double>> ad_negloglik(1);
    ad_negloglik[0] = -1*( ll + ll_p);
    CppAD::ADFun<double> f(ad_theta, ad_negloglik);

    return f;

}

#endif // DDM4_H