#ifndef DDM7_W_H
#define DDM7_W_H

#include <RcppEigen.h>
#include <cppad/cppad.hpp>
#include "model_types.h"

inline CppAD::AD<double> ddm7_w_integrand(
    const CppAD::AD<double>& tx,
    const CppAD::AD<double>& a, const CppAD::AD<double>& w,
    const CppAD::AD<double>& v, const CppAD::AD<double>& t0,
    const CppAD::AD<double>& sv,
    const int& kmax )
{
    
    // Schritt 1: effektive Entscheidungszeit
    CppAD::AD<double> td = tx - t0;
    CppAD::AD<double> td_safe = CppAD::CondExpGt( td, CppAD::AD<double>(1e-10),
                                                  td, CppAD::AD<double>(1e-10) );
    // Schritt 2: multiplikative Terme (sv analytisch marginalisiert)
    CppAD::AD<double> tdsv2 = 1.0 + td_safe*sv*sv;
    CppAD::AD<double> log_mult1 = std::log( DDM_PI ) - 2.0*CppAD::log(a) - 0.5*CppAD::log( tdsv2 );

    CppAD::AD<double> aw = a*w;                       
    CppAD::AD<double> log_mult2 = -0.5*( v*v*td_safe + 2.0*v*aw - aw*aw*sv*sv ) / tdsv2;

    // Schritt 3: Approximation der unendlichen Reihe (large-time)
    CppAD::AD<double> tmp_sum = 0.0;
    for ( int k = 1; k <= kmax; k++ ){
 
        CppAD::AD<double> pt_1     = CppAD::sin( DDM_PI*k*w );
        CppAD::AD<double> log_pt_2 = (-0.5*DDM_PI*DDM_PI*k*k*td_safe)/(a*a);
        CppAD::AD<double> log_hist = std::log( static_cast<double>(k) ) + log_pt_2;
        tmp_sum += pt_1*CppAD::exp( log_hist );
 
    }

    CppAD::AD<double> out = tmp_sum * CppAD::exp( log_mult1 + log_mult2 );
    return CppAD::CondExpGt( out, CppAD::AD<double>(1e-29),out, CppAD::AD<double>(1e-29) );
}

inline CppAD::AD<double> ddm7_w_density(
    const CppAD::AD<double>& tx,
    const CppAD::AD<double>& a, const CppAD::AD<double>& w,
    const CppAD::AD<double>& v, const CppAD::AD<double>& t0,
    const CppAD::AD<double>& sv, const CppAD::AD<double>& sw,
    const CppAD::AD<double>& st0,
    const Eigen::VectorXd& pts, const Eigen::VectorXd& wgh,
    const int& kmax )
{
    int npts = pts.size();

    // t0 - Grenzen:
    CppAD::AD<double> L_t0      = t0 - 0.5*st0;
    CppAD::AD<double> U_t0_full = t0 + 0.5*st0;
    CppAD::AD<double> U_t0      = CppAD::CondExpLt( tx, U_t0_full, tx, U_t0_full );

    CppAD::AD<double> width_t      = U_t0 - L_t0;
    CppAD::AD<double> width_t_safe = CppAD::CondExpGt( width_t, CppAD::AD<double>(1e-10),
                                                       width_t, CppAD::AD<double>(1e-10) );

    CppAD::AD<double> half_range_t = 0.5*width_t_safe;
    CppAD::AD<double> mid_t        = 0.5*(U_t0 + L_t0);

    // Tensorprodukt-Quadratur:
    CppAD::AD<double> total = 0.0;

    for ( int i = 0; i < npts; i++ ){

        //- aktueller Startpunkt:
        CppAD::AD<double> w_i = w + 0.5*sw*pts(i);

        for ( int j = 0; j < npts; j++ ){

            //* aktuelles t0:
            CppAD::AD<double> t0_j = half_range_t*pts(j) + mid_t;

            //* Integrand auswerten und gewichten:
            CppAD::AD<double> f_ij = ddm7_w_integrand( tx, a, w_i, v, t0_j, sv, kmax );
            total += f_ij * wgh(i) * wgh(j);

        }

    }

    // Normierung: 0.5 (fuer z) * width_t/(2*st0) (fuer t0)
    CppAD::AD<double> out = 0.25 * width_t_safe / st0 * total;

    return CppAD::CondExpGt( out, CppAD::AD<double>(1e-29),
                             out, CppAD::AD<double>(1e-29) );
}

// ---------------------------------------------------------------------
//  Tape-Konstruktion
// ---------------------------------------------------------------------

inline CppAD::ADFun<double> ddm7_w_tape(
    const Eigen::VectorXd& theta,
    const Eigen::VectorXd& rts, const Eigen::VectorXd& xs,
    const Eigen::VectorXd& pts, const Eigen::VectorXd& wgh,
    const Eigen::VectorXd& muPrior_sp, const Eigen::VectorXd& sdPrior_sp,
    const double min_rt, const int kmax )
{

    int p = theta.size();   // Anzahl der Parameter (hier 7)
    int n = rts.size();

    // Schritt 1: Tape starten
    std::vector<CppAD::AD<double>> ad_theta(p);
    for (int i = 0; i < p; i++) {
        ad_theta[i] = theta(i);
    }
    CppAD::Independent(ad_theta);

    // -----------------------------------------------------------------
    // Schritt 2: Zuweisung der Parameter
    // -----------------------------------------------------------------

    // v, sv, and a 
    CppAD::AD<double> v = ad_theta[0];
    CppAD::AD<double> sv = CppAD::exp( ad_theta[4] );
    CppAD::AD<double> a = CppAD::exp( ad_theta[1] );

    // sw and w in [0,1]; w depends on sw
    CppAD::AD<double> sw = 1.0/(1.0 + CppAD::exp( -ad_theta[5] ));
    CppAD::AD<double> u_w = 1.0/(1.0 + CppAD::exp( -ad_theta[2] ));
    CppAD::AD<double> w = 0.5*sw + (1.0 - sw)*u_w;
    
    // st0 in (0, min_rt)
    CppAD::AD<double> st0 = min_rt / (1.0 + CppAD::exp( -ad_theta[6] ));
    // t0 - st0/2 in (0, min_rt - st0):
    CppAD::AD<double> u_t0 = 1.0/(1.0 + CppAD::exp( -ad_theta[3] ));
    CppAD::AD<double> t0 = 0.5*st0 + (min_rt - st0)*u_t0;

    // -----------------------------------------------------------------
    // Schritt 3: Berechnung der Posterior
    // -----------------------------------------------------------------

    // 3.1 log-Likelihood der Daten

    CppAD::AD<double> ll = 0.0;
    for (int i = 0; i < n; i++) {

        CppAD::AD<double> rt = rts(i);

        // Spiegelung fuer die obere Schwelle:
        CppAD::AD<double> tmp = ( xs(i) == 0 )
            ? ddm7_w_density( rt, a, w, v, t0, sv, sw, st0, pts, wgh, kmax )
            : ddm7_w_density( rt, a, 1-w, -v, t0, sv, sw, st0, pts, wgh, kmax );
        ll += CppAD::log( tmp );

    }
    
    // 3.2 Priors (Normalverteilung auf der unbeschraenkten Skala)

    CppAD::AD<double> ll_p = 0.0;
    for (int i = 0; i < p; i++) {
        CppAD::AD<double> mu    = muPrior_sp(i);
        CppAD::AD<double> sd    = sdPrior_sp(i);
        CppAD::AD<double> resid = ad_theta[i] - mu;
        ll_p += -CppAD::log(sd) - (resid*resid)/(2*sd*sd);
    }

    // -----------------------------------------------------------------
    // Schritt 4: Tape abschliessen
    // -----------------------------------------------------------------

    std::vector<CppAD::AD<double>> ad_negloglik(1);
    ad_negloglik[0] = -1*( ll + ll_p );
    CppAD::ADFun<double> f(ad_theta, ad_negloglik);

    // Bei npts^2 * kmax Reihengliedern pro Trial wird das Tape gross.
    // optimize() eliminiert tote Knoten und gemeinsame Teilausdruecke.
    f.optimize();

    return f;

}

#endif // DDM7_W_H
