#ifndef DDM7_H
#define DDM7_H

#include <RcppEigen.h>
#include <cppad/cppad.hpp>
#include "model_types.h"

// =====================================================================
//  7-Parameter Drift-Diffusion-Model (Ratcliff)
//
//  Parameter (auf der Originalskala):
//    v    Drift
//    a    Schwelle
//    z    Startpunkt (absolut, nicht relativ!)
//    t0   Non-Decision-Time (Mittelwert)
//    sv   SD der Drift-Variabilitaet
//    sz   Breite der uniformen Startpunkt-Variabilitaet
//    st0  Breite der uniformen t0-Variabilitaet
//
//  Die sv-Variabilitaet ist analytisch marginalisiert, sz und st0
//  werden per Gauss-Legendre-Quadratur (Tensorprodukt) integriert.
//
//  Spezifikation nach Dao et al. (2025).
// =====================================================================

// inline constexpr double DDM_PI = 3.14159265358979323846;

// ---------------------------------------------------------------------
//  Integrand: Dichte bei festem z und festem t0
// ---------------------------------------------------------------------

inline CppAD::AD<double> ddm7_integrand(
    const CppAD::AD<double>& tx,
    const CppAD::AD<double>& a, const CppAD::AD<double>& z,
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
    CppAD::AD<double> log_mult1 = std::log( DDM_PI ) - 2.0*CppAD::log(a)
                                  - 0.5*CppAD::log( tdsv2 );
    CppAD::AD<double> log_mult2 = -0.5*( v*v*td_safe + 2.0*v*z - z*z*sv*sv ) / tdsv2;

    // Schritt 3: Approximation der unendlichen Reihe (large-time)
    CppAD::AD<double> tmp_sum = 0.0;

    for ( int k = 1; k <= kmax; k++ ){

        //* Elemente der Summe:
        CppAD::AD<double> pt_1     = CppAD::sin( (DDM_PI*k*z)/a );
        CppAD::AD<double> log_pt_2 = (-0.5*DDM_PI*DDM_PI*k*k*td_safe)/(a*a);

        //* aufsummieren:
        CppAD::AD<double> log_hist = CppAD::log( CppAD::AD<double>(k) ) + log_pt_2;
        tmp_sum += pt_1*CppAD::exp( log_hist );

    }

    CppAD::AD<double> out = tmp_sum * CppAD::exp( log_mult1 + log_mult2 );

    // Boden einziehen: die abgeschnittene Reihe kann negativ werden.
    return CppAD::CondExpGt( out, CppAD::AD<double>(1e-29),
                             out, CppAD::AD<double>(1e-29) );
}

inline CppAD::AD<double> ddm7_density(
    const CppAD::AD<double>& tx,
    const CppAD::AD<double>& a, const CppAD::AD<double>& z,
    const CppAD::AD<double>& v, const CppAD::AD<double>& t0,
    const CppAD::AD<double>& sv, const CppAD::AD<double>& sz,
    const CppAD::AD<double>& st0,
    const Eigen::VectorXd& pts, const Eigen::VectorXd& wgh,
    const int& kmax )
{
    int npts = pts.size();

    // tau-Grenzen: L_t0 = t0 - st0/2, U_t0 = min(tx, t0 + st0/2)
    //
    // Bei der unten gewaehlten Parametrisierung gilt strukturell
    // t0 + st0/2 < min_rt <= tx, das Minimum greift also nie.
    // Der Guard bleibt als Absicherung stehen (ein Tape-Knoten pro Trial).
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
        CppAD::AD<double> z_i = z + 0.5*sz*pts(i);

        for ( int j = 0; j < npts; j++ ){

            //* aktuelles t0:
            CppAD::AD<double> t0_j = half_range_t*pts(j) + mid_t;

            //* Integrand auswerten und gewichten:
            CppAD::AD<double> f_ij = ddm7_integrand( tx, a, z_i, v, t0_j, sv, kmax );
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

inline CppAD::ADFun<double> ddm7_tape(
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
    //
    // Reihenfolge im theta-Vektor:
    //   [0] v, [1] a, [2] z, [3] t0, [4] sv, [5] sz, [6] st0
    //
    // Die Transformationen sind so gewaehlt, dass alle Nebenbedingungen
    // des Modells strukturell erfuellt sind und die Dichte auf ganz R^7
    // glatt bleibt (keine flachen Regionen, kein -inf):
    //
    //   sz  > 0
    //   st0 in (0, min_rt)
    //   z - sz/2  > 0        (Startpunkt-Intervall ueber 0)
    //   z + sz/2  < a        (Startpunkt-Intervall unter der Schwelle)
    //   t0 - st0/2 > 0       (t0-Intervall ueber 0)
    //   t0 + st0/2 < min_rt  (t0-Intervall unter dem schnellsten RT)
    // -----------------------------------------------------------------

    CppAD::AD<double> v   = ad_theta[0];
    CppAD::AD<double> sv  = CppAD::exp( ad_theta[4] );
    CppAD::AD<double> sz  = CppAD::exp( ad_theta[5] );

    // z > sz/2  und  a > z + sz/2:
    CppAD::AD<double> z = CppAD::exp( ad_theta[2] ) + 0.5*sz;
    CppAD::AD<double> a = CppAD::exp( ad_theta[1] ) + z + 0.5*sz;

    // st0 in (0, min_rt) -- Voraussetzung fuer die t0-Parametrisierung:
    CppAD::AD<double> st0 = min_rt / (1.0 + CppAD::exp( -ad_theta[6] ));
    // t0 - st0/2 in (0, min_rt - st0):
    CppAD::AD<double> t0  = 0.5*st0 + (min_rt - st0)/(1.0 + CppAD::exp( -ad_theta[3] ));

    //CppAD::AD<double> st0 = CppAD::exp( ad_theta[6] );
    //CppAD::AD<double> t0  = 0.5*st0 + CppAD::exp( ad_theta[3] );

    // -----------------------------------------------------------------
    // Schritt 3: Berechnung der Posterior
    // -----------------------------------------------------------------

    // 3.1 log-Likelihood der Daten

    CppAD::AD<double> ll = 0.0;
    for (int i = 0; i < n; i++) {

        CppAD::AD<double> rt = rts(i);

        // Spiegelung fuer die obere Schwelle:
        CppAD::AD<double> tmp = ( xs(i) == 0 )
            ? ddm7_density( rt, a, z,   v, t0, sv, sz, st0, pts, wgh, kmax )
            : ddm7_density( rt, a, a-z, -v, t0, sv, sz, st0, pts, wgh, kmax );

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

#endif // DDM7_H
