#ifndef HMC_H
#define HMC_H

#include <RcppEigen.h>
#include "model_types.h"
#include "leapfrog.h"

// HMC step

struct HmcState {
    Eigen::VectorXd theta;
    double alpha;
    int accept;
    bool divergent;
};


inline HmcState hmc_step(
    Eigen::VectorXd current_theta,
    const ModelFn& model_fn,
    const Eigen::VectorXd& M,
    const Eigen::VectorXd& invM,
    double epsilon,
    int L )
{
    
    // set theta to current position:
    Eigen::VectorXd theta = current_theta;
    
    // get impulse:
    Eigen::VectorXd r = get_impulse( M );
    Eigen::VectorXd current_r = r;
    double current_H = compute_H( current_theta, current_r, model_fn, invM );

    // make L leapfrog steps:
    LeapfrogState lf = leapfrog( theta, r, model_fn, invM, epsilon, L);
    theta = lf.theta;
    r = lf.r;

    // should we accept the proposal?

    //- 1. symmetrize r:
    r = -1*r;

    //- 2. compute energy
    double proposed_H = compute_H( r, invM, lf.fn );

    //- 3. accept or reject the state at end of trajectory
    double log_alpha = current_H - proposed_H;
    if ( std::isnan(log_alpha) || std::isinf(log_alpha) ) log_alpha = -R_PosInf;
    double alpha = std::min( 1.0, std::exp( log_alpha ) );

    //- 4. is the proposal divergent?
    bool divergent = std::abs(log_alpha) > 1000.0;

    // finally:
    int accept = 1;
    if ( std::log( R::runif(0,1) ) >= log_alpha ) {
        theta  = current_theta;
        accept = 0;
    }

    HmcState out;
    out.theta  = theta;
    out.alpha  = alpha;
    out.accept = accept;
    out.divergent = divergent;
    return out;
}

#endif // HMC_H