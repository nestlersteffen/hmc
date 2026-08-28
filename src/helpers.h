#ifndef HELPERS_H
#define HELPERS_H

#include <RcppEigen.h>
#include "model_types.h"

inline double compute_H( const Eigen::VectorXd& theta, const Eigen::VectorXd& r,
    const ModelFn& model_fn, const Eigen::VectorXd& invM )
{
    ModelResult res = model_fn( theta );
    return res.fn + 0.5 * ( r.array().square() * invM.array() ).sum();
}

inline double compute_H( const Eigen::VectorXd& r, const Eigen::VectorXd& invM, double fn_val )
{
    return fn_val + 0.5 * ( r.array().square() * invM.array() ).sum();
}

inline Eigen::VectorXd get_impulse(const Eigen::VectorXd& M) {
    
    int p = M.size();
    Eigen::VectorXd r(p);
    for (int i = 0; i < p; ++i) {
        r(i) = R::rnorm(0.0, std::sqrt(M(i)));
    }
    return r;
}

inline double find_reasonable_epsilon(
    const Eigen::VectorXd& theta,
    const ModelFn& model_fn,
    const Eigen::VectorXd& M,
    const Eigen::VectorXd& invM )
{

    double epsilon = 1.0;
    
    // get impulse:
    Eigen::VectorXd r0 = get_impulse( M );
    double H0 = compute_H( theta, r0, model_fn, invM );

    // make a leapfrog step:
    LeapfrogState step = leapfrog_step( theta, r0, model_fn, invM, epsilon );
    double H1 = compute_H( step.r, invM, step.fn );

    // safety check:
    if ( std::isnan( H1 ) || std::isinf( H1 ) ) return 0.001;

    // now, iterate:
    double alpha = std::min( 1.0, std::exp( H0 - H1 ) );
    int direction = ( alpha > 0.5 ) ? 1 : -1;
    int max_iter = 50;
    int iter = 0;

    while( iter < max_iter ) {
    
        // adapt epsilon:
        iter += 1;
        epsilon = epsilon * std::pow( 2, direction);
    
        // make the leapfrog step:
        step = leapfrog_step( theta, r0, model_fn, invM, epsilon, direction );
        H1 = compute_H( step.r, invM, step.fn );
        alpha = std::min( 1.0, std::exp( H0 - H1 ) );
    
        // check whether we can leave the loop
        if (direction ==  1 && alpha <= 0.5) break;
        if (direction == -1 && alpha >= 0.5) break;

    }

    if ( iter == max_iter ) Rcpp::warning("find_reasonable_epsilon_cpp: max_iter erreicht");
    
    return epsilon;
}

#endif // HELPERS_H