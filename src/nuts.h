#ifndef NUTS_H
#define NUTS_H

#include <RcppEigen.h>
#include "model_types.h"
#include "leapfrog.h"

struct NutsState {
    Eigen::VectorXd theta;
    double alpha;
    int j;
    int n_steps;
    bool divergent;
    Eigen::VectorXd divergent_theta;
};

inline NutsState nuts_step(
    Eigen::VectorXd current_theta,
    const ModelFn& model_fn,
    const Eigen::VectorXd& M,
    const Eigen::VectorXd& invM,
    double epsilon,
    int max_depth )
{

    // get impulse:
    Eigen::VectorXd r0 = get_impulse( M );
    ModelResult res0 = model_fn( current_theta );
    double fn0 = res0.fn;
    double H0 = compute_H( r0, invM, fn0 );

    // set thetas and rs
    Eigen::VectorXd theta_fwd = current_theta;
    Eigen::VectorXd theta_bwd = current_theta;
    Eigen::VectorXd r_fwd = r0;
    Eigen::VectorXd r_bwd = r0;

    // containers for dynamic savings:
    std::vector<Eigen::VectorXd> all_theta;
    std::vector<Eigen::VectorXd> all_r;
    std::vector<double> all_fn;
    
    // save first elements:
    all_theta.push_back(current_theta);
    all_r.push_back(r0);
    all_fn.push_back(fn0);

    int j = 0;
    bool batch_divergent = false;
    int p = current_theta.size();
    Eigen::VectorXd divergent_theta(p);
    divergent_theta.setZero();

    while( true ) {

        // no. of steps:
        int steps = 1 << j;

        // get direction:
        int direction = ( R::runif(0.0, 1.0) < 0.5 ) ? -1 : 1;
    
        // set the start point depending on direction
        Eigen::VectorXd theta, r;
        if ( direction == 1) {
            theta = theta_fwd; r = r_fwd;
        } else {
            theta = theta_bwd; r = r_bwd; 
        }
    
        // make leap-frog steps:
        for ( int i = 0; i < steps; i++ ) {
            
            // make the steps:
            LeapfrogState step = leapfrog_step( theta, r, model_fn, invM, epsilon, direction );
            theta = step.theta;
            r     = step.r;

            // aktuelle Energie sofort anschauen
            double H_i = step.fn + 0.5 * (r.array().square() * invM.array()).sum();
            
            if ( std::isnan(H_i) || std::isinf(H_i) || std::abs(H_i - H0) > 1000.0 ) {
                batch_divergent = true;
                divergent_theta = theta;
                break;   // sofort raus aus der inneren steps-Schleife
            }

            // add the result:
            all_theta.push_back(theta);
            all_r.push_back(r);
            all_fn.push_back(step.fn);
        }
        
        if ( batch_divergent ) break;   

        // adapt end of trajectory
        if ( direction == 1) {
            theta_fwd = theta; r_fwd = r;
        } else {
            theta_bwd = theta; r_bwd = r; 
        }

        // have we reached the uturn?
        Eigen::VectorXd diff_theta = theta_fwd - theta_bwd;
        double uturn_fwd = diff_theta.dot(r_fwd);
        double uturn_bwd = diff_theta.dot(r_bwd);
        if ( std::isnan( uturn_fwd ) || std::isnan( uturn_bwd ) || uturn_fwd < 0 || uturn_bwd < 0 ) break;

        // Sicherheitsgrenze:
        if ( j >= max_depth ) break;
        
        // adapt j:
        j = j + 1;
      
    }

    // start slice sampling:
    double log_u = std::log( R::runif(0,1) ) - H0;
    
    // acceptable points:
    int n_points = all_theta.size();
    std::vector<double> H_all( n_points );
    std::vector<int> acceptable;
    acceptable.reserve( n_points );

    for ( int i = 0; i < n_points; i++ ) {
        double kinetic = 0.5 * ( all_r[i].array().square() * invM.array() ).sum();
        H_all[i] = all_fn[i] + kinetic;
        if ( -H_all[i] >= log_u ) acceptable.push_back( i );
    }

    Eigen::VectorXd new_theta, new_r;
    double new_fn;

    if ( acceptable.empty() ) {
        new_theta = current_theta;
        new_r     = r0;
        new_fn    = fn0;
    } else {
        int rand_idx = static_cast<int>( R::runif(0.0, 1.0) * acceptable.size() );
        if ( rand_idx >= (int)acceptable.size() ) rand_idx = acceptable.size() - 1;
        int idx   = acceptable[rand_idx];
        new_theta = all_theta[idx];
        new_r     = all_r[idx];
        new_fn    = all_fn[idx];
    }
    double H_proposal = compute_H( new_r, invM, new_fn );

    // mittlere Akzeptanzwahrscheinlichkeit:
    double alpha_sum = 0.0;
    for ( int i = 1; i < n_points; i++ ) {
        alpha_sum += std::min( 1.0, std::exp( H0 - H_all[i] ) );
    }
    // double alpha = alpha_sum / ( n_points - 1 );
    double alpha = ( n_points > 1 ) ? alpha_sum / ( n_points - 1 ) : 0.0;

    // is the proposal divergent?
    bool divergent = batch_divergent || std::abs( H0 - H_proposal ) > 1000.0;

    // finally:
    NutsState out;
    out.theta     = new_theta;
    out.alpha     = alpha;
    out.j         = j;
    out.n_steps   = 1 << j;
    out.divergent = divergent;
    out.divergent_theta = divergent_theta;
    return out;
}

#endif // NUTS_H