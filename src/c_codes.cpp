
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <cppad/cppad.hpp>
#include <memory>
#include <functional>

// cpp - List type object:

struct ModelResult {
	double fn;
	Eigen::VectorXd gr;
};

using ModelFn = std::function<ModelResult(const Eigen::VectorXd&)>;

// a structure for the Leapfrog-State ( similar to a list object in R )

struct LeapfrogState {
    Eigen::VectorXd theta;
    Eigen::VectorXd r;
    double fn;
    Eigen::VectorXd gr;
};

// function to compute a single leapfrog-step
 
LeapfrogState leapfrog_step(
    const Eigen::VectorXd& theta,
    const Eigen::VectorXd& r,
    const ModelFn& model_fn,
    const Eigen::VectorXd& invM,
    double epsilon,
    int direction = 1 )
{
    // first half kick:
    ModelResult res0 = model_fn( theta );
    Eigen::VectorXd r_new = r - direction * (epsilon / 2.0) * res0.gr;
 
    // drift
    Eigen::VectorXd theta_new = theta + direction * epsilon * ( invM.array() * r_new.array() ).matrix();
 
    // second half kick
    ModelResult res1 = model_fn( theta_new );
    r_new = r_new - direction * (epsilon / 2.0) * res1.gr;
 
    // make output:
    LeapfrogState out;
    out.theta = theta_new;
    out.r     = r_new;
    out.fn    = res1.fn;
    out.gr    = res1.gr;
    return out;
}
 
// make L leapfrog-steps (in case of hmc ) 
 
LeapfrogState leapfrog(
    Eigen::VectorXd theta,
    Eigen::VectorXd r,
    const ModelFn& model_fn,
    const Eigen::VectorXd& invM,
    double epsilon,
    int L )
{
    LeapfrogState state;
    for ( int i = 0; i < L; i++ ) {
        state = leapfrog_step( theta, r, model_fn, invM, epsilon, 1 );
        theta = state.theta;
        r     = state.r;
    }
    return state;
}

// ****************** compute_H function:

// fn wird aus model_fn(theta) berechnet
double compute_H( const Eigen::VectorXd& theta, const Eigen::VectorXd& r,
    const ModelFn& model_fn, const Eigen::VectorXd& invM )
{
    ModelResult res = model_fn( theta );
    return res.fn + 0.5 * ( r.array().square() * invM.array() ).sum();
}

// fn ist schon bekannt (z.B. aus leapfrog_step) und wird direkt uebergeben
double compute_H( const Eigen::VectorXd& r, const Eigen::VectorXd& invM, double fn_val )
{
    return fn_val + 0.5 * ( r.array().square() * invM.array() ).sum();
}

// ******************************************* HMC

Eigen::VectorXd get_impulse(const Eigen::VectorXd& M) {
    
    int p = M.size();
    Eigen::VectorXd r(p);
    for (int i = 0; i < p; ++i) {
        r(i) = R::rnorm(0.0, std::sqrt(M(i)));
    }
    return r;
}

// HMC step

struct HmcState {
    Eigen::VectorXd theta;
    double alpha;
    int accept;
    bool divergent;
};


HmcState hmc_step(
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

// ******************************************* NUTS

struct NutsState {
    Eigen::VectorXd theta;
    double alpha;
    int j;
    int n_steps;
    bool divergent;
};

NutsState nuts_step(
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

            // add the result:
            all_theta.push_back(theta);
            all_r.push_back(r);
            all_fn.push_back(step.fn);
        }
        
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
    
    // // acceptable points:
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
    double alpha = alpha_sum / ( n_points - 1 );

    // is the proposal divergent?
    bool divergent = std::abs( H0 - H_proposal ) > 1000.0;

    // finally:
    NutsState out;
    out.theta     = new_theta;
    out.alpha     = alpha;
    out.j         = j;
    out.n_steps   = 1 << j;
    out.divergent = divergent;
    return out;
}

// ******************************************************

double find_reasonable_epsilon(
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

// **************************************************************************

// the test function:

ModelResult regression( const Eigen::VectorXd& theta, 
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

// make the ModelFn object:

ModelFn make_regression_model(const Eigen::VectorXd& y, const Eigen::MatrixXd& X,
    double lambda2, double a, double b)
{
    return [y, X, lambda2, a, b](const Eigen::VectorXd& theta) {
        return regression(theta, y, X, lambda2, a, b);
    };
}

// make the pointer to the result of make regression model with model specific values fixed

// [[Rcpp::export]]

Rcpp::XPtr<ModelFn> build_regression_model_xptr(
    Eigen::VectorXd y, Eigen::MatrixXd X, double lambda2, double a, double b )
{
    ModelFn* fn = new ModelFn( make_regression_model( y, X, lambda2, a, b ) );
    return Rcpp::XPtr<ModelFn>( fn, true );
}

// make an evaluator for testing

// [[Rcpp::export]]

Rcpp::List evaluate_model_ptr( Eigen::VectorXd theta, Rcpp::XPtr<ModelFn> model_ptr )
{
    const ModelFn& model_fn = *model_ptr;
    ModelResult res = model_fn( theta );

    return Rcpp::List::create(
        Rcpp::Named("fn") = res.fn,
        Rcpp::Named("gr") = res.gr
    );
}

// **************************************************************************

// function that builds the AD-tape

CppAD::ADFun<double> regression_tape(
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

ModelFn make_regression_model_cppad( const Eigen::VectorXd& theta_init,
    const Eigen::VectorXd& y, const Eigen::MatrixXd& X,
    double lambda2, double a, double b)
{
    
    // Das Tape wird über std::shared_ptr<CppAD::ADFun<double>> in der 
    // Lambda-Closure gehalten (per Value eingefangen, aber das kopiert 
    // nur den shared_ptr, nicht das Tape selbst – der shared_ptr sorgt dafür, 
    // dass das Tape am Leben bleibt, solange die Closure existiert)
    auto tape = std::make_shared<CppAD::ADFun<double>>(
        regression_tape( theta_init, y, X, lambda2, a, b ) );

    // der eigentliche Aufruf:
    return [tape]( const Eigen::VectorXd& theta ) {
        
        int p = theta.size();
        std::vector<double> theta_eval(p);
        for ( int i = 0; i < p; i++ ) theta_eval[i] = theta(i);

        std::vector<double> fn_val = tape->Forward( 0, theta_eval );
        // std::vector<double> gr_val = tape->Jacobian( theta_eval );
        std::vector<double> w(1, 1.0);                    
        std::vector<double> gr_val = tape->Reverse(1, w); 

        ModelResult res;
        res.fn = fn_val[0];
        res.gr = Eigen::Map<Eigen::VectorXd>( gr_val.data(), p );
        return res;
    
    };

}

// [[Rcpp::export]]

Rcpp::XPtr<ModelFn> build_regression_model_cppad_xptr(
    Eigen::VectorXd theta_init, // muss übergeben werden, um Tape zu initialisieren
    Eigen::VectorXd y, Eigen::MatrixXd X, double lambda2, double a, double b )
{
    ModelFn* fn = new ModelFn( make_regression_model_cppad( theta_init, y, X, lambda2, a, b ) );
    return Rcpp::XPtr<ModelFn>( fn, true );
}

// **************************************************************************

// [[Rcpp::export]]

double find_reasonable_epsilon_cpp(
    Eigen::VectorXd theta, Rcpp::XPtr<ModelFn> model_ptr, 
    Eigen::VectorXd M, Eigen::VectorXd invM )
{
    const ModelFn& model_fn = *model_ptr;
    return find_reasonable_epsilon( theta, model_fn, M, invM );
}

// [[Rcpp::export]]

Rcpp::List hmcstep_cpp( 
    Eigen::VectorXd theta, Rcpp::XPtr<ModelFn> model_ptr, 
    Eigen::VectorXd M, Eigen::VectorXd invM,
    double epsilon, int L )
{
    const ModelFn& model_fn = *model_ptr;
    HmcState out = hmc_step( theta, model_fn, M, invM, epsilon, L );
 
    return Rcpp::List::create(
        Rcpp::Named("theta") = out.theta,
        Rcpp::Named("alpha") = out.alpha,
        Rcpp::Named("accept") = out.accept,
        Rcpp::Named("divergent") = out.divergent
    );
}

// [[Rcpp::export]]

Rcpp::List nutstep_cpp(
    Eigen::VectorXd theta, Rcpp::XPtr<ModelFn> model_ptr,
    Eigen::VectorXd M, Eigen::VectorXd invM,
    double epsilon, int max_depth )
{
    const ModelFn& model_fn = *model_ptr;
    NutsState out = nuts_step( theta, model_fn, M, invM, epsilon, max_depth );
 
    return Rcpp::List::create(
        Rcpp::Named("theta") = out.theta,
        Rcpp::Named("alpha") = out.alpha,
        Rcpp::Named("j") = out.j,
        Rcpp::Named("n_steps") = out.n_steps,
        Rcpp::Named("divergent") = out.divergent
    );
}
