
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include "model_types.h"
#include "leapfrog.h"
#include "helpers.h"
#include "hmc.h"
#include "nuts.h"
#include "my_model.h"
#include "ddm_lan.h"

// **************************
// ***   REGRESSION part
// **************************

// [[Rcpp::export]]

Rcpp::XPtr<ModelFn> build_regression_model_xptr(
    Eigen::VectorXd y, Eigen::MatrixXd X, double lambda2, double a, double b )
{
    ModelFn* fn = new ModelFn( make_regression_model( y, X, lambda2, a, b ) );
    return Rcpp::XPtr<ModelFn>( fn, true );
}

// [[Rcpp::export]]

Rcpp::XPtr<ModelFn> build_regression_model_cppad_xptr(
    Eigen::VectorXd theta_init, // muss übergeben werden, um Tape zu initialisieren
    Eigen::VectorXd y, Eigen::MatrixXd X, double lambda2, double a, double b )
{
    ModelFn* fn = new ModelFn( make_regression_model_cppad( theta_init, y, X, lambda2, a, b ) );
    return Rcpp::XPtr<ModelFn>( fn, true );
}

// **************************
// ***   DDM 4 part
// **************************

// [[Rcpp::export]]

Rcpp::XPtr<ModelFn> build_ddm4_cppad_xptr(
    Eigen::VectorXd theta_init, // muss übergeben werden, um Tape zu initialisieren
    Eigen::VectorXd rts, Eigen::VectorXd xs, 
    Eigen::VectorXd muPrior_sp, Eigen::VectorXd sdPrior_sp,
    double min_rt,
    double s2, int kmax )
{
    ModelFn* fn = new ModelFn( make_ddm4_cppad( theta_init, rts, xs, muPrior_sp, sdPrior_sp, min_rt, s2, kmax ) );
    return Rcpp::XPtr<ModelFn>( fn, true );
}

// **************************
// ***   DDM 4 lan part
// **************************

// [[Rcpp::export]]

void lan_load_weights_export( const std::string& path, const std::string& ddm ) 
{
    lan_load_weights( path, ddm );
}

// [[Rcpp::export]]

double ddm4_lan_posterior_export( 
    const Eigen::VectorXd& theta,
    const Eigen::VectorXd& rts, const Eigen::VectorXd& xs, 
    const Eigen::VectorXd& muPrior_sp, const Eigen::VectorXd& sdPrior_sp,
    double min_rt ) 
{
    return ddm4_lan_posterior( theta, rts, xs, muPrior_sp, sdPrior_sp, min_rt );
}

// [[Rcpp::export]]

Rcpp::XPtr<ModelFn> ddm4_lan_xptr(
    const Eigen::VectorXd& rts, const Eigen::VectorXd& xs, 
    const Eigen::VectorXd& muPrior_sp, const Eigen::VectorXd& sdPrior_sp,
    double min_rt )
{
    ModelFn* fn = new ModelFn( make_ddm4_lan( rts, xs, muPrior_sp, sdPrior_sp, min_rt ) );
    return Rcpp::XPtr<ModelFn>( fn, true );
}

// **************************
// ***   DDM 7 part
// **************************

// [[Rcpp::export]]

Rcpp::XPtr<ModelFn> build_ddm7_cppad_xptr(
    Eigen::VectorXd theta_init, // muss übergeben werden, um Tape zu initialisieren
    Eigen::VectorXd rts, Eigen::VectorXd xs,
    Eigen::VectorXd pts, Eigen::VectorXd wgh, 
    Eigen::VectorXd muPrior_sp, Eigen::VectorXd sdPrior_sp,
    double min_rt, int kmax )
{
    ModelFn* fn = new ModelFn( make_ddm7_cppad( theta_init, rts, xs, pts, wgh, muPrior_sp, sdPrior_sp, min_rt, kmax ) );
    return Rcpp::XPtr<ModelFn>( fn, true );
}

// **************************
// ***   general part
// **************************

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
        Rcpp::Named("divergent") = out.divergent,
        Rcpp::Named("divergent_theta") = out.divergent_theta
    );
}
