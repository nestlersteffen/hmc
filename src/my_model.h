#ifndef MY_MODEL_H
#define MY_MODEL_H

#include <RcppEigen.h>
#include <cppad/cppad.hpp>
#include <memory>
#include <functional>
#include "model_types.h"
#include "regression.h"
#include "ddm4.h"
#include "ddm4_lan.h"
#include "ddm7.h"

// **************************
// ***     REGRESSION
// **************************

inline ModelFn make_regression_model(const Eigen::VectorXd& y, const Eigen::MatrixXd& X,
    double lambda2, double a, double b)
{
    return [y, X, lambda2, a, b](const Eigen::VectorXd& theta) {
        return regression(theta, y, X, lambda2, a, b);
    };
}

inline ModelFn make_regression_model_cppad( const Eigen::VectorXd& theta_init,
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

// **************************
// ***     DDM 4
// **************************

inline ModelFn make_ddm4_cppad( const Eigen::VectorXd& theta_init,
    const Eigen::VectorXd& rts, const Eigen::VectorXd& xs,
    const Eigen::VectorXd& muPrior_sp, const Eigen::VectorXd& sdPrior_sp,
    const double min_rt,
    const double s2, const int kmax )
{
    
    // make the tape:
    auto tape = std::make_shared<CppAD::ADFun<double>>(
        ddm4_tape( theta_init, rts, xs, muPrior_sp, sdPrior_sp, min_rt, s2, kmax ) );

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

// **************************
// ***     DDM 4 - LAN
// **************************

inline ModelFn make_ddm4_lan(
    const Eigen::VectorXd& rts, const Eigen::VectorXd& xs,
    const Eigen::VectorXd& muPrior_sp, const Eigen::VectorXd& sdPrior_sp,
    const double min_rt )
{
    return [rts, xs, muPrior_sp, sdPrior_sp, min_rt](const Eigen::VectorXd& theta) {
        return ddm4_lan( theta, rts, xs, muPrior_sp, sdPrior_sp, min_rt );
    };
}

inline ModelFn make_ddm4_lan_batch(
    const Eigen::VectorXd& rts, const Eigen::VectorXd& xs,
    const Eigen::VectorXd& muPrior_sp, const Eigen::VectorXd& sdPrior_sp,
    const double min_rt )
{
    return [rts, xs, muPrior_sp, sdPrior_sp, min_rt](const Eigen::VectorXd& theta) {
        return ddm4_lan_batch( theta, rts, xs, muPrior_sp, sdPrior_sp, min_rt );
    };
}

// **************************
// ***     DDM 7
// **************************

inline ModelFn make_ddm7_cppad( const Eigen::VectorXd& theta_init,
    const Eigen::VectorXd& rts, const Eigen::VectorXd& xs,
    const Eigen::VectorXd& pts, const Eigen::VectorXd& wgh,
    const Eigen::VectorXd& muPrior_sp, const Eigen::VectorXd& sdPrior_sp,
    const double min_rt, const int kmax )
{
    
    // make the tape:
    auto tape = std::make_shared<CppAD::ADFun<double>>(
        ddm7_tape( theta_init, rts, xs, pts, wgh, 
            muPrior_sp, sdPrior_sp, min_rt, kmax ) );

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

#endif // MY_MODEL_H