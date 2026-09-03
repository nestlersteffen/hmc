
#ifndef DDM_LAN_H
#define DDM_LAN_H

#include <RcppEigen.h>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "model_types.h"

// weights matrices for lan:
inline Eigen::MatrixXd W1, W2, W3, W4;
inline Eigen::VectorXd b1, b2, b3, b4;

inline Eigen::MatrixXd load_binary_matrix( const std::string& filename, int rows, int cols) 
{
    // file einlesen:
    std::ifstream file(filename, std::ios::binary);
    // stelle rows x cols double-Speicherplatz zur Verfügung
    std::vector<double> buffer(static_cast<size_t>(rows) * cols);
    // lese die binären Werte ein und wandle sie in doubles um
    file.read( reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(double) );
    // mappe das in eine Matrix 
    return Eigen::Map<Eigen::MatrixXd>( buffer.data(), rows, cols );
}

inline Eigen::VectorXd load_binary_vector(const std::string& filename) {

    // file einlesen:
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    // Länge des Vektor wird auf der Basis der Dateigröße bestimmt
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    // jetzt wieder einlesen und umwandeln
    long n = size / sizeof(double);
    std::vector<double> buffer(static_cast<size_t>(n));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    // mappen 
    return Eigen::Map<Eigen::VectorXd>(buffer.data(), n);
}

inline void lan_load_weights(const std::string& path, const std::string& ddm) 
{
    if (ddm == "seven") {
        W1 = load_binary_matrix(path + "W1.bin", 9, 100);
    } else {  // "four"
        W1 = load_binary_matrix(path + "W1.bin", 6, 100);
    }
 
    W2 = load_binary_matrix(path + "W2.bin", 100, 100);
    W3 = load_binary_matrix(path + "W3.bin", 100, 120);
    W4 = load_binary_matrix(path + "W4.bin", 120,   1);
 
    b1 = load_binary_vector(path + "b1.bin");
    b2 = load_binary_vector(path + "b2.bin");
    b3 = load_binary_vector(path + "b3.bin");
    b4 = load_binary_vector(path + "b4.bin");

}

// functions for forward call only:

inline Eigen::VectorXd lan_forward( const Eigen::MatrixXd& X ) 
{
    // Layer 1:
    Eigen::MatrixXd H1 = X * W1;                 
    H1.rowwise() += b1.transpose();
    H1 = H1.array().tanh();
    // Layer 2:
    Eigen::MatrixXd H2 = H1 * W2;                 
    H2.rowwise() += b2.transpose();
    H2 = H2.array().tanh();
    // Layer 3:
    Eigen::MatrixXd H3 = H2 * W3;                 
    H3.rowwise() += b3.transpose();
    H3 = H3.array().tanh();
    // Layer 4:
    Eigen::MatrixXd H4 = H3 * W4;                
    H4.rowwise() += b4.transpose();              
    return H4.col(0);                            
}

inline void lan_forward_backward( const Eigen::MatrixXd& X, const int& idx, 
    double& ll_data, Eigen::VectorXd& gr_data )
{
    
    // - Forward-Pass:
    
    // Layer 1:
    Eigen::MatrixXd H1 = X * W1;                 
    H1.rowwise() += b1.transpose();
    H1 = H1.array().tanh();
    // Layer 2:
    Eigen::MatrixXd H2 = H1 * W2;                 
    H2.rowwise() += b2.transpose();
    H2 = H2.array().tanh();
    // Layer 3:
    Eigen::MatrixXd H3 = H2 * W3;                 
    H3.rowwise() += b3.transpose();
    H3 = H3.array().tanh();
    // Layer 4:
    Eigen::MatrixXd H4 = H3 * W4;                
    H4.rowwise() += b4.transpose();              
    ll_data = H4.col(0).sum();  

    // - Backward-Pass:
    int n = X.rows();
    Eigen::MatrixXd D4 = Eigen::MatrixXd::Ones(n, 1);   // dy_i/dz4_i = 1 fuer alle i
    Eigen::MatrixXd D3 = (D4 * W4.transpose()).array() * (1.0 - H3.array().square());
    Eigen::MatrixXd D2 = (D3 * W3.transpose()).array() * (1.0 - H2.array().square());
    Eigen::MatrixXd D1 = (D2 * W2.transpose()).array() * (1.0 - H1.array().square());
    Eigen::MatrixXd GradInput = D1 * W1.transpose();    // (n x 6)
    gr_data = GradInput.leftCols(idx).colwise().sum().transpose();

}

// function to compute the posterior -- to check the computations

inline double ddm4_lan_posterior( const Eigen::VectorXd& theta,
    const Eigen::VectorXd& rts, const Eigen::VectorXd& xs, 
    const Eigen::VectorXd& muPrior_sp, const Eigen::VectorXd& sdPrior_sp,
    const double& min_rt ) 
{
    
    // Eigen::setNbThreads(1);

    // some general args:
    int p = theta.size();
    int n = rts.size();

    // transform parameters:
    double v, a, z, t0;
    v  = theta(0);
    // a  = std::exp( theta(1) ) + std::exp( theta(2) );
    // z  = std::exp( theta(2) );
    // t0 = std::exp( theta(3) );
    a  = std::exp( theta(1) );
    z  = a*( 1.0 / ( 1.0 + std::exp( -theta(2) ) ) );
    t0 = min_rt / ( 1.0 + std::exp(-theta(3) ) );

    // compute llfct:
    Eigen::MatrixXd X(n, 6);
    X.col(0).setConstant(a);
    X.col(1).setConstant(v);
    X.col(2).setConstant(t0);
    X.col(3).setConstant(z);
    X.col(4) = xs;
    X.col(5) = rts;
    Eigen::VectorXd ll_values = lan_forward(X);
    double ll = ll_values.sum();

    // compute prior:
    double ll_p = 0.0;
    for (int i = 0; i < p; i++) {
        double mu = muPrior_sp(i);
        double sd = sdPrior_sp(i);
        double resid = theta(i) - mu;
        ll_p += -std::log(sd) - (resid*resid)/(2*sd*sd);
    }

    // return ...
    double out = -1.0*(ll+ll_p);
    return out;
}

inline ModelResult ddm4_lan( const Eigen::VectorXd& theta, 
    const Eigen::VectorXd& rts, const Eigen::VectorXd& xs,
    const Eigen::VectorXd& muPrior_sp, const Eigen::VectorXd& sdPrior_sp,
    const double min_rt ) 
{
    
    //- transform parameters:
    double v, a, z, t0;
    v  = theta(0);
    // a  = std::exp( theta(1) ) + std::exp( theta(2) );
    // z  = std::exp( theta(2) );
    // t0 = std::exp( theta(3) );
    a  = std::exp( theta(1) );
    double w = 1 / ( 1.0 + std::exp(-theta(2) ) );
    z  = a*w;
    double u_t0 = 1 / ( 1.0 + std::exp(-theta(3) ) );
    t0 = min_rt*u_t0;

    //- get information for the loop:
    int n = rts.size();
    int p = theta.size();

    //- make the forward pass to obtain the data ll-value:
    Eigen::MatrixXd X(n, 6);
    X.col(0).setConstant(a);
    X.col(1).setConstant(v);
    X.col(2).setConstant(t0);
    X.col(3).setConstant(z);
    X.col(4) = xs;
    X.col(5) = rts;
    
    double ll_data;
    Eigen::VectorXd gr_data;
    lan_forward_backward(X,4,ll_data,gr_data);

    //- compute prior part:
    double ll_p = 0.0;
    for (int i = 0; i < p; i++) {
        double mu = muPrior_sp(i);
        double sd = sdPrior_sp(i);
        double resid = theta(i) - mu;
        ll_p += -std::log(sd) - (resid*resid)/(2*sd*sd);
    }

    //- first part of gradient:
    Eigen::MatrixXd J(4,4);
    J.setZero();
    J(0,1) = a; 
    J(1,0) = 1;
    J(2,3) = t0*(1-u_t0);
    J(3,1) = z; J(3,2) = z*(1-w);
    Eigen::VectorXd grad = J.transpose()*gr_data;

    //- now we add the prior part of the gradient:
    for (int i = 0; i < p; i++) {
        double mu = muPrior_sp(i);
        double sd = sdPrior_sp(i);
        double resid = theta(i) - mu;
        grad(i) += -resid/(sd*sd);
    }

    //- make the list:
    ModelResult res;
    res.fn = -1*( ll_data + ll_p );
    res.gr = -1*grad;
    return res;

}

#endif // DDM_LAN_H 