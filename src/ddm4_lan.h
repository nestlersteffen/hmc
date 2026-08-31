
#ifndef DDM4_LAN_H
#define DDM4_LAN_H

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

    Rcpp::Rcout << "Weights loaded from " << path << std::endl;

}

inline double lan_forward(const Eigen::VectorXd& input )
{
    Eigen::VectorXd x = input;

    x = (W1.transpose() * x + b1).array().tanh();
    x = (W2.transpose() * x + b2).array().tanh();
    x = (W3.transpose() * x + b3).array().tanh();
    x = W4.transpose() * x + b4;  // linear, letzte Schicht ohne Aktivierung

    return x(0);
}

inline Eigen::VectorXd lan_forward_backward( const Eigen::VectorXd& input, const int& idx )
{
    
    // - Aktivierungen fuer Forward- und Backward-Pass zwischenspeichern:
    std::vector<Eigen::VectorXd> activations(5);

    // - Forward-Pass:
    activations[0] = input;
    Eigen::VectorXd x = input;

    x = (W1.transpose() * x + b1).array().tanh();
    activations[1] = x;
    x = (W2.transpose() * x + b2).array().tanh();
    activations[2] = x;
    x = (W3.transpose() * x + b3).array().tanh();
    activations[3] = x;
    x = W4.transpose() * x + b4;
    activations[4] = x;
    double output = x(0);

    // - Backward-Pass:
    Eigen::VectorXd d(1);
    d(0) = 1.0;
    d = (W4 * d).array() * (1.0 - activations[3].array().square());
    d = (W3 * d).array() * (1.0 - activations[2].array().square());
    d = (W2 * d).array() * (1.0 - activations[1].array().square());

    // - Gradient bzgl. des Inputs:
    Eigen::VectorXd grad_input = W1 * d;

    // - Output + erste idx Gradienten-Eintraege zusammenstellen:
    Eigen::VectorXd result(1 + idx);
    result(0) = output;
    result.segment(1, idx) = grad_input.head(idx);
    return result;
}

inline double ddm4_lan_llfct( const Eigen::VectorXd& rts, const Eigen::VectorXd& xs, 
     const double& v, const double& a, const double& t0, const double& z ) 
{
    // some prelims:
    const int n = rts.size();
    double ll = 0.0;
    Eigen::VectorXd input(6);
    // loop through the data:
    for ( int i = 0; i < n; i++ ) {
        input(0) = a;
        input(1) = v;
        input(2) = t0;
        input(3) = z;
        input(4) = xs(i);
        input(5) = rts(i);
        ll += lan_forward( input );
    }
    return ll;
}

inline ModelResult ddm4_lan( const Eigen::VectorXd& theta, 
    const Eigen::VectorXd& rts, const Eigen::VectorXd& xs,
    // const Eigen::VectorXd& muPrior_sp, const Eigen::VectorXd& sdPrior_sp,
    const double min_rt ) 
{
    
    //- transform parameters:
    double v, a, z, t0;
    v  = theta(0);
    a  = std::exp( theta(1) ) + std::exp( theta(2) );
    z  = std::exp( theta(2) );
    t0 = std::exp( theta(3) );

    //- get information for the loop:
    int n = rts.size();
    Eigen::VectorXd output(5);
    output.setZero();
    Eigen::VectorXd input(6);
    input.setZero();
    
    //- loop:
    for ( int i = 0; i < n; i++ ) {
        input(0) = a;
        input(1) = v;
        input(2) = t0;
        input(3) = z;
        input(4) = xs(i);
        input(5) = rts(i);
        output += lan_forward_backward( input, 4 );
    }

    //- final step:
    // arma::mat J( 4, 4, arma::fill::zeros );
    // J(0,1) = exp( alpha( 1 ) ); 
    // J(0,2) = z; J(1,0) = 1; J(2,3) = t0; J(3,2) = z;
    // arma::vec grad_us = J.t()*output.subvec(1,4);

    //- make the list:
    ModelResult res;
    res.fn = output(0);
    res.gr = output.segment(1, 4);
    return res;

}

#endif // DDM4_LAN_H 