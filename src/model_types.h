#ifndef MODEL_TYPES_H
#define MODEL_TYPES_H

#include <RcppEigen.h>
#include <functional>

// cpp - List type object:

struct ModelResult {
	double fn;
	Eigen::VectorXd gr;
};

using ModelFn = std::function<ModelResult(const Eigen::VectorXd&)>;

#endif // MODEL_TYPES_H