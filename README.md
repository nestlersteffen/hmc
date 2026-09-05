# hmcX

This package provides functions implementing a HMC-sampler and a NUTS-Sampler in R and C++. Define your own models in R and C++ with CppAD and then use hmc_chain (hmc_hain_cpp) or nuts_chain (nuts_chain_cpp) to obtain a single chain of the HMC or a NUTS-sampler. Both functions presume that your model is implemented in a function that provides the negative value of the log-posterior and the gradient of this function. 

Note that I implemented these functions to better understand STAN. You are free to use it, but there is no guarantee for correctness. See the examples on how to implement your model and to use the functions.

## Installation

``` r
# install.packages("devtools")
devtools::install_github("nestlersteffen/hmcX")
```

To update, simply rerun the installation command.

## Examples

The package contains a regression example. The function that computes the negative log-posterior and the gradient is shipped with the package. Note that is a closure allowing the function to remember and access function arguments passed to it at initialization. 

``` r

# make some data:

set.seed(123)
n <- 1000
X <- mvtnorm::rmvnorm(n,rep(0,2),matrix(c(1,0.3,0.3,1),2,2))
y <- 3 + X%*%c(0.4,0.2) + rnorm(n,0,sqrt(1.5))

# initialize the closure (lambda2, a, and b are parameters of the priors)

my_regression_model <- make_regression( y=y, X=cbind(1,X), lambda2=10, a=1, b=1)

# some initial values for the chain (we sample the residual sd on the log-scale)

inits <- c( rep(0, 3), log( sd( y ) ) )

# define some arguments (see ?make_args):

args  <- make_args( biter=2000, burnin=1000 ) #biter = length of the chain

# do NUTS-sampling:

fit <- nuts_chain_r(model_fn=my_regression_model, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )

# fit is a list (I do not provide nice summary functions)

```

## Contributing

Issues and pull requests are not actively monitored. For questions, 
suggestions, or bug reports, please contact me directly via mail.

## Status

Work in progress.