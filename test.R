
source("C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/R/dual_averaging.R")
source("C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/R/estimate_mass_matrix.R")
source("C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/R/make_args.R")
source("C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/R/hmc_cpp.R")
source("C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/R/nuts_cpp.R")

library( Rcpp )
Sys.setenv(PKG_CPPFLAGS = '-I"C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/inst/include/"')
sourceCpp( "C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/src/exports.cpp", rebuild=TRUE )

# script <- '
# library(Rcpp)
# Sys.setenv(PKG_CPPFLAGS = \'-I"C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/inst/include/" -fmax-errors=5\')
# sourceCpp("C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/src/exports.cpp", rebuild = TRUE)
# '
# writeLines(script, "build.R")

# system2("Rscript", "build.R",
#         stdout = "compile_log.txt",
#         stderr = "compile_log.txt")

# %%%%%%%%%%%%%%%%%%%%%%%
# %%% Regression

set.seed(123)
n <- 1000
X <- mvtnorm::rmvnorm(n,rep(0,2),matrix(c(1,0.3,0.3,1),2,2))
y <- 3 + X%*%c(0.4,0.2) + rnorm(n,0,sqrt(1.5))

inits <- c( rep(0, 3), log( sd( y ) ) )
args  <- make_args(biter=1000, burnin=500 )

# build the pointer:

model_ptr <- build_regression_model_cppad_xptr( theta_init=inits, y=y, X=cbind(1,X), lambda2=10, a=1, b=1 )
evaluate_model_ptr( inits, model_ptr )

# now sample:

test <- hmc_chain_cpp( model_ptr=model_ptr, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
round( colMeans( test$parms ), 4)
test$p_accept_post
test$p_divergent_post

test <- nuts_chain_cpp(model_ptr=model_ptr, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
round( colMeans( test$parms ), 4)
mean( test$steps )
mean( test$alphas )
test$p_divergent_post

# %%%%%%%%%%%%%%%
# %%% DDM 4:

set.seed(123)

v     <- 1.00; a <- 0.80; z <- 0.40; t0 <- 0.3
df    <- rtdists::rdiffusion( 100, a=a, v=v, t0=t0, z=z )
df$xs <- ifelse( df$response == "lower", 0, 1 )

v  <- 0
a  <- runif(1,0.5,2)
z  <- a/2
w  <- z/a
t0 <- min( df$rt ) + 0.1*runif(1,0,0.1)
#inits <- c( v, log( a - z ), log( z ), log( t0 )  )
inits <- c( v, log( a ), qlogis( w ), log( t0 )  )

args  <- make_args(biter=1000, burnin=500, badapt1=100, badapt2=100, max_depth_adapt=10 )

# build the pointer:

model_ptr <- build_ddm4_cppad_xptr( theta_init=inits, rts=df$rt, xs=df$xs, 
    muPrior_sp=c(0,0,0,0), sdPrior_sp=c(1,1,1,1), 
    #muPrior_sp=c(0,-0.693,-0.693,-0.83), sdPrior_sp=c(3,0.6,0.3,0.28), 
    min( df$rt ), 1, 50 )
evaluate_model_ptr( inits, model_ptr )
# $fn
# [1] 371.5621

# $gr
# [1]  -26.59969  522.23221  297.85680 4323.16106

# test <- hmc_chain_cpp( model_ptr=model_ptr, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
# round( colMeans( test$parms ), 4)
# test$p_accept_post
# test$p_divergent_post

test <- nuts_chain_cpp( model_ptr=model_ptr, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
round( colMeans( test$parms ), 4)
mean( test$steps )
mean( test$alphas )
test$p_divergent_post

(apply(test$divergent_samples[rowSums(test$divergent_samples) != 0, ], 2, sd))
apply( test$parms, 2, sd )

# %%%%%%%%%%%%%%%
# %%% DDM 7:

ddm7_simulate <- function( ni=100, alpha=NULL, type_alpha=NULL  )
{
    #- transform parameters
    v   <- alpha[1]
    sv  <- exp(alpha[5])
    sz  <- exp(alpha[6])
    st0 <- exp(alpha[7])
    z   <- exp(alpha[3]) + 0.5*sz
    t0  <- exp(alpha[4]) + 0.5*st0
    a   <- exp(alpha[2]) + z + 0.5*sz
    #- now simulate the data:
    sim_sample <- rtdists::rdiffusion( n = ni, a=a, v=v, z=z, sz=sz, sv=sv, t0=t0-0.5*st0,
        st0=st0, s=1, precision=3 )
    sim_sample$xs <- ifelse( sim_sample$response == "lower", 0, 1 )
    sim_sample <- sim_sample[,c( "rt", "xs" )]
    return( sim_sample )
}

nll_ddm7 <- function( alpha=NULL, rts, xs )
{
    #- transform parameters
    v   <- alpha[1]
    sv  <- exp(alpha[5])
    sz  <- exp(alpha[6])
    st0 <- exp(alpha[7])
    z   <- exp(alpha[3]) + 0.5*sz
    t0  <- exp(alpha[4]) + 0.5*st0
    a   <- exp(alpha[2]) + z + 0.5*sz
    #- now simulate the data:
    ll <- rtdists::ddiffusion( 
        rts, xs+1, a=a, v=v, z=z, sz=sz, sv=sv, t0=t0-0.5*st0,
        st0=st0, s=1, precision=3 )
    if ( any( ll == 0) | any( ll == Inf ) ) {
        idx <- which( ll == 0 | ll == Inf ) 
        ll[idx] <- 10^-29
    }
    -sum( log( ll ) )
}

pts <- c(-0.973906528517172, -0.865063366688985, -0.679409568299024,
             -0.433395394129247, -0.148874338981631,  0.148874338981631,
              0.433395394129247,  0.679409568299024,  0.865063366688985, 0.973906528517172)
wgh <- c(0.066671344308688, 0.149451349150581, 0.219086362515982,
             0.269266719309996, 0.295524224714753, 0.295524224714753,
             0.269266719309996, 0.219086362515982, 0.149451349150581, 0.066671344308688)

set.seed(123)

v  <- 1.00; a <- 0.80; z <- 0.40; t0 <- 0.3; sv <- 1; sz <- 0.3; st0 <- 0.15
MU <- c(v, log(a - z - 0.5*sz), log(z - 0.5*sz), log(t0 - 0.5*st0), log(sv), log(sz), log(st0))
df <- ddm7_simulate( ni = 100, alpha = MU )

inits <- c( 0.8, log( 0.6 - 0.3 - 0.5*0.25 ), 
                log( 0.3 - 0.5*0.25), 
                log( 0.2 - 0.5*0.1 ), 
                log( 0.8 ), log( 0.25 ), log( 0.10 ) )
args  <- make_args(biter=500, burnin=250, badapt1=100, badapt2=100, max_depth_adapt=10 )

nll_ddm7( inits, df$rt, df$xs )
numDeriv::grad( x=inits, func=nll_ddm7, rts=df$rt, xs=df$xs)

# build the pointer:

model_ptr <- build_ddm7_cppad_xptr( 
    theta_init=inits, 
    rts=df$rt, xs=df$xs, pts=pts, wgh=wgh, 
    muPrior_sp=c(0,0,0,0,0,0,0), sdPrior_sp=c(1,1,1,1,1,1,1), 
    min( df$rt ), 20 )
evaluate_model_ptr( inits, model_ptr )

test <- nuts_chain_cpp( model_ptr=model_ptr, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
round( colMeans( test$parms ), 4)
mean( test$steps )
mean( test$alphas )
test$p_divergent_post