
source("E:/Github/hmc/R/helpers.R")
source("E:/Github/hmc/R/leapfrog.R")
source("E:/Github/hmc/R/hmc.R")
source("E:/Github/hmc/R/nuts.R")

source("E:/Github/hmc/R/dual_averaging.R")
source("E:/Github/hmc/R/estimate_mass_matrix.R")
source("E:/Github/hmc/R/make_args.R")




# %%% Regression:

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

