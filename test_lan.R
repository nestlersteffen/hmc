
# source("C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/R/dual_averaging.R")
# source("C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/R/estimate_mass_matrix.R")
# source("C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/R/make_args.R")
# source("C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/R/hmc_cpp.R")
# source("C:/Users/Steffen/Steffen Arbeitet/Gitordner/hmc/R/nuts_cpp.R")

source("E:/Github/hmc/R/dual_averaging.R")
source("E:/Github/hmc/R/estimate_mass_matrix.R")
source("E:/Github/hmc/R/make_args.R")
source("E:/Github/hmc/R/hmc_cpp.R")
source("E:/Github/hmc/R/nuts_cpp.R")

library( Rcpp )
Sys.setenv( PKG_CPPFLAGS = '-I"E:/Github/hmc/inst/include/"', 
            # PKG_LIBS = "-fopenmp",
            PKG_CXXFLAGS = "-O3 -mavx2 -mfma -DNDEBUG" )
sink("build_log.txt")
sourceCpp( "E:/Github/hmc/src/exports.cpp", rebuild=TRUE )
sink()

# Laden der Matrizen

path <- "E:/Github/hmc/inst/extdata/four/"
lan_load_weights_export( path=path, ddm="four" )

## %%%%%%%%%% Beispiel 1

# Daten einlesen:

df <- read.table("E:/Github/hmc/sim1.txt", header = TRUE )
colnames( df ) <- c("rt","xs") 
head( df )

v  <- 0.3
a  <- 0.5
z  <- 0.25
t0 <- 0.10
w  <- z/a
#inits1 <- c(v,log(a-z),log(z),log(t0))
inits2 <- c(v,log(a),qlogis(w), qlogis(t0/min( df$rt) ) )

ddm4_lan_posterior_export( inits2, df$rt, df$xs, c(0,0,0,0), c(1,1,1,1), min( df$rt ) )
numDeriv::grad( x=inits2, func=ddm4_lan_posterior_export,
    rts=df$rt, xs=df$xs, muPrior_sp=c(0,0,0,0), sdPrior_sp=c(1,1,1,1), 
    min_rt=min( df$rt ))

model_ptr <- ddm4_lan_xptr( rts=df$rt, xs=df$xs, 
    muPrior_sp=c(0,0,0,0), sdPrior_sp=c(1,1,1,1), 
    min( df$rt ) )
evaluate_model_ptr( inits2, model_ptr )

## %%%%%%%%%% Beispiel 2

set.seed(123)

v     <- 1.00; a <- 0.80; z <- 0.40; t0 <- 0.3
df    <- rtdists::rdiffusion( 100, a=a, v=v, t0=t0, z=z )
df$xs <- ifelse( df$response == "lower", 0, 1 )

v  <- 0
a  <- runif(1,0.5,2)
z  <- a/2
t0 <- min( df$rt ) + 0.1*runif(1,0,0.1)
w  <- z/a
# inits <- c( v, log(a-z), log(z), log(t0) )

inits <- c( v, log( a ), qlogis( w ), log( t0 )  )

ddm4_lan_posterior_export( inits, df$rt, df$xs, c(0,0,0,0), c(1,1,1,1), min( df$rt ) )
numDeriv::grad( x=inits, func=ddm4_lan_posterior_export,
    rts=df$rt, xs=df$xs, muPrior_sp=c(0,0,0,0), sdPrior_sp=c(1,1,1,1), 
    min_rt=min( df$rt ))

model_ptr <- ddm4_lan_xptr( rts=df$rt, xs=df$xs, 
    muPrior_sp=c(0,0,0,0), sdPrior_sp=c(1,1,1,1), 
    min( df$rt ) )
evaluate_model_ptr( inits, model_ptr )

model_ptr_test <- build_ddm4_cppad_xptr( theta_init=inits, rts=df$rt, xs=df$xs, 
    muPrior_sp=c(0,0,0,0), sdPrior_sp=c(1,1,1,1), 
    min( df$rt ), 1, 50 )
evaluate_model_ptr( inits, model_ptr_test )

## the hmc chain:

args <- make_args( biter=1000, burnin=500, badapt1=100, badapt2=100, max_depth_adapt=10 )

system.time({ 
test <- nuts_chain_cpp( model_ptr=model_ptr2, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
})
round( colMeans( test$parms ), 4)
mean( test$steps )
mean( test$alphas )
test$p_divergent_post

system.time({ 
test2 <- nuts_chain_cpp( model_ptr=model_ptr_test, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
})
round( colMeans( test2$parms ), 4)
mean( test2$steps )
mean( test2$alphas )
test2$p_divergent_post
