
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
#sink("build_log.txt")
sourceCpp( "E:/Github/hmc/src/exports.cpp", rebuild=TRUE )
#sink()

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

## timing

# library( microbenchmark)

# sample_sizes <- c(10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000 )
# n_rep <- 100
# results_tf <- list()

# for (n in sample_sizes) {
  
#     # make a dataset
#     df <- rtdists::rdiffusion( n = n, a=a, v=v, z=z, t0=t0, s=1, precision=3 )
#     df$xs <- ifelse( df$response == "lower", 0, 1 )
#     rt_n  <- df$rt
#     xs_n  <- df$xs
  
#     mb <- microbenchmark(
#         llfct = ddm4_lan_posterior_export( inits2, rt_n, xs_n, c(0,0,0,0), c(1,1,1,1), min( df$rt ) ), 
#         times = n_rep
#     )
  
#     mb_df <- as.data.frame(mb)
#     mb_df$n <- n
#     results_tf[[as.character(n)]] <- mb_df

# }

# results_tf <- do.call("rbind", results_tf )
# results_tf <- aggregate(time~expr+n,data=results_tf,
#     FUN=function(x) c( md=median(x)/1e6, m=mean(x)/1e6, sd=sd(x)/1e6) )
# final <- as.data.frame( cbind( results_tf$expr,results_tf$n,round(results_tf$time,5)) )
# colnames(final)[1:2] <- c("type","n")
# final
# #   type     n       md        m      sd
# # 1    1    10  0.02375  0.03686 0.10640
# # 2    1    50  0.08080  0.08342 0.01238
# # 3    1   100  0.15425  0.15853 0.02604
# # 4    1   500  0.79255  0.83013 0.09738
# # 5    1  1000  1.60900  1.64810 0.11008
# # 6    1  5000  8.32880  8.47542 0.34156
# # 7    1 10000 19.81130 19.82109 0.51512
# # 8    1 20000 47.08320 46.87887 1.21588
# # 9    1 30000 69.91540 69.92739 1.99648
numDeriv::grad( x=inits2, func=ddm4_lan_posterior_export,
    rts=df$rt, xs=df$xs, muPrior_sp=c(0,0,0,0), sdPrior_sp=c(1,1,1,1), 
    min_rt=min( df$rt ))

model_ptr <- ddm4_lan_xptr( rts=df$rt, xs=df$xs, 
    muPrior_sp=c(0,0,0,0), sdPrior_sp=c(1,1,1,1), 
    min( df$rt ) )
evaluate_model_ptr( inits2, model_ptr )

model_ptr2 <- ddm4_lan_batch_xptr( rts=df$rt, xs=df$xs, 
    muPrior_sp=c(0,0,0,0), sdPrior_sp=c(1,1,1,1), 
    min( df$rt ) )
evaluate_model_ptr( inits2, model_ptr2 )

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
test <- nuts_chain_cpp( model_ptr=model_ptr, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
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
