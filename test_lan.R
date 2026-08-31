
nll4 <- function( parm, rt, xs ) {
    ll <- rtdists::ddiffusion( rt, xs + 1, a=parm[2], v=parm[1], t0=parm[4], z=parm[3] )
    if ( any( ll == 0) | any( ll == Inf ) ) {
         idx <- which( ll == 0 | ll == Inf ) 
         ll[idx] <- 10^-29
    }
    -sum( log( ll ) )
}

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
Sys.setenv(PKG_CPPFLAGS = '-I"E:/Github/hmc/inst/include/"')

sink("build_log.txt")
sourceCpp( "E:/Github/hmc/src/exports.cpp", rebuild=TRUE )
sink()

# Laden der Matrizen

path <- "E:/Github/hmc/inst/extdata/four/"
lan_load_weights_export( path=path, ddm="four" )

# llfct der Daten:

set.seed(123)

v     <- 1.00; a <- 0.80; z <- 0.40; t0 <- 0.3
df    <- rtdists::rdiffusion( 10000, a=a, v=v, t0=t0, z=z )
df$xs <- ifelse( df$response == "lower", 0, 1 )

v  <- 0
a  <- runif(1,0.5,2)
z  <- a/2
t0 <- min( df$rt ) + 0.1*runif(1,0,0.1)
w  <- z/a

system.time({ 
    ll <- nll4( c(v,a,z,t0), df$rt[1], df$xs[1] ) 
})

numDeriv::grad( x=c(v,a,z,t0), func=nll4, rt=df$rt[1], xs=df$xs[1] )

ddm4_lan_llfct_cpp( df$rt, df$xs, v, a, t0, z )

lan_forward_backward_cpp( c(a,v,t0,z,df$xs[1],df$rt[1] ), 4 )

model_ptr <- ddm4_lan_xptr( rts=df$rt, xs=df$xs, 
    #muPrior_sp=c(0,0,0,0), sdPrior_sp=c(1,1,1,1), 
    min( df$rt ) )

system.time({
    evaluate_model_ptr( c(v,log(a-z),log(z),log(t0)), model_ptr )
})



# %%%%%%%%%%%%%%%
# %%% DDM 4: analytic

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

test <- nuts_chain_cpp( model_ptr=model_ptr, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
round( colMeans( test$parms ), 4)
mean( test$steps )
mean( test$alphas )
test$p_divergent_post

# %%%%%%%%%%%%%%%
# %%% DDM 4: neural net




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