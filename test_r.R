
source("E:/Github/hmc/R/helpers.R")
source("E:/Github/hmc/R/leapfrog.R")
source("E:/Github/hmc/R/hmc.R")
source("E:/Github/hmc/R/nuts.R")
source("E:/Github/hmc/R/dual_averaging.R")
source("E:/Github/hmc/R/estimate_mass_matrix.R")
source("E:/Github/hmc/R/make_args.R")
source("E:/Github/hmc/R/examples.R")

# %%%%%%%%%%%%%%%%%%%%%%%%
# %%%    Regression

set.seed(123)
n <- 1000
X <- mvtnorm::rmvnorm(n,rep(0,2),matrix(c(1,0.3,0.3,1),2,2))
y <- 3 + X%*%c(0.4,0.2) + rnorm(n,0,sqrt(1.5))

inits <- c( rep(0, 3), log( sd( y ) ) )
args  <- make_args(biter=1000, burnin=500 )

# build the pointer:

model_ptr_r <- make_regression( y=y, X=cbind(1,X), lambda2=10, a=1, b=1)
model_ptr_r( inits )

# now sample:

test <- hmc_chain_r( model_fn=model_ptr_r, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
round( colMeans( test$parms ), 4)
test$p_accept_post
test$p_divergent_post

test <- nuts_chain_r(model_fn=model_ptr_r, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
round( colMeans( test$parms ), 4)
mean( test$steps )
mean( test$alphas )
test$p_divergent_post

# %%%%%%%%%%%%%
# %%%  DDM 4

library( keras3 )
library( tensorflow )

#- load the trained model and make a warm-start:

model <- keras3::load_model( "E:/Github/hmc/inst/extdata/four/Trained_Model_4param.keras" )
df    <- rtdists::rdiffusion( 1, a=0.50, v=0, t0=0.1, z=0.25 )
df$xs <- ifelse( df$response == "lower", 0, 1 )
invisible( model( matrix( c( 0.5, 0, 0.25, 0.1, df$rt[1], df$xs[1] ), nrow=1 ) ) )

#- define a function that we can use in the sampler:

make_ddm4_with_lan <- function( dnn=NULL, rt=NULL, xs=NULL, 
    muPrior_sp=NULL, sdPrior_sp=NULL, min_rt=NULL )
{
    function( theta ) {

        #- get DDM parms on correct scale:
        v    <- theta[1] 
        a    <- exp( theta[2] )
        w    <- 1 / (1 + exp( -theta[3] ) )
        z    <- a*w
        u_t0 <- 1/(1 + exp( -theta[4] ) )
        t0   <- min_rt*u_t0 
        
        #- no. of reaction times and number of parameters
        n <- length( rt )
        p <- length( theta )
        
        #- parameter to "watch" for:
        a0  <- tf$Variable(a,  dtype = tf$float32)
        v0  <- tf$Variable(v,  dtype = tf$float32)
        z0  <- tf$Variable(z,  dtype = tf$float32)
        t00 <- tf$Variable(t0, dtype = tf$float32)
        
        #- data are constants:
        rt_tensor <- tf$constant( rt, dtype = tf$float32)
        xs_tensor <- tf$constant( xs, dtype = tf$float32)
        ones_n    <- tf$ones( shape = list(n), dtype = tf$float32 )
        
        # %%% compute part of the data:

        with( tf$GradientTape() %as% tape, {
            
            # broadcast parms to length n:
            col_a  <- a0  * ones_n
            col_v  <- v0  * ones_n
            col_t0 <- t00 * ones_n
            col_z  <- z0  * ones_n
            # make a temporary data matrix:
            tmpMat <- tf$stack( list( col_a, col_v, col_t0, col_z, xs_tensor, rt_tensor), axis = 1L)
            # compute ll value
            ll <- dnn( tmpMat )
            ll_sum <- tf$reduce_sum(ll)
        } )
        
        #- get gradient for v, a, t0 and w
        gr <- tape$gradient( ll_sum, list(a0, v0, t00, z0 ) )
        gr <- lapply( gr, function(x) as.array( x ) )
        gr <- do.call("c", gr )
        
        #- we have to map the gradient to theta
        J <- matrix(0,4,4)
        J[1,2] <- a; J[2,1] <- 1; J[3,4] <- t0*(1-u_t0);
        J[4,2] <- z; J[4,3] <- z*(1-w);
        gr <- t(J) %*% gr
        
        # %%% add things for the prior
        etheta <- theta - muPrior_sp
        ll_p   <- -log( sdPrior_sp ) - 0.5*( etheta*etheta )/( sdPrior_sp*sdPrior_sp )
        gr_p   <- -etheta/( sdPrior_sp*sdPrior_sp )
                
        #- make output:
        return( list( fn=-1*( as.array( ll_sum ) + sum( ll_p ) ), gr=-1*( as.vector( gr ) + gr_p ) ) )
    }

}

set.seed(123)

v     <- 1.00; a <- 0.80; z <- 0.40; t0 <- 0.3
df    <- rtdists::rdiffusion( 100, a=a, v=v, t0=t0, z=z )
df$xs <- ifelse( df$response == "lower", 0, 1 )

v  <- 0
a  <- runif(1,0.5,2)
z  <- a/2
w  <- z/a
t0 <- min( df$rt ) + 0.1*runif(1,0,0.1)
inits <- c( v, log( a ), qlogis( w ), log( t0 )  )

args  <- make_args(biter=1000, burnin=500, badapt1=100, badapt2=100, max_depth_adapt=10 )

# build the pointer:

model_ptr_r <- make_ddm4_with_lan( dnn=model, rt=df$rt, xs=df$xs, 
    muPrior_sp=c(0,0,0,0), sdPrior_sp=c(1,1,1,1), min( df$rt ) )
model_ptr_r( inits )

system.time({ 
    test <- nuts_chain_r(model_fn=model_ptr_r, args=args, verbose=TRUE, inits=inits, find_epsilon=TRUE )
})
round( colMeans( test$parms ), 4)
mean( test$parms[,1] )
mean( exp( test$parms[,2] ) )
mean( exp( test$parms[,2] )*plogis( test$parms[,3] ) )
mean( min( df$rt )*plogis( test$parms[,4] ) )
mean( test$steps )
mean( test$alphas )
test$p_divergent_post
