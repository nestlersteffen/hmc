
#- function to tune the mass matrix:

#  Note: step_fn is either hmc_step or nuts_step depending on used algorithm

estimate_mass_matrix <- function( theta0=NULL, n_iter=NULL, step_fn=NULL, epsilon=NULL, verbose=FALSE )
{
    #- define matrices:
    p <- length( theta0 )
    warmup_samples      <- matrix( 0, nrow=n_iter, ncol=p )
    warmup_samples[1, ] <- theta0
 
 	#- let's go:
    for ( m in 2:n_iter ) {
 
        if ( verbose && m %% 5 == 0 ) print( paste0( "iteration: ", m ) )
 
        res <- step_fn( warmup_samples[m - 1, ], epsilon )
        warmup_samples[m, ] <- res$theta
    }

	#- compute new mass matrix: 
    M    <- diag( cov( warmup_samples ) )
    invM <- 1/M
 
    return( list( M=M, invM=invM, last_theta=warmup_samples[n_iter, ], warmup_samples=warmup_samples ) )
}