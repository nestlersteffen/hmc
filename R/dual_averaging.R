
#- function implmenting dual averging (see Hoffmann & Gelman, 2014)

#  Note: step_fn is either hmc_step or nuts_step depending on used algorithm

dual_averaging <- function( theta0=NULL, n_iter=NULL, step_fn=NULL, epsilon_init=NULL, 
	args=NULL, verbose=FALSE )
{
    
    #- collect args:
    delta <- args$adapt$delta
    gamma <- args$adapt$gamma
    t0    <- args$adapt$t0
    kappa <- args$adapt$kappa

    #- define parameters:
    mu          <- log( 10 * epsilon_init )
    D_bar       <- 0
    eps_bar     <- epsilon_init
    log_eps_bar <- log( eps_bar )
    theta       <- theta0
 	#- let's go:
    for ( m in 1:n_iter ) {
 
        if ( verbose && m %% 5 == 0 ) print( paste0( "iteration: ", m ) )
        
        #- make a step:
        res   <- step_fn( theta, eps_bar )
        theta <- res$theta
        alpha <- res$alpha
 
        #- compute D_bar:
        D_bar <- ( 1 - 1/(m + t0) ) * D_bar + ( 1/(m + t0) ) * ( delta - alpha )
 
        #- now obtain new log( epsilon )
        log_eps_bar <- m^(-kappa) * ( mu - sqrt(m)/gamma * D_bar ) +
            ( 1 - m^(-kappa) ) * log_eps_bar
 
        # if ( is.nan( log_eps_bar ) || is.infinite( log_eps_bar ) ) {
        #     log_eps_bar <- log( 0.001 )
        # }
        eps_bar <- exp( log_eps_bar )
        # print( paste0("m: ", m, " | eps_bar: ", eps_bar ))
    }
    return( list( theta=theta, epsilon=eps_bar ) )
}