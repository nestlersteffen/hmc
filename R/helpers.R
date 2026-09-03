
#- Hamiltonian energy: H(theta, r) = U(theta) + 0.5 * r' invM r

#  since invM is a vector, we use 0.5*sum(r^2*invM)
 
compute_H_r <- function( theta=NULL, r=NULL, model_fn=NULL, invM=NULL, fn_val=NULL )
{
    if ( is.null( fn_val ) ) fn_val <- model_fn( theta )$fn
    as.numeric( fn_val + 0.5 * sum( r^2 * invM ) )
}

#- function to find a reasonable first espilon value (see Hoffmann & Gelman, 2014, algorithm 4)

find_reasonable_epsilon_r <- function( theta=NULL, model_fn=NULL, M=NULL, invM=NULL ) 
{
    
    p       <- length(theta)
    epsilon <- 1.0
    
	#- start point
    r0 <- rnorm( p, mean=0, sd=sqrt(M) )
    H0 <- compute_H_r( theta=theta, r=r0, model_fn=model_fn, invM=invM )
    
    #- make a leapfrog step
    step <- leapfrog_step_r( theta=theta, r=r0, model_fn=model_fn, invM=invM, epsilon=epsilon )
    H1   <- compute_H_r( r=step$r, invM=invM, fn_val=step$fn )
    
    #- Sicherheitscheck vor Richtungsbestimmung
    if ( is.nan( H1 ) || is.infinite( H1 ) ) return( 0.001 )
    
    #- now, iterate:
    alpha     <- min(1, exp(as.numeric(H0 - H1)))
    direction <- if (alpha > 0.5) 1 else -1
    max_iter  <- 50
    iter      <- 0
    
    while ( iter < max_iter ) {
        # adapt epsilon:
        iter    <- iter + 1
        epsilon <- epsilon * 2^direction
        # make the leapfrog 
        step <- leapfrog_step_r( theta=theta, r=r0, model_fn=model_fn, invM=invM, epsilon=epsilon )
    	H1   <- compute_H_r( r=step$r, invM=invM, fn_val=step$fn )
        # check whether we can leave the loop
        alpha <- min( 1, exp( as.numeric( H0 - H1 ) ) )
        if (direction ==  1 && alpha <= 0.5) break
        if (direction == -1 && alpha >= 0.5) break
    }
    
    if ( iter == max_iter ) warning("find_reasonable_epsilon: max_iter erreicht")
    
    return(epsilon)
}