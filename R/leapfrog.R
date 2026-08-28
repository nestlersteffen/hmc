
#-  a single, symmetric leapfrog step:
#
#   if this step is done L-times with direction=1, then this is a HMC step.
#   For NUTS we need direction=-1, to integrate backwards in time
#
#   theta, r: numeric vectors
#   model_fn: function(theta) -> list( fn=.., gr=...)
#   invM: a numeric vector - diagonal of the inverse mass matrix
#   epsilon: stepsize
#   direction: +1 ( forward) or -1 ( backwards), default=1

leapfrog_step_r <- function( theta=NULL, r=NULL, model_fn=NULL, invM=NULL, 
	epsilon=NULL, direction=1 )
{
	#- compute half kick:
	g0    <- model_fn( theta )$gr
	r     <- r - direction*epsilon/2 * g0
    #- drift:
    theta <- theta + direction*epsilon * invM * r
    #- second half kick
    res   <- model_fn( theta )
    r     <- r - direction*epsilon/2 * res$gr
    return( list( theta=theta, r=r, fn=res$fn, gr=res$gr ) )
}

#-   L aufeinanderfolgende Leapfrog-Schritte ( fuer hmc_step)
 
leapfrog_r <- function( theta=NULL, r=NULL, model_fn=NULL, invM=NULL, 
	epsilon=NULL, L=NULL )
{
	#- make the L steps:
    for ( i in 1:L ) {
        step  <- leapfrog_step_r( theta=theta, r=r, model_fn=model_fn, 
        	invM=invM, epsilon=epsilon, direction=1 )
        theta <- step$theta
        r     <- step$r
    }
    return( list( theta=theta, r=r, fn=step$fn ) )
}