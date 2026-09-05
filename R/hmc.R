
#- this function implements a step in HMC
#
#  the code is based on code by Radford M. Neal
#  see http://www.cs.utoronto.ca/~radford/

hmc_step_r <- function( current_theta=NULL, model_fn=NULL, epsilon=NULL, L=NULL, 
	M=NULL, invM=NULL )
{
	
	#- set theta to current position:
	theta <- current_theta
	p     <- length( theta )
	
	#- get impulse:
	r <- rnorm( p, mean=0, sd=sqrt(M) )
	current_r <- r
	current_H <- compute_H_r( theta=current_theta, r=current_r, model_fn=model_fn, invM=invM )

	#- make L leapfrog steps
	lf <- leapfrog_r( theta=theta, r=r, model_fn=model_fn, invM=invM, epsilon=epsilon, L=L)
	theta <- lf$theta
	r <- lf$r

	#- should we accept the proposal?

	# 1. symmetrize
	r <- -r

	# 2. compute energy
	proposed_H <- compute_H_r( r=r, invM=invM, fn_val=lf$fn )

	# 3. accept or reject the state at end of trajectory
	log_alpha <- current_H - proposed_H
	if ( is.nan(log_alpha) || is.na(log_alpha) ) log_alpha <- -Inf
	alpha <- min( 1, exp( log_alpha ) ) 

	# 4. is the proposal divergent?
	divergent <- abs(log_alpha) > 1000

	accept <- 1
	if ( log( runif(1) ) >= log_alpha ) {
		theta  <- current_theta
		accept <- 0
	}

	return( list( theta=theta, log_alpha=log_alpha, alpha=alpha, accept=accept, divergent=divergent ) )
}

#- a wrapper function 

hmc_step_fn_r <- function( model_fn=NULL, L=NULL, M=NULL, invM=NULL )
{
    function( theta, epsilon ) {
        res <- hmc_step_r( current_theta=theta, model_fn=model_fn, epsilon=epsilon,
            L=L, M=M, invM=invM )
        list( theta=res$theta, alpha=res$alpha )
    }
}

#' Obtain samples from a HMC-chain with all subfunctions implemented in R
#'
#' @param model_fn A closure for the model to be estimated. Expects a single argument, the parameter vector. 
#' @param args A list of control arguments, see make_args()
#' @param verbose A boolean argument to see the progress of the sampler, defaults to FALSE
#' @param inits A vector of initial parameter values to start the chain
#' @param find_epsilon A boolean argument to control whether an initial stepsize should be obtained with bracketing, defaults to TRUE
#' @export

hmc_chain_r <- function( model_fn=NULL, args=NULL, verbose=NULL, inits=NULL, find_epsilon=TRUE )
{
  
    #- set args for the HMC chain:
    biter   <- args$biter
    burnin  <- args$burnin
    badapt1 <- args$badapt1
    badapt2 <- args$badapt2
    nchain  <- args$nchain
    L       <- if( !is.null( args$L ) ) args$L else 20

    #- should we obtain a reasonable espilon value?
	theta <- inits
    M <- invM <- rep( 1, length( inits ) )
	epsilon <- if ( find_epsilon ) {
		find_reasonable_epsilon_r( theta=inits, model_fn=model_fn, M=M, invM=invM ) 
		} else args$epsilon

	#- %%%%%%%%%%%%%%%%%%%%%
	#-     warm-up phase
	#- %%%%%%%%%%%%%%%%%%%%%

	#- warmup-phase 1: initial tuning of epsilon with dual averaging and M = I
    if ( verbose ) print( " ==== start warmup I ==== ")

	step_fn <- hmc_step_fn_r( model_fn=model_fn, L=L, M=M, invM=invM )
    phase1  <- dual_averaging( theta0=inits, step_fn=step_fn, n_iter=badapt1, 
    	epsilon_init=epsilon, verbose=verbose )

	#-   warmup-phase 2: use tuned epsilon to tune M and invM
    if ( verbose ) print( " ==== start warmup II ==== ")
	
	phase2 <- estimate_mass_matrix( theta0=phase1$theta, step_fn=step_fn, n_iter=badapt2, 
    	epsilon=phase1$epsilon, verbose=verbose )
	M    <- phase2$M
	invM <- phase2$invM
	
	#- warmup-phase 3: another tuning of epsilon with dual averaging and the new M
    if ( verbose ) print( " ==== start warmup III ==== ")

    step_fn <- hmc_step_fn_r( model_fn=model_fn, L=L, M=M, invM=invM )
	phase3  <- dual_averaging( theta0=phase2$last_theta, step_fn=step_fn, n_iter=badapt1, 
    	epsilon_init=phase1$epsilon, verbose=verbose )

	epsilon <- phase3$epsilon
	theta   <- phase3$theta

	#- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #-   now the true sampling phase
	#- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	if ( verbose ) print( " ==== start sampling ==== ")

    #- make matrices and then sample...
    samples     <- matrix( 0, nrow=biter, ncol=length(inits) )
    samples[1,] <- theta
    accept <- accept_post <- n_divergent <- n_divergent_post <- 0

    for ( nn in 2:biter ) {
        
        #- life signal:
        if ( verbose && nn%%10 == 0 ) {
            print( paste0( "Iteration: ", nn ) )
        }

        #- get proposal:
        tmp_hmc   <- hmc_step_r( current_theta=samples[nn-1,], model_fn=model_fn, 
        	epsilon=epsilon, L=L, M=M, invM=invM )
        samples[nn,] <- tmp_hmc$theta
        accept <- accept + tmp_hmc$accept
		if ( nn > burnin ) accept_post <- accept_post + tmp_hmc$accept 
		n_divergent <- n_divergent + tmp_hmc$divergent
		if ( nn > burnin ) n_divergent_post <- n_divergent_post + tmp_hmc$divergent 
    }

    #- %%%%%%%%%%%%%%%%%%%
    #-   output
    #- %%%%%%%%%%%%%%%%%%%

    #- prepare output:
    parms <- samples[-c(1:(burnin)),]
    
    #- proportion of accepted proposals:
    p_accept <- accept/( biter )
    p_accept_post <- accept_post / ( ( biter - burnin ) )

    #- propotion of divergent proposals:
    p_divergent <- n_divergent/( biter )
    p_divergent_post <- n_divergent_post / ( ( biter - burnin ) )

    #- output:
    out <- list( parms=parms, p_accept=p_accept, p_accept_post=p_accept_post, 
    	p_divergent=p_divergent, p_divergent_post=p_divergent_post, 
    	samples=samples )
    return( out )

}