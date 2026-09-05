
nuts_step_fn_cpp <- function( model_ptr=NULL, M=NULL, invM=NULL, max_depth=NULL )
{
    function( theta, epsilon ) {
        res <- nutstep_cpp( theta=theta, model_ptr=model_ptr, M=M, invM=invM, 
            epsilon=epsilon, max_depth=max_depth )
        list( theta=res$theta, alpha=res$alpha )
    }
}

#' Obtain samples from a NUTS-chain with all subfunctions implemented in C++
#'
#' @param model_ptr A pointer to the model to be estimated. Expects a single argument, the parameter vector. 
#' @param args A list of control arguments, see make_args()
#' @param verbose A boolean argument to see the progress of the sampler, defaults to FALSE
#' @param inits A vector of initial parameter values to start the chain
#' @param find_epsilon A boolean argument to control whether an initial stepsize should be obtained with bracketing, defaults to TRUE
#' @export

nuts_chain_cpp <- function( model_ptr=NULL, args=NULL, verbose=NULL, inits=NULL,find_epsilon=TRUE )
{
  
    #- set args for the NUTS chain:
    biter     <- args$biter
    burnin    <- args$burnin
    badapt1   <- args$badapt1
    badapt2   <- args$badapt2
    nchain    <- args$nchain
    max_depth <- args$max_depth
    max_depth_adapt <- args$max_depth_adapt

    #- should we obtain a reasonable espilon value?
	theta <- inits
    M <- invM <- rep( 1, length( inits ) )
	epsilon <- if ( find_epsilon ) {
		find_reasonable_epsilon_cpp( theta=theta, model_ptr=model_ptr, M=M, invM=invM )
		} else args$epsilon

    print( epsilon )

	#- %%%%%%%%%%%%%%%%%%%%%
	#-     warm-up phase
	#- %%%%%%%%%%%%%%%%%%%%%

	#- warmup-phase 1: initial tuning of epsilon with dual averaging and M = I
    if ( verbose ) print( " ==== start warmup I ==== ")

	step_fn <- nuts_step_fn_cpp( model_ptr=model_ptr, M=M, invM=invM, max_depth=max_depth_adapt )
    phase1  <- dual_averaging( theta0=inits, step_fn=step_fn, n_iter=badapt1, 
    	epsilon_init=epsilon, verbose=verbose )

    # print( phase1$epsilon )

	#-   warmup-phase 2: use tuned epsilon to tune M and invM
    if ( verbose ) print( " ==== start warmup II ==== ")
	
	phase2 <- estimate_mass_matrix( theta0=phase1$theta, step_fn=step_fn, n_iter=badapt2, 
    	epsilon=phase1$epsilon, verbose=verbose )
	M    <- phase2$M
	invM <- phase2$invM

    # print( M )
    # print( invM )
	
    #- warmup-phase 3: another tuning of epsilon with dual averaging and the new M
    if ( verbose ) print( " ==== start warmup III ==== ")

    step_fn <- nuts_step_fn_cpp( model_ptr=model_ptr, M=M, invM=invM, max_depth=max_depth_adapt )
	phase3  <- dual_averaging( theta0=phase2$last_theta, step_fn=step_fn, n_iter=badapt1, 
    	epsilon_init=phase1$epsilon, verbose=verbose )

	epsilon <- phase3$epsilon
	theta   <- phase3$theta

    # print( epsilon )
    # print( theta )
	
	#- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #-   now the true sampling phase
	#- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #- make matrices and then sample...
    samples     <- matrix( 0, nrow=biter, ncol=length(inits) )
    samples[1,] <- theta
    alphas <- steps <- rep( 0, biter )
    n_divergent <- n_divergent_post <- 0
    divergent_samples <- matrix( 0, nrow=biter, ncol=length(inits) )

    if ( verbose ) print( " ==== start sampling ==== ")
      
    for ( nn in 2:biter ) {
        
        #- life signal:
        if ( verbose && nn%%1 == 0 ) print( paste0( "Iteration: ", nn ) )

        #- get proposal:
        tmp_nuts <- nutstep_cpp( theta=samples[nn-1,], model_ptr=model_ptr, M=M, invM=invM, 
            epsilon=epsilon, max_depth=max_depth )
        samples[nn,] <- tmp_nuts$theta
        steps[nn]    <- tmp_nuts$n_steps
        alphas[nn]   <- tmp_nuts$alpha
        n_divergent  <- n_divergent + tmp_nuts$divergent
        divergent_samples[nn,] <- tmp_nuts$divergent_theta
		if ( nn > burnin ) n_divergent_post <- n_divergent_post + tmp_nuts$divergent

    }

    #- %%%%%%%%%%%%%%%%%%%
    #-   output
    #- %%%%%%%%%%%%%%%%%%%

    #- prepare output:
    parms <- samples[-c(1:(burnin)),]

    #- propotion of divergent proposals:
    p_divergent <- n_divergent/( biter )
    p_divergent_post <- n_divergent_post / ( ( biter - burnin ) )
    
    #- output:
    out <- list( parms=parms, steps=steps, alphas=alphas, p_divergent=p_divergent, 
    	p_divergent_post=p_divergent_post, samples=samples, 
        divergent_samples=divergent_samples )
    return( out )

}