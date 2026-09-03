
#- this function implements a step in NUTS
#  the code is based on Gelman & Hoffman (2014)

nuts_step_r <- function ( current_theta=NULL, model_fn=NULL, epsilon=NULL,  
	M=NULL, invM=NULL, max_depth=NULL )
{
	
	#- get impulse and H for this impulse:
	p  <- length( current_theta )
	r0 <- rnorm( p, mean=0, sd=sqrt(M) )
  	
  	res0 <- model_fn( current_theta )
  	fn0  <- res0$fn
  	H0   <- compute_H_r( r=r0, invM=invM, fn_val=fn0 )

	#- set thetas and rs
	theta_fwd <- current_theta
	theta_bwd <- current_theta
	r_fwd <- r0
	r_bwd <- r0
	  
	all_theta <- matrix( current_theta, nrow=1 )
	all_r     <- matrix( r0, nrow = 1 )
	all_fn    <- c(fn0)

	#- we make the NUTS steps:
	j <- 0
	batch_divergent <- FALSE
	divergent_theta <- rep(0,p)

	while( TRUE ) {
    
    	# no. of steps:
    	steps <- 2^j
    
        # sample direction:
    	direction <- sample(c(-1, 1), 1)
    
	    # set the start point depending on direction
	    if (direction == 1) {
	        theta <- theta_fwd; r <- r_fwd
		} else {
			theta <- theta_bwd; r <- r_bwd 
		}
    
	    # make leap-frog steps:
	    for ( i in 1:steps) {
		    
		    # make the steps:
		    step  <- leapfrog_step_r( theta=theta, r=r, model_fn=model_fn, invM=invM,
		    	epsilon=epsilon, direction=direction )
            theta <- step$theta
            r     <- step$r

            # look at the current energy:
            H_i <- compute_H_r( r=r, invM=invM, fn_val=step$fn )
            if ( is.na( H_i ) | is.infinite( H_i ) | abs(H_i - H0) > 1000.0 ) {
                batch_divergent <- TRUE;
                divergent_theta <- theta;
                break
            }

		    # save theta dynamically:
		    all_theta <- rbind( all_theta, theta )
		    all_r     <- rbind( all_r, t(r) )
		    all_fn    <- c( all_fn, step$fn )
	    }

	    if ( batch_divergent ) break   
	    
	    # adapt end of trajectorie
	    if (direction == 1) {
	        theta_fwd <- theta; r_fwd <- r 
		} else {
			theta_bwd <- theta; r_bwd <- r 
		}
	    
	    # have we reached the uturn?
	    diff_theta <- theta_fwd - theta_bwd
	    uturn <- sum(diff_theta * r_fwd) < 0 | sum(diff_theta * r_bwd) < 0
		if ( is.na(uturn) || uturn ) break
	    
	    # Sicherheitsgrenze:
	    if ( j >= max_depth ) break
	    
	    # adapt j:
	    j <- j + 1
	  
	}
  
  	#- slice-Variable am Startpunkt ziehen
	log_u <- log( runif(1) ) - as.numeric(H0)

	# Im Leapfrog-Loop: H für jeden Punkt speichern
	H_all <- all_fn + 0.5*( ( all_r^2 ) %*% invM )
	
	# nur akzeptable Punkte: H* <= -log_u
	acceptable <- which( -H_all >= log_u )

	# Fallback falls keine akzeptablen Punkte
	if ( length( acceptable ) == 0) {
	    new_theta <- current_theta
	    new_r     <- r0
	    new_fn    <- fn0
	} else {
	    idx       <- sample(acceptable, 1)
	    new_theta <- all_theta[idx, ]
	    new_r     <- all_r[idx, ]
	    new_fn    <- all_fn[idx]
	}
   	H_proposal <- compute_H_r( r=new_r, invM=invM, fn_val=new_fn )
  
  	# mittlere Akzeptanzwahrscheinlichkeit:
  	alpha <- if ( nrow( all_theta ) > 1 ) mean( pmin( 1, exp( as.numeric( H0 ) - H_all[-1] ) ) ) else 0.0

  	# make a divergence check
	divergent <- batch_divergent | abs( H0 - H_proposal ) > 1000

  	# return result:
  	return( list( theta=new_theta, alpha=alpha, j=j, n_steps=2^j, divergent=divergent, divergent_theta = divergent_theta ) )

}

#- a wrapper function 

nuts_step_fn_r <- function( model_fn=NULL, L=NULL, M=NULL, invM=NULL, max_depth=NULL )
{
    function( theta, epsilon ) {
        res <- nuts_step_r( current_theta=theta, model_fn=model_fn, epsilon=epsilon,
            M=M, invM=invM, max_depth=max_depth )
        list( theta=res$theta, alpha=res$alpha )
    }
}

#- one nuts chain ...

nuts_chain_r <- function( model_fn=NULL, args=NULL, verbose=NULL, inits=NULL,find_epsilon=TRUE )
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
		find_reasonable_epsilon_r( theta=inits, model_fn=model_fn, M=M, invM=invM ) 
		} else args$epsilon

	#- %%%%%%%%%%%%%%%%%%%%%
	#-     warm-up phase
	#- %%%%%%%%%%%%%%%%%%%%%

	#- warmup-phase 1: initial tuning of epsilon with dual averaging and M = I
    if ( verbose ) print( " ==== start warmup I ==== ")

	step_fn <- nuts_step_fn_r( model_fn=model_fn, L=L, M=M, invM=invM, max_depth=max_depth_adapt )
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

    step_fn <- nuts_step_fn_r( model_fn=model_fn, L=L, M=M, invM=invM, max_depth=max_depth_adapt )
	phase3  <- dual_averaging( theta0=phase2$last_theta, step_fn=step_fn, n_iter=badapt1, 
    	epsilon_init=phase1$epsilon, verbose=verbose )

	epsilon <- phase3$epsilon
	theta   <- phase3$theta
	
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
        tmp_nuts <- nuts_step_r( current_theta=samples[nn-1,], model_fn=model_fn, epsilon=epsilon, 
			M=M, invM=invM, max_depth=max_depth )
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