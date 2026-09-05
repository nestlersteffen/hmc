
#' Make a list of control arguments
#'
#' @param biter Length of the chain. 
#' @param burnin Number of burnin samples
#' @param badapt1 Number of samples to find optimal stepsizes with dual averaging, defaults to 100
#' @param badapt1 Number of samples to find optimal transition matrix, defaults to 100
#' @param max_depth Determines the number of maximal leapfrog steps in NUTS, defaults to 10, i.e. 2^10 = 1,024 steps
#' @param max_depth_adapt Similar to max_depth but for the warmup-phase, defaults to 10
#' @param epsilon A default stepsize, defaults to 0.10
#' @param L The number of steps in HMC, defaults to 20
#' @param ... some further arguments for dual averaging

#' @export

make_args <- function( biter=1000, burnin=500, badapt1=100, badapt2=100, max_depth=10, 
    max_depth_adapt=10, epsilon=0.10, L=20, delta=0.80, gamma=0.05, t0=10, kappa=0.75 )
{
    
    #- make adapt - list:
    adapt <- list( delta=delta, gamma=gamma, t0=t0, kappa=kappa)
    #- ...
    return( list( biter=biter, 
        burnin=burnin, 
        badapt1=badapt1,
        badapt2=badapt2,
        epsilon=epsilon, L=20,
        max_depth=max_depth, max_depth_adapt=max_depth_adapt, adapt=adapt ) )
}